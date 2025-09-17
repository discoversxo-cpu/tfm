import streamlit as st
import pandas as pd
import numpy as np
from snowflake.snowpark import Session
import unicodedata
from sklearn.metrics.pairwise import cosine_similarity

SNOWFLAKE_CONFIG = {
    "account": st.secrets["account"],
    "user": st.secrets["user"],
    "password": st.secrets["password"],
    "role": st.secrets["role"],
    "warehouse": st.secrets["warehouse"],
    "database": st.secrets["database"],
    "schema": st.secrets["schema"]
}

def norm_0_1(series: pd.Series) -> pd.Series:
    s = series.astype(float)
    return (s - s.min()) / (s.max() - s.min()) if s.max() > s.min() else 0 * s

def normalize_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    df_norm = pd.DataFrame(index=df.index)
    for c in cols:
        if c in df.columns:
            df_norm[f"{c}_index"] = norm_0_1(df[c])
    return df_norm

def build_province_month_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    column_map = {
        "NOMBRE_CCAA": "ccaa",
        "NOMBRE_PRO": "province",
        "PERIODO": "period",
        "TEMP_MED": "mean_temp",
        "TEMP_MAX": "max_temp",
        "TEMP_MIN": "min_temp",
        "PRECIPITACION": "rain",
        "ADR": "price",
        "PROPORCION_OCUPACION_HABIT": "crowd",
        "PROPORCION_OCUPACION_CAMAS": "occ_beds",
        "PROPORCION_OCUPACION_CAMAS_FINDE": "occ_beds_weekend",
        "ESTABLECIMIENTOS_ABIERTOS": "establishments",
        "NUMERO_DE_HABITACIONES": "n_rooms",
        "NUMERO_DE_CAMAS": "n_beds",
        "EMPLEADOS": "employees",
        "TURISTAS": "tourists",
        "ALTITUD": "altitude",
        "GEOGRAFIA": "geography"
    }

    df = raw_df.rename(columns=column_map).copy()
    df["period_str"] = df["period"].astype(str).str.replace("M", "-") 
    df["year_month"] = pd.to_datetime(df["period_str"], format="%Y-%m", errors="coerce").dt.to_period("M")
    df["geography"] = df["geography"].str.lower()

    event_cols = ["AFICIONES_Y_JUEGOS", "ARTES_Y_SOCIEDAD", "DEPORTES_Y_BIENESTAR",
                  "FESTIVALES", "GASTRONOMIA", "FAMILIA"]
    
    df["events_total"] = df[event_cols].sum(axis=1)

    feature_cols = [
        "province", "year_month",
        "mean_temp", "max_temp", "min_temp", "rain",
        "events_total", *event_cols, "price", "crowd",
        "establishments", "n_rooms", "n_beds", "employees", "tourists",
        "altitude", "geography"
    ]

    features_df = df[feature_cols].dropna(subset=["province", "year_month"])
    numeric_cols = features_df.select_dtypes(include=np.number).columns.tolist()
    features_df_norm = normalize_columns(features_df, numeric_cols)
    features_df = pd.concat([features_df, features_df_norm], axis=1)

    return features_df

@st.cache_data
def load_data_from_snowflake():
    try:
        with Session.builder.configs(SNOWFLAKE_CONFIG).create() as sf_session:
            master_forecasted_df = sf_session.table("TUI_TFM.PROCESSED.MASTER_FORECASTED").to_pandas()
            experiencis_df = sf_session.table("TUI_TFM.PROCESSED.EXP_TUI").to_pandas()
            
            # --- Aplicar tu función inmediatamente ---
            master_forecasted_df = build_province_month_features(master_forecasted_df)
            
            return experiencis_df, master_forecasted_df
    except Exception as e:
        st.error(f"❌ Error al cargar desde Snowflake: {e}")
        return pd.DataFrame(), pd.DataFrame()

def normalize(text: str) -> str:
    return ''.join(
        c for c in unicodedata.normalize("NFD", text)
        if unicodedata.category(c) != "Mn"
    ).lower().strip() 

def validar_provincia(raw_input: str, df_provincias: pd.Series) -> str:
    PROV_ALIASES = {
        "las palmas": "Canarias",
        "palmas": "Canarias",
        "islas canarias": "Canarias",
        "coruna": "Coruña",
        "a coruna": "Coruña",
        "la coruna": "Coruña",
        "baleares": "Islas Baleares",
        "balears": "Islas Baleares",
        "illes balears": "Islas Baleares",
        "balears, illes": "Islas Baleares",
        "Rioja, La": "Rioja",
        "la rioja": "Rioja"
    }
    norm_input = normalize(raw_input)
    prov_map = {normalize(p): p for p in df_provincias.unique()}
    if norm_input in PROV_ALIASES: return PROV_ALIASES[norm_input].lower()
    elif norm_input in prov_map: return prov_map[norm_input].lower()
    else:
        for norm_name, official_name in prov_map.items():
            if norm_input in norm_name: return official_name.lower()
    return raw_input

def _geo_accept_set(t: str) -> set:
    geo_map = {
        "playa": {"playa", "mixto"},
        "montaña": {"montaña", "mixto"},
        "urbano": {"urbano"},
        "mixto": {"playa", "montaña", "mixto"}
    }
    return geo_map.get(t, set())

def _geo_mask(series_geo: pd.Series, tourism: str) -> pd.Series:
    accept = _geo_accept_set(tourism)
    if not accept:
        return pd.Series(True, index=series_geo.index)
    return series_geo.isin(accept)

def categorize(value, series):
    p25, p75 = series.quantile([0.25, 0.75])
    if value <= p25: return "baja"
    elif value <= p75: return "media"
    else: return "alta"

def compute_user_score(df, target_temp, event_weights, crowd, rain, budget):
    # Temp score
    temp_score = np.exp(-((df["mean_temp"] - target_temp).abs() / 4.0))

    # Event score
    event_signal = sum(df.get(ev, pd.Series(0, index=df.index)) * w for ev, w in event_weights.items())
    event_index = norm_0_1(np.log1p(event_signal.astype(float)))

    # Oferta
    offer_cols = ["establishments_index", "n_beds_index", "employees_index"]
    offer_score = df[offer_cols].sum(axis=1)

    # Penalizaciones
    crowd_penalty = 1 - df["crowd_index"]
    price_penalty = 1 - df["price_index"]
    rain_penalty = 1 - df["rain_index"]

    # Pesos de tolerancia
    tol_weights = {
        "crowd": {"baja": 0.5, "media": 0.30, "alta": 0}.get(crowd, 0.30),
        "rain": {"baja": 0.5, "media": 0.30, "alta": 0}.get(rain, 0.30),
        "price": {"baja": 0.5, "media": 0.30, "alta": 0.1}.get(budget, 0.30),
    }

    tol_comp = (
        tol_weights["crowd"] * crowd_penalty +
        tol_weights["price"] * price_penalty +
        tol_weights["rain"] * rain_penalty
    )

    # Score final
    return (
        0.30 * temp_score +
        0.25 * event_index +
        0.20 * offer_score +
        0.25 * tol_comp
    )
    
def _alts_once(df_slice, base_row, sim_matrix, tourism_type, sim_geo_bonus, 
               top_set, target_temp, crowd_delta, temp_dev, user_drop, require_geo=True):
    pos = {p: i for i, p in enumerate(df_slice["province"].tolist())}
    i = pos[base_row["province"]]
    candidates = df_slice.copy()
    sim_base = sim_matrix[i]
    if tourism_type:
        cg = candidates["geography"]
        geo_accept = _geo_accept_set(tourism_type)
        geo_ok = cg.isin(geo_accept)
        sim_adj = sim_base + sim_geo_bonus * geo_ok.astype(float).values
    else:
        sim_adj = sim_base
        geo_ok = pd.Series(True, index=candidates.index)
    mask = (
        (~candidates["province"].isin(top_set)) &
        (candidates["crowd_index"] < base_row["crowd_index"] * crowd_delta) &
        (candidates["user_score"] >= base_row["user_score"] * user_drop) &
        ((candidates["mean_temp"] - float(target_temp)).abs() <= float(temp_dev))
    )
    if require_geo:
        mask = mask & geo_ok
    cand = candidates[mask].copy()
    if cand.empty:
        return cand
    cand["similarity"] = sim_adj[mask]
    return cand.nlargest(3, "similarity")

def _try_relaxations(df, sim_matrix, base, relaxations, already : list, **kwargs):
    for crowd_delta, temp_dev, user_drop, require_geo in relaxations:

        cand = _alts_once(
            df, base, sim_matrix,
            kwargs["tourism_type"], kwargs["sim_geo_bonus"],
            kwargs["top_set"], kwargs["target_temp"],
            crowd_delta, temp_dev, user_drop, require_geo
        )
        cand = cand[~cand["province"].isin(kwargs["top_set"])]
        cand = cand[~cand["province"].isin(already)]
        if not cand.empty:
            return cand
    return pd.DataFrame()

def _progressive_relax(df, sim_matrix, base, already : list, **kwargs):
    crowd_relax, temp_relax, user_drop_relax = 1.8, 3, 0.15
    step, max_iter = 0.05, 20
    for _ in range(max_iter):
        cand = _alts_once(
            df, base, sim_matrix,
            kwargs["tourism_type"], kwargs["sim_geo_bonus"],
            kwargs["top_set"], kwargs["target_temp"],
            crowd_relax, temp_relax, user_drop_relax, require_geo=False
        )
        cand = cand[~cand["province"].isin(kwargs["top_set"])]
        cand = cand[~cand["province"].isin(already)]
        if not cand.empty:
            cand["note"] = "(relaxed)"
            return cand
        crowd_relax *= 1.1
        temp_relax += 1
        user_drop_relax += step
    return pd.DataFrame()

def recomendar_alternativas(
    complete_slice_df: pd.DataFrame, 
    features_df: pd.DataFrame,
    final_json,
    top_n: int = 3,
    min_crowd_delta: float = 0.5,
    max_user_score_drop: float = 0.15,
    max_temp_deviation_c: float = 3.0,
    sim_geo_bonus: float = 0.08,
    ensure_alternatives: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    
    cols = {
    "aficiones y juegos": "Hobbies and Games",
    "artes y sociedad": "Arts and Society",
    "deportes y bienestar": "Sports and Wellness",
    "festivales": "Festivals",
    "gastronomia": "Food",
    "familia": "Family",
    "sin preferencia": "sin preferencia"  # o "UNKNOWN", según prefieras
}
    event_weights = {}
    selected_event_cols = []
    selected_events = []

    if final_json["preferencias_experiencias"] == "sin preferencia":
        event_weights = {"events_total":1.0}
    else:
        cats = final_json["preferencias_experiencias"].split(",")
        selected_events = [cols[e.strip().lower()] for e in cats]
        selected_event_cols = [c.strip().upper().replace(" ", "_") for c in cats]
        weights_base = np.linspace(1.0, 0.5, len(selected_event_cols))
        weights = weights_base / weights_base.sum()
        event_weights = {c:w for c,w in zip(selected_event_cols, weights)}
        
        
    if final_json["modo_chat"] == "2":

        prov_map = {p.lower(): p for p in features_df["province"].unique()}        
        prov_key = prov_map[final_json["provincia_base"].lower()]

        base_row = features_df[features_df["province"] == prov_key].iloc[0]
        tourism_type = base_row["geography"]
        target_temp = float(base_row["mean_temp"])
        crowd = categorize(base_row["crowd_index"], features_df["crowd_index"])
        rain = categorize(base_row["rain_index"], features_df["rain_index"])
        budget = categorize(base_row["price_index"], features_df["price_index"])

        if tourism_type in ("playa","montaña","urbano","mixto"):
            mask_geo = _geo_mask(features_df["geography"], tourism_type)
            features_df = features_df[mask_geo]
            if features_df.empty:
                st.warning("EMPTY do to tourism_type")
                return pd.DataFrame(), pd.DataFrame(), []
        
    else:
        tourism_type = final_json["tipo_geografia"].lower()
        target_temp = float(final_json["temperatura"])
        crowd = final_json["tolerancia_multitudes"].lower()
        rain = final_json["tolerancia_lluvia"].lower()
        budget = final_json["presupuesto"].lower()

    features_df = features_df[(features_df["mean_temp"] - target_temp).abs() <= 4.0]

    features_df["user_score"] = compute_user_score(
        features_df, target_temp, event_weights, crowd, rain, budget
    )

    complete_slice_df["user_score"] = compute_user_score(
        complete_slice_df, target_temp, event_weights, crowd, rain, budget
    )

    if final_json["modo_chat"] == "2":
        top_base = features_df[features_df["province"] == prov_key]
        top_set = {prov_key}
    else:
        top_base = features_df.nlargest(top_n,"user_score")
        top_set = set(top_base["province"])

    columns_to_keep = [c for c in top_base.columns if c.endswith("_index")]
    alt_blocks = []
    relaxations = [
        (0, 0, 0, True),
        (min_crowd_delta, max_temp_deviation_c, max_user_score_drop, True),
        (min_crowd_delta*2, max_temp_deviation_c+1, max_user_score_drop+0.05, True),
        (min_crowd_delta*3, max_temp_deviation_c+2, max_user_score_drop+0.10, True)
    ]

    M_features = features_df[columns_to_keep].to_numpy()
    sim_features = cosine_similarity(np.nan_to_num(M_features, nan=0.0))
    M_complete = complete_slice_df[columns_to_keep].to_numpy()
    sim_complete = cosine_similarity(np.nan_to_num(M_complete, nan=0.0))
    for _, base in top_base.iterrows():
        cand_final = pd.DataFrame()
        cand = _try_relaxations(features_df, sim_features, base, relaxations,
                                already=cand_final["province"].tolist() if not cand_final.empty else [],
                                tourism_type=tourism_type, sim_geo_bonus=sim_geo_bonus,
                                top_set=top_set, target_temp=target_temp)
        if not cand.empty:
            cand_final = pd.concat([cand_final, cand])
        if len(cand_final) < top_n and ensure_alternatives:
            cand = _try_relaxations(complete_slice_df, sim_complete, base, relaxations,
                                    already=cand_final["province"].tolist() if not cand_final.empty else [],
                                    tourism_type=tourism_type, sim_geo_bonus=sim_geo_bonus,
                                    top_set=top_set, target_temp=target_temp)
            if not cand.empty:
                cand_final = pd.concat([cand_final, cand])
        if len(cand_final) < top_n and ensure_alternatives:
            cand = _progressive_relax(complete_slice_df, sim_complete, base,
                                    already=cand_final["province"].tolist() if not cand_final.empty else [],
                                    tourism_type=tourism_type, sim_geo_bonus=sim_geo_bonus,
                                    top_set=top_set, target_temp=target_temp)
            if not cand.empty:
                cand_final = pd.concat([cand_final, cand])
        if not cand_final.empty:
            alt_blocks.append(cand_final)
    if alt_blocks:
        alternatives = pd.concat(alt_blocks, ignore_index=True).drop_duplicates("province")
        alternatives = alternatives.sort_values("similarity", ascending=False).head(top_n).reset_index(drop=True)
    else:
        alternatives = pd.DataFrame()
    if final_json["modo_chat"] == "2":
        if alternatives.empty:
            st.warning("No hay alternativas suficientemente similares a la provincia base.")
            return "", pd.DataFrame(), selected_events
        return base_row["province"], alternatives, selected_events
    else:
        if top_base.empty:
            st.warning("⚠️ No hay resultados para ese mes.")
            return pd.DataFrame(), pd.DataFrame(), selected_events
        if alternatives.empty:
            st.info("No hay alternativas menos masificadas suficientemente similares.")
        return top_base, alternatives, selected_events

def recomendar_actividades(df, provincia, categorias=None, top_n=3):
    df_provincia = df[df["NOMBRE_PRO"] == provincia].copy()
    if df_provincia.empty:
        return pd.DataFrame()
    C = df["RATING"].mean()
    v = df_provincia["REVIEWS_COUNT"].replace(0, np.nan)
    R = df_provincia["RATING"]
    weighted_score = (v / (v + 50)) * R + (50 / (v + 50)) * C
    df_provincia["score"] = weighted_score.fillna(-1).astype(float)
    if categorias:
        categorias = [c.lower() for c in categorias]
        df_provincia["match_cat"] = df_provincia["CATEGORIAS"].apply(
            lambda x: any(cat in str(x).lower() for cat in categorias)
        )
        df_provincia = df_provincia.sort_values(
            by=["match_cat", "score"],
            ascending=[False, False]
        )
    else:
        df_provincia = df_provincia.sort_values(
            by="score",
            ascending=False
        )
    recomendadas = df_provincia.head(top_n)
    return recomendadas[["TITULO", "DESCRIPCION", "CATEGORIAS", "RATING", "REVIEWS_COUNT", "LINK"]]

def unificado(experiencias_df, province_month_df, complete_slice_df, final_json):
    result = recomendar_alternativas(complete_slice_df, province_month_df, final_json)
    top_df, provincia_base, alt_df, selected_events = (
        (result[0], None, result[1], result[2]) if final_json["modo_chat"] == "1"
        else (None, result[0], result[1], result[2])
    )

    top_provincias = top_df["province"].tolist() if top_df is not None and not top_df.empty else []
    alt_provincias = alt_df["province"].tolist() if alt_df is not None and not alt_df.empty else []

    st.markdown("""
    <style>
    .card { border:1px solid #ddd; border-radius:12px; padding:15px; margin-bottom:15px;
           box-shadow: 0 4px 12px rgba(0,0,0,0.15); display:flex; flex-direction:column; justify-content:space-between; transition: transform 0.3s;}
    .card:hover { transform: translateY(-5px); box-shadow: 0 6px 18px rgba(0,0,0,0.2); }
    .descripcion { max-height:100px; overflow:hidden; transition: max-height 0.5s ease; }
    .descripcion.expandida { max-height:1000px; }
    .leer-mas { font-style:italic; text-decoration:underline; font-size:12px; cursor:pointer; color:#0073e6; display:block; margin-top:5px; }
    .ver-mas { background-color:#009879; color:white; padding:5px 8px; border:none; border-radius:6px; font-size:12px;
               text-align:center; text-decoration:none; display:inline-block; margin-top:5px; }

    h2 { color:#ff5722; font-size:28px; font-weight:900; text-transform: uppercase;
         text-shadow: 1px 1px 2px rgba(0,0,0,0.3); margin-bottom:15px; }
    h3 { color:#009688; font-size:22px; font-weight:700; text-transform: uppercase;
         text-shadow: 1px 1px 1px rgba(0,0,0,0.2); margin-bottom:10px; }
    </style>
    <script>
    function toggleDescripcion(id){ var elem = document.getElementById(id); elem.classList.toggle('expandida'); }
    </script>
    """, unsafe_allow_html=True)

    # --- Componente HTML mínimo solo para scroll con altura 1 ---
    import streamlit.components.v1 as components
    components.html("""
        <div id="scroll_ref" style="height:1px; margin-top:-100px;"></div>
        <script>
        (function(){
            const MAX_TRIES = 30;
            const INTERVAL_MS = 300;
            let tries = 0;

            function scrollToEl(el){
                try{
                    // el margin negativo hará que quede un poco por encima
                    el.scrollIntoView({behavior:'smooth', block:'start'});
                }catch(e){}
            }

            function attempt(){
                tries++;
                const el = document.getElementById('scroll_ref');
                if(el){
                    scrollToEl(el);
                    setTimeout(()=>scrollToEl(el), 200); // segundo intento
                    return;
                }
                if(tries < MAX_TRIES){
                    setTimeout(attempt, INTERVAL_MS);
                }
            }

            setTimeout(attempt, 300);
        })();
        </script>
    """, height=1)

    # --- Función para mostrar cards ---
    def mostrar_actividades_cards(provincias, titulo, skip_base=False, primer_titulo=False):
        if primer_titulo:
            st.markdown(f'<h2 id="primer_titulo_actividad">{titulo}</h2>', unsafe_allow_html=True)
        else:
            st.markdown(f'<h2>{titulo}</h2>', unsafe_allow_html=True)

        for prov in provincias:
            if skip_base and provincia_base and prov == provincia_base:
                continue
            if prov.upper() not in experiencias_df["NOMBRE_PRO"].str.upper().values:
                st.warning(f"No hay actividades en {prov}")
                continue

            actividades = recomendar_actividades(
                experiencias_df, prov, categorias=selected_events, top_n=3
            )
            if actividades.empty:
                st.info(f"No hay actividades encontradas en {prov}")
                continue

            st.markdown(f'<h3>{prov}</h3>', unsafe_allow_html=True)
            cols = st.columns(len(actividades))
            for i, (_, row) in enumerate(actividades.iterrows()):
                card_id = f"desc_{prov}_{i}"
                with cols[i]:
                    st.markdown(f"""
                        <div class="card">
                            <h4>{row['TITULO']}</h4>
                            <p style="font-style:italic; color:#555;">{row['CATEGORIAS']}</p>
                            <div id="{card_id}" class="descripcion">
                                {row['DESCRIPCION']}
                                <span class="leer-mas" onclick="toggleDescripcion('{card_id}')">Leer más</span>
                            </div>
                            <p>⭐ {row['RATING'] if pd.notna(row['RATING']) else ''} ({row['REVIEWS_COUNT']} reviews)</p>
                            <a href="{row['LINK']}" target="_blank" class="ver-mas">Ver más</a>
                        </div>
                    """, unsafe_allow_html=True)

    # Mostrar actividades principales
    if provincia_base:
        mostrar_actividades_cards([provincia_base], f"ACTIVIDADES EN {provincia_base.upper()}", primer_titulo=True)
    else:
        mostrar_actividades_cards(top_provincias, "ACTIVIDADES EN PROVINCIAS PRINCIPALES", primer_titulo=True)

    # Mostrar alternativas
    if alt_df is not None and not alt_df.empty:
        mostrar_actividades_cards(
            alt_provincias,
            "ACTIVIDADES EN PROVINCIAS MENOS CONCURRIDAS",
            skip_base=True
        )
