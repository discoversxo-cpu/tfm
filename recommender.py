import pandas as pd
import numpy as np
import streamlit as st
from snowflake.snowpark import Session
from sklearn.metrics.pairwise import cosine_similarity
import unicodedata

@st.cache_data
def load_data_from_snowflake():
    """Abre sesión, descarga tablas necesarias y cierra la sesión."""
    try:
        # Cargar la configuración desde el archivo secrets.toml
        snowflake_config = st.secrets["snowflake"]
        with Session.builder.configs(snowflake_config).create() as sf_session:
            st.info("✅ Conectado a Snowflake")
            master_forecasted_df = sf_session.table("TUI_TFM.PROCESSED.MASTER_FORECASTED").to_pandas()
            experiencis_df = sf_session.table("TUI_TFM.PROCESSED.EXP_TUI").to_pandas()
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

        "las palmas": "Palmas, Las",
        "palmas": "Palmas, Las",
        "coruna": "Coruña, A",
        "a coruna": "Coruña, A",
        "la coruna": "Coruña, A",
        "baleares": "Balears, Illes",
        "balears": "Balears, Illes",
        "islas baleares": "Balears, Illes",
        "illes balears": "Balears, Illes",
        "la rioja": "Rioja, La",
        "rioja": "Rioja, La"

    }

    norm_input = normalize(raw_input)
    prov_map = {normalize(p): p for p in df_provincias.unique()}

    if norm_input in PROV_ALIASES: return PROV_ALIASES[norm_input].lower()

    elif norm_input in prov_map: return prov_map[norm_input].lower()
    
    else:

        for norm_name, official_name in prov_map.items():

            if norm_input in norm_name: return official_name.lower()

    raise ValueError(f"Provincia '{raw_input}' no encontrada")

def norm_0_1(series: pd.Series) -> pd.Series:

    s = series.astype(float)
    return (s - s.min()) / (s.max() - s.min()) if s.max() > s.min() else 0 * s

def normalize_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:

    df_norm = pd.DataFrame(index=df.index)

    for c in cols:

        if c in df.columns:

            df_norm[f"{c}_index"] = norm_0_1(df[c])

    return df_norm

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

    if value <= p25:
        return "low"
    elif value <= p75:
        return "medium"
    else:
        return "high"
    
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
        (candidates["crowd_index"] < base_row["crowd_index"] + float(crowd_delta)) &
        (candidates["user_score"] >= base_row["user_score"] * (1.0 - float(user_drop))) &
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
        # filtra top_set y lo ya agregado
        cand = cand[~cand["province"].isin(kwargs["top_set"])]
        cand = cand[~cand["province"].isin(already)]
        if not cand.empty:
            return cand
    return pd.DataFrame()


def _progressive_relax(df, sim_matrix, base, already : list, **kwargs):
    crowd_relax, temp_relax, user_drop_relax = 0.2, 3, 0.15
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

@st.cache_data
def build_province_month_features(raw_df: pd.DataFrame) -> pd.DataFrame:

    column_map = {

        "NOMBRE_CCAA": "ccaa",
        "NOMBRE_PRO": "province", "PERIODO": "period",
        "TEMP_MED": "mean_temp", "TEMP_MAX": "max_temp", "TEMP_MIN": "min_temp",
        "PRECIPITACION": "rain", "ADR": "price",
        "PROPORCION_OCUPACION_HABIT": "crowd",
        "PROPORCION_OCUPACION_CAMAS": "occ_beds",
        "PROPORCION_OCUPACION_CAMAS_FINDE": "occ_beds_weekend",
        "ESTABLECIMIENTOS_ABIERTOS": "establishments",
        "NUMERO_DE_HABITACIONES": "n_rooms", "NUMERO_DE_CAMAS": "n_beds",
        "EMPLEADOS": "employees", "TURISTAS": "tourists",
        "ALTITUD": "altitude",
        "GEOGRAFIA": "geography"

    }

    df = raw_df.rename(columns=column_map).copy()

    df["period_str"] = df["period"].astype(str).str.replace("M", "-") 
    df["year_month"] = pd.to_datetime(df["period_str"], format="%Y-%m", errors="coerce").dt.to_period("M")

    df["geography"] = df["geography"].str.lower()

    event_cols = ["HOBBIES_AND_GAMES", "ARTS_AND_SOCIETY", "SPORTS_AND_WELLNESS", "FESTIVALS", "FOOD", "FAMILY"]
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

def recomendar_alternativas_stream(
        
    complete_slice_df: pd.DataFrame, 
    features_df: pd.DataFrame, 
    modo: str,
    preference: str,
    selected_events: list[str],
    provincia_base: str = None,
    tourism_type: str = None,
    target_temp: int = None,
    crowd_tol: str = None,
    rain_tol: str = None,
    budget_tol: str = None,
    top_n: int = 3,
    min_crowd_delta: float = 0.05,
    max_user_score_drop: float = 0.15,
    max_temp_deviation_c: float = 3.0,
    sim_geo_bonus: float = 0.08,
    ensure_alternatives: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:

    event_weights = {}
    selected_event_cols = []

    if preference == "s":

        selected_event_cols = [c.replace(" ", "_") for c in selected_events]
        weights_base = np.linspace(1.0,0.5,len(selected_events))
        weights = weights_base / weights_base.sum()
        event_weights = {c:w for c,w in zip(selected_event_cols,weights)}

    else:

        event_weights = {"events_total":1.0}

    if modo == "2":
        
        prov_key = provincia_base
        base_row = features_df[features_df["province"] == prov_key].iloc[0]

        tourism_type = base_row["geography"]
        target_temp = base_row["mean_temp"]
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

        crowd = crowd_tol
        rain = rain_tol
        budget = budget_tol


    features_df = features_df[(features_df["mean_temp"] - target_temp).abs() <= 4.0]

    if features_df.empty:

        st.warning("EMPTY do to temperature")
        return pd.DataFrame(), pd.DataFrame(), []

    event_signal = sum(features_df.get(ev, pd.Series(0, index=features_df.index)) * w for ev, w in event_weights.items())
    
    offer_cols = ["establishments", "n_beds", "employees"]
    
    event_index = norm_0_1(np.log1p(event_signal.astype(float)))
    temp_score = np.exp(-((features_df["mean_temp"] - float(target_temp)).abs() / 4.0))
    offer_score = norm_0_1(features_df[offer_cols].sum(axis=1))
    
    crowd_penalty = (1 - features_df["crowd_index"])
    price_penalty = (1 - features_df["price_index"])
    rain_penalty = (1 - features_df["rain_index"])
    
    tol_weights = {
        "crowd":{"low":0.5,"medium":0.30,"high":0}.get(crowd,0.30),
        "rain":{"low":0.5,"medium":0.30,"high":0}.get(rain,0.30),
        "price":{"low":0.5,"medium":0.30,"high":0.1}.get(budget,0.30)
    }
    
    tol_comp = (
        tol_weights["crowd"] * crowd_penalty +
        tol_weights["price"] * price_penalty +
        tol_weights["rain"] * rain_penalty
    )
    
    base_user_score = (
        0.30 * temp_score +
        0.25 * event_index +
        0.25 * tol_comp +
        0.20 * offer_score
    )
    
    features_df["user_score"] = base_user_score
    complete_slice_df["user_score"] = base_user_score

    if modo == "2":

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

    if modo == "2":
        
        if alternatives.empty:
            
            st.warning("No hay alternativas suficientemente similares a la provincia base.")
            return pd.DataFrame(), pd.DataFrame(), selected_events
        
        return top_base, alternatives, selected_events
    
    else:
     
        if top_base.empty:
            
            st.warning("⚠️ No hay resultados para ese mes.")
            return pd.DataFrame(), pd.DataFrame(), selected_events
        
        if alternatives.empty:
            
            st.info("No hay alternativas menos masificadas suficientemente similares.")
        
        return top_base, alternatives, selected_events

def recomendar_actividades(df, provincia, categorias=None, top_n=3):
    # Filtrar por provincia
    df_provincia = df[df["NOMBRE_PRO"] == provincia].copy()

    # Calcular score ponderado
    C = df["RATING"].mean()
    v = df_provincia["REVIEWS_COUNT"].replace(0, np.nan)
    R = df_provincia["RATING"]
    weighted_score = (v / (v + 50)) * R + (50 / (v + 50)) * C
    df_provincia["score"] = weighted_score.fillna(-1).astype(float)

    if categorias:
        categorias = [c.lower() for c in categorias]

        # Nueva columna indicando si pertenece a alguna categoría
        df_provincia["match_cat"] = df_provincia["CATEGORIAS"].apply(
            lambda x: any(cat in str(x).lower() for cat in categorias)
        )

        # Ordenar: primero los que hacen match, luego por score
        df_provincia = df_provincia.sort_values(
            by=["match_cat", "score"],
            ascending=[False, False]
        )
    else:
        # Ordenar solo por score
        df_provincia = df_provincia.sort_values(
            by="score",
            ascending=False
        )

    # Seleccionar top_n
    recomendadas = df_provincia.head(top_n)

    return recomendadas[["NOMBRE_PRO", "TITULO", "RATING", "REVIEWS_COUNT", "score", "LINK"]]


