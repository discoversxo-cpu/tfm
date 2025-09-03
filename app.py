import streamlit as st
import pandas as pd
from recommender import (
    load_data_from_snowflake,
    build_province_month_features,
    recomendar_alternativas_stream,
    recomendar_actividades,
    _geo_mask
)

if __name__ == "__main__":
    st.title("Sistema de Recomendaciones de Destinos")
    st.write("¡Bienvenido al sistema de recomendaciones de viajes! Aquí puedes encontrar el destino perfecto para tu próxima aventura.")
    
    # Cargar datos
    with st.spinner('Cargando datos...'):
        experiencias_df, master_forecasted_df = load_data_from_snowflake()

    with st.spinner('Procesando datos...'):
        province_month_df = build_province_month_features(master_forecasted_df)

    # UI para el modo de recomendación
    modo = st.radio(
        "Selecciona un modo de recomendación:",
        ('1) Recomendación automática (por tus gustos)', '2) Selección de provincia base (en base a la que elijas)'),
        index=0,
        key='modo_radio'
    )

    modo_val = '1' if '1' in modo else '2'

    st.subheader("Selecciona tus preferencias de viaje")
    
    # UI para fecha
    anio_val = st.selectbox("Año de viaje:", [2025, 2026], key='anio_select')

    if anio_val == 2025:
        meses_disponibles = list(range(1, 13))  # Julio a Diciembre
        default_index = meses_disponibles.index(7) 
    else: # anio_val == 2026
        meses_disponibles = list(range(1, 7)) # Enero a Junio
        default_index = meses_disponibles.index(1)

    mes_val = st.selectbox("Mes de viaje:", meses_disponibles, index=default_index, key='mes_select')
        
        # --- Slice de periodo ---
    period = pd.Period(f"{anio_val}-{mes_val}", freq="M")
    province_month_df = province_month_df[province_month_df["year_month"] == period].copy()
    complete_slice_df = province_month_df.copy()
    
    # --- UI para modo 2 (Provincia base) ---
    if modo_val == '2':
        provincia_list = sorted(master_forecasted_df["NOMBRE_PRO"].unique().tolist())
        provincia_base_val = st.selectbox(
            "Provincia base:",
            provincia_list,
            index=provincia_list.index("Madrid") if "Madrid" in provincia_list else 0,
            key='provincia_base_select'
        )

        preference = st.radio("¿Tienes preferencia por algún tipo de evento?", ("Sí", "No"), key='preference_radio')
        selected_events_names = []
        if preference == "Sí":
            available_event_cols = ["HOBBIES AND GAMES", "ARTS AND SOCIETY", "SPORTS AND WELLNESS",
                                    "FESTIVALS", "FOOD", "FAMILY"]
            selected_events_names = st.multiselect(
                "Selecciona hasta 3 eventos:",
                options=available_event_cols,
                max_selections=3,
                key='selected_events_multiselect'
            )
        
        if not selected_events_names:
            selected_events_names = []
            preference = "No"

        # Botón de recomendación para modo 2
        if st.button("Recomendar", key='recomendar_btn_modo2'):
            with st.spinner('Buscando alternativas...'):
                top_provincias, alternativas, eventos_recomendados = recomendar_alternativas_stream(
                    complete_slice_df=complete_slice_df,
                    features_df=province_month_df,
                    modo=modo_val,
                    provincia_base=provincia_base_val,
                    preference="s" if preference == "Sí" else "n",
                    selected_events=selected_events_names
                )
            
            if not top_provincias.empty and not alternativas.empty:
                st.success(f"Hemos encontrado las siguientes alternativas a {provincia_base_val}:")
                st.dataframe(alternativas[["province", "similarity", "user_score"]].rename(columns={
                    "province": "Alternativa de Provincia",
                    "similarity": "Similitud",
                    "user_score": "Puntuación de Usuario"
                }), hide_index=True)
                
                st.subheader(f"Actividades recomendadas en {provincia_base_val}")
                actividades_base = recomendar_actividades(experiencias_df, provincia_base_val, categorias=eventos_recomendados)
                st.dataframe(actividades_base[["TITULO", "RATING", "LINK"]].rename(columns={
                    "TITULO": "Título", "RATING": "Rating", "LINK": "Link"
                }), hide_index=True)
                st.subheader("Actividades recomendadas en provincias alternativas")
                for _, row in alternativas.iterrows():
                    st.markdown(f"**{row['province'].title()}**")
                    actividades_alt = recomendar_actividades(experiencias_df, row['province'], categorias=eventos_recomendados)
                    st.dataframe(actividades_alt[["TITULO", "RATING", "LINK"]].rename(columns={
                        "TITULO": "Título", "RATING": "Rating", "LINK": "Link"
                    }), hide_index=True)

    # --- UI para modo 1 (Automático) ---
    else:
        tourism_options = ["playa", "montaña", "urbano", "mixto", "ninguna"]
        tourism_type = st.selectbox("Tipo de turismo:", tourism_options, index=4, key='tourism_type_select')

        if tourism_type in ("playa","montaña","urbano","mixto"):
            mask_geo = _geo_mask(province_month_df["geography"], tourism_type)
            province_month_df = province_month_df[mask_geo]
            if province_month_df.empty:
                st.warning("EMPTY do to tourism_type")

        temp_min = round(province_month_df["mean_temp"].min())
        temp_max = round(province_month_df["mean_temp"].max())
        target_temp = st.slider(
            "Temperatura deseada (ºC):",
            min_value=temp_min, max_value=temp_max, value=(temp_min + temp_max) // 2, key='target_temp_slider'
        )
        
        crowd_tol = st.selectbox("Tolerancia a multitudes:", ["low", "medium", "high"], key='crowd_tol_select')
        rain_tol = st.selectbox("Tolerancia a la lluvia:", ["low", "medium", "high"], key='rain_tol_select')
        budget_tol = st.selectbox("Presupuesto:", ["low", "medium", "high"], key='budget_tol_select')
        
        preference = st.radio("¿Tienes preferencia por algún tipo de evento?", ("Sí", "No"), key='preference_radio')
        selected_events_names = []
        if preference == "Sí":
            available_event_cols = ["HOBBIES AND GAMES", "ARTS AND SOCIETY", "SPORTS AND WELLNESS",
                                    "FESTIVALS", "FOOD", "FAMILY"]
            selected_events_names = st.multiselect(
                "Selecciona hasta 3 eventos:",
                options=available_event_cols,
                max_selections=3,
                key='selected_events_multiselect'
            )

            if not selected_events_names:
                selected_events_names = []
                preference = "No"
        
        if st.button("Recomendar", key='recomendar_btn_modo1'):
            with st.spinner('Buscando destinos...'):
                top_provincias, alternativas, eventos_recomendados = recomendar_alternativas_stream(
                    complete_slice_df=complete_slice_df,
                    features_df=province_month_df,
                    modo=modo_val,
                    preference="s" if preference == "Sí" else "n",
                    selected_events=selected_events_names,
                    tourism_type=tourism_type,
                    target_temp=target_temp,
                    crowd_tol=crowd_tol,
                    rain_tol=rain_tol,
                    budget_tol=budget_tol
                )
            
            if not top_provincias.empty:
                st.success("¡Destinos principales encontrados!")
                st.dataframe(top_provincias[["province", "user_score"]].rename(columns={
                    "province": "Provincia",
                    "user_score": "Puntuación de Usuario"
                }), hide_index=True)

                st.subheader("Actividades recomendadas en estos destinos")
                for _, row in top_provincias.iterrows():
                    st.markdown(f"**{row['province'].title()}**")
                    actividades_top = recomendar_actividades(experiencias_df, row['province'], categorias=eventos_recomendados)
                    st.dataframe(actividades_top[["TITULO", "RATING", "LINK"]].rename(columns={
                        "TITULO": "Título", "RATING": "Rating", "LINK": "Link"
                    }), hide_index=True)

                if not alternativas.empty:
                    st.subheader("Alternativas menos masificadas")
                    st.dataframe(alternativas[["province", "similarity", "user_score"]].rename(columns={
                        "province": "Provincia Alternativa",
                        "similarity": "Similitud",
                        "user_score": "Puntuación de Usuario"
                    }), hide_index=True)

                    st.subheader("Actividades en las provincias alternativas")
                    for _, row in alternativas.iterrows():
                        st.markdown(f"**{row['province'].title()}**")
                        actividades_alt = recomendar_actividades(experiencias_df, row['province'], categorias=eventos_recomendados)
                        st.dataframe(actividades_alt[["TITULO", "RATING", "LINK"]].rename(columns={
                            "TITULO": "Título", "RATING": "Rating", "LINK": "Link"
                        }), hide_index=True)
