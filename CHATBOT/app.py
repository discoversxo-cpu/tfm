import streamlit as st
import pandas as pd
from streamlit_functions import (
    load_data_from_snowflake,
    build_province_month_features,
    recomendar_alternativas_stream,
    recomendar_actividades,
    get_params_from_chat,
    _geo_mask
)

st.set_page_config(layout="wide", page_title="TFM Asistente de Viajes IA")

# --- Inicialización y Carga de datos ---
if "data_loaded" not in st.session_state:
    with st.spinner('Cargando datos...'):
        st.session_state.experiencias_df, st.session_state.master_forecasted_df = load_data_from_snowflake()
        if st.session_state.master_forecasted_df.empty:
            st.stop()
    with st.spinner('Procesando datos...'):
        st.session_state.province_month_df = build_province_month_features(st.session_state.master_forecasted_df)
    st.session_state.data_loaded = True

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "¡Hola! Soy tu asistente de viajes de TUI. Cuéntame sobre tu próximo viaje para que pueda ayudarte a encontrar el destino perfecto."}]

st.title("Asistente de Viajes con IA")
st.write("Escribe tus preferencias y te ayudaré a encontrar los mejores destinos y actividades para tu viaje.")

# Muestra los mensajes de chat existentes
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# --- Lógica principal del chat ---
if prompt := st.chat_input("¿En qué puedo ayudarte?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    with st.spinner("Pensando..."):
        # Llamar a la función que interactúa con la IA
        params = get_params_from_chat(
            st.session_state.messages,
            st.session_state.province_month_df["province"].unique().tolist(),
            ["HOBBIES AND GAMES", "ARTS AND SOCIETY", "SPORTS AND WELLNESS", "FESTIVALS", "FOOD", "FAMILY"],
            ["playa", "montaña", "urbano", "mixto", "ninguna"]
        )

    # Lógica para manejar la respuesta de la IA
    if isinstance(params, dict):
        # La IA ha devuelto un JSON, lo que significa que tiene todos los parámetros
        modo_val = params.get('modo')
        mes_val = params.get('mes')
        anio_val = params.get('anio')
        provincia_base_val = params.get('provincia_base')
        tourism_type = params.get('turismo')
        selected_events_names = params.get('eventos', [])
        crowd_tol = params.get('multitudes')
        rain_tol = params.get('lluvia')
        budget_tol = params.get('presupuesto')
        target_temp = params.get('temperatura')

        # Si el modo es '1', no necesitamos la provincia base.
        if modo_val == '1':
            provincia_base_val = None
        
        # Validar si el usuario seleccionó eventos pero la lista está vacía
        preference = "Sí" if selected_events_names else "No"

        # Filtrar el DataFrame por la fecha que dio la IA
        period = pd.Period(f"{anio_val}-{mes_val}", freq="M")
        province_month_df_filtered = st.session_state.province_month_df[st.session_state.province_month_df["year_month"] == period].copy()
        
        # Llamar al modelo de recomendación
        top_provincias, alternativas, eventos_recomendados = recomendar_alternativas_stream(
            complete_slice_df=st.session_state.province_month_df,
            features_df=province_month_df_filtered,
            modo=modo_val,
            provincia_base=provincia_base_val,
            preference="s" if preference == "Sí" else "n",
            selected_events=selected_events_names,
            tourism_type=tourism_type,
            target_temp=target_temp,
            crowd_tol=crowd_tol,
            rain_tol=rain_tol,
            budget_tol=budget_tol
        )
        
        # Mostrar los resultados en la interfaz
        if not top_provincias.empty:
            st.session_state.messages.append({"role": "assistant", "content": "¡He encontrado algunas recomendaciones para ti! Aquí están los resultados:"})
            st.chat_message("assistant").write("¡He encontrado algunas recomendaciones para ti! Aquí están los resultados:")
            st.dataframe(top_provincias[["province", "user_score"]].rename(columns={"province": "Provincia", "user_score": "Puntuación de Usuario"}), hide_index=True)
            
            st.subheader("Actividades recomendadas en estos destinos")
            for _, row in top_provincias.iterrows():
                st.markdown(f"**{row['province'].title()}**")
                actividades_top = recomendar_actividades(st.session_state.experiencias_df, row['province'], categorias=eventos_recomendados)
                st.dataframe(actividades_top[["TITULO", "RATING", "LINK"]].rename(columns={"TITULO": "Título", "RATING": "Rating", "LINK": "Link"}), hide_index=True)

            if not alternativas.empty:
                st.subheader("Alternativas menos masificadas")
                st.dataframe(alternativas[["province", "similarity", "user_score"]].rename(columns={"province": "Provincia Alternativa", "similarity": "Similitud", "user_score": "Puntuación de Usuario"}), hide_index=True)
                st.subheader("Actividades en las provincias alternativas")
                for _, row in alternativas.iterrows():
                    st.markdown(f"**{row['province'].title()}**")
                    actividades_alt = recomendar_actividades(st.session_state.experiencias_df, row['province'], categorias=eventos_recomendados)
                    st.dataframe(actividades_alt[["TITULO", "RATING", "LINK"]].rename(columns={"TITULO": "Título", "RATING": "Rating", "LINK": "Link"}), hide_index=True)
            
        else:
            st.session_state.messages.append({"role": "assistant", "content": "Lo siento, no pude encontrar destinos que coincidan con tus preferencias. ¿Te gustaría intentar con otras opciones?"})
            st.chat_message("assistant").write("Lo siento, no pude encontrar destinos que coincidan con tus preferencias. ¿Te gustaría intentar con otras opciones?")
    else:
        # La IA ha devuelto una respuesta conversacional
        st.session_state.messages.append({"role": "assistant", "content": params})
        st.chat_message("assistant").write(params)
