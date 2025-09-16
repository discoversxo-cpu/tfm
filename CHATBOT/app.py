import streamlit as st
import traceback
import requests
import json
import time
from streamlit_functions import load_data_from_snowflake, unificado, _geo_mask, validar_provincia
import pandas as pd

# --- Cargar datos desde Snowflake ---
experiencias_df, master_forecasted_df = load_data_from_snowflake()

if experiencias_df.empty and  master_forecasted_df.empty:
    st.warning("⚠️ No se pudieron cargar los datos")

if "experiencias_df" not in st.session_state:
    st.session_state.experiencias_df = experiencias_df
    st.session_state.master_forecasted_df = master_forecasted_df

if "province_month_df" not in st.session_state:
    st.session_state.province_month_df = pd.DataFrame()

if "complete_slice_df" not in st.session_state:
    st.session_state.complete_slice_df = pd.DataFrame()

OLLAMA_API_URL = "http://localhost:11434"
MODEL = "gemma3:4b"

st.set_page_config(page_title="Chat con Discover", page_icon="💬")
st.title("💬 Chatea con Discover")

# --- Instrucciones para el usuario ---
st.markdown("""
<div style='background-color:#e0f7fa; padding:15px; border-radius:12px; margin-bottom:15px; font-size:14px; line-height:1.5;'>
<b>🌟 Bienvenido al Chat de Viajes 🌟</b><br><br>
✈️ Responde a las preguntas para planear tu viaje de manera fácil y divertida.<br>
💡 Puedes escribir tu respuesta directamente o elegir las opciones sugeridas.<br>
📝 El chat recordará tus elecciones y te guiará paso a paso hasta completar tu plan de viaje.<br>
🎉 ¡Disfruta la experiencia y prepárate para descubrir tu próxima aventura!<br><br>
🔄 Si quieres planear <b>otro viaje</b>, simplemente recarga la página.
</div>
""", unsafe_allow_html=True)

# --- CSS moderno tipo app de mensajería ---
st.markdown("""
<style>
/* Fondo de la página */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(to bottom, #e0f7fa, #ffffff);
    color: #333333;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}

/* Título principal */
h1, h2, h3, h4 {
    color: #00796b;
    font-weight: 700;
}

/* Contenedor de chat */
div[data-testid="stChatMessageList"] {
    max-height: 65vh;
    overflow-y: auto;
    padding: 10px;
}

/* Mensajes del usuario */
.user-msg {
    background-color: #80cbc4;
    color: white;
    text-align: right;
    padding: 12px 20px;
    border-radius: 25px 25px 0px 25px;
    margin: 8px 0;
    display: inline-block;
    box-shadow: 0 2px 6px rgba(0,0,0,0.2);
    max-width: 80%;
}

/* Mensajes del asistente */
.assistant-msg {
    background-color: #ffffff;
    color: #333333;
    text-align: left;
    padding: 12px 20px;
    border-radius: 25px 25px 25px 0px;
    margin: 8px 0;
    display: inline-block;
    box-shadow: 0 2px 6px rgba(0,0,0,0.15);
    max-width: 80%;
}

/* Mensajes del asistente en estado “pending” */
.assistant-msg.pending {
    background-color: #c8e6c9;
    color: #1b5e20;
    font-style: italic;
}

/* Input de chat */
div.stTextInput > div > input {
    border-radius: 20px !important;
    padding: 12px 16px !important;
    border: 1px solid #ccc;
    width: 80% !important;
}

/* Botón de enviar */
div.stButton > button {
    background-color: #00796b !important;
    color: white !important;
    border-radius: 20px !important;
    padding: 10px 20px !important;
    font-weight: bold;
    margin-left: 10px;
}
div.stButton > button:hover {
    background-color: #004d40 !important;
}

/* Scrollbar estilo moderno */
div[data-testid="stChatMessageList"]::-webkit-scrollbar {
    width: 8px;
}
div[data-testid="stChatMessageList"]::-webkit-scrollbar-thumb {
    background-color: #00796b;
    border-radius: 4px;
}
div[data-testid="stChatMessageList"]::-webkit-scrollbar-track {
    background: transparent;
}
</style>
""", unsafe_allow_html=True)

# --- Inicializar variables de sesión ---
if "json_data" not in st.session_state:
    st.session_state.json_data = {}

if "current_field" not in st.session_state:
    st.session_state.current_field = "anio"

if "messages" not in st.session_state:
    st.session_state.messages = []

if "started" not in st.session_state:
    st.session_state.started = False

# --- Función para definir opciones dinámicas de mes ---
def mes_opciones():
    if st.session_state.json_data.get("anio") == "2025":
        return ["julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre"]
    else:
        return ["enero", "febrero", "marzo", "abril", "mayo", "junio"]
    
def provincia_opciones():
    if "complete_slice_df" in st.session_state:
        df = st.session_state.province_month_df
        return sorted(df["province"].dropna().unique()) + ["sin preferencia"]
    return []

def temperatura_opciones():
    if "complete_slice_df" in st.session_state:
        df = st.session_state.province_month_df
        temp_min = round(df["mean_temp"].min())
        temp_max = round(df["mean_temp"].max())
        return [str(i) for i in range(int(temp_min), int(temp_max)+1)] + ["sin preferencia"]
    return []

def mean_temp():
    if "complete_slice_df" in st.session_state:
        df = st.session_state.province_month_df
        temp_min = round(df["mean_temp"].min())
        temp_max = round(df["mean_temp"].max())
        return str(round((temp_max+temp_min)/2))



# --- Definir flujo de campos y opciones ---
fields_info = {
    "anio": {"options": ["2025", "2026"], "prompt": "¿En qué año quieres viajar?"},
    "mes": {"options": mes_opciones, "prompt": "¿En qué mes planeas viajar?"},
    "preferencias_experiencias": {
        "options": ["aficiones y juegos", "artes y sociedad", "deportes y bienestar",
                    "festivales", "gastronomia", "familia", "sin preferencia"],
        "prompt": "Elige hasta 3 experiencias que te interesen"
    },
    "provincia_base": {"options": provincia_opciones, "prompt": "¿En qué provincia te gustaría viajar?"},
    "tipo_geografia": {"options": ["playa", "montaña", "urbano", "mixto","sin preferencia"], "prompt": "¿Qué tipo de geografia prefieres?"},
    "temperatura": {"options": temperatura_opciones, "prompt": "¿Qué clima prefieres?"},
    "tolerancia_multitudes": {"options": ["baja", "media", "alta"], "prompt": "¿Qué tolerancia tienes a las multitudes?"},
    "tolerancia_lluvia": {"options": ["baja", "media", "alta"], "prompt": "¿Qué tolerancia tienes a la lluvia?"},
    "presupuesto": {"options": ["baja", "media", "alta"], "prompt": "¿Cuál es tu nivel de presupuesto?"},
}

# --- Normas de normalización por campo ---
normalization_rules = {
    "anio": (
        "Busca si la respuesta contiene un año válido (2025 o 2026). "
        "Si no se puede interpretar, devuelve 'sin preferencia'. "
        "Devuelve únicamente el año o 'sin preferencia'."),
    "mes": (
        "Busca si la respuesta corresponde a un mes. "
        "Si no coincide, devuelve 'sin preferencia'."),
    "preferencias_experiencias": (
        "El usuario puede mencionar uno o varias experiencias (máximo 3). "
        "Mapea cada uno a solo una opcion de las siguientes: aficiones y juegos, artes y sociedad, "
        "deportes y bienestar, festivales, gastronomia, familia. "
        "Devuelve únicamente los que mencionó, en minúsculas y separados por comas. "
        "Si no hay coincidencias o no tiene preferencia, devuelve 'sin preferencia'."),
    "provincia_base": (
        "Busca si el usuario menciona una provincia española válida. "
        "No inventes provincias. "
        "Si no se puede mapear claramente o el usuario no tiene preferencia, devuelve 'sin preferencia'."
        "Devuelve solo el nombre en minúsculas o 'sin preferencia'."),
    "tipo_geografia": (
        "Considera los siguientes valores: playa, montaña, urbano o mixto. "
        "Mapea la respuesta del usuario a uno de esos valores. "
        "Si no se puede mapear claramente o el usuario no tiene preferencia, devuelve 'sin preferencia'."),
    "temperatura": (
        f"Mapea la respuesta del usuario a uno de los siguientes valores: {temperatura_opciones}. "
        "Si no se puede mapear claramente a uno de esos valores o el usuario no tiene preferencia, devuelve 'sin preferencia'."),
    "tolerancia_multitudes": "Analiza todo el contexto de la respuesta del usuario y siempre mapea la respuesta a alguna de estas opciones: baja, media o alta.",
    "tolerancia_lluvia": "Analiza todo el contexto de la respuesta del usuario y siempre mapea la respuesta a alguna de estas opciones: baja, media o alta.",
    "presupuesto": "Analiza todo el contexto de la respuesta del usuario y siempre mapea la respuesta a alguna de estas opciones: baja, media o alta."
}

# --- Prompt inicial conciso ---
SYSTEM_PROMPT = (
    "Eres un asistente de viaje especializado en turismo dentro de españa: "
    "Reglas generales: "
    "1) Interpreta siempre el contexto completo de la respuesta del usuario, "
    "2) Extrae el valor relevante y normalízalo en minúsculas. "
    "3) Maneja errores ortográficos y ajusta al valor válido más cercano. "
    "4) Si no se puede mapear claramente o el usuario no tiene preferencia, devuelve 'sin preferencia'."
    "5) Devuelve solo el valor, sin explicaciones ni formato extra."
)

# --- Mostrar historial con burbujas modernas ---
for msg in st.session_state.messages:
    role = msg["role"]
    with st.chat_message(role):
        css_class = "user-msg" if role=="user" else "assistant-msg"
        st.markdown(f"<div class='{css_class}'>{msg['content']}</div>", unsafe_allow_html=True)

if st.session_state.current_field is None and "final_json" in st.session_state:

    unificado(
        experiencias_df,
        st.session_state.province_month_df,
        st.session_state.complete_slice_df,
        st.session_state.final_json
    )

# --- Mostrar input solo si quedan campos por llenar ---
if not("final_json" in st.session_state):
    if prompt := st.chat_input("Escribe tu mensaje:"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(f"<div class='user-msg'>{prompt}</div>", unsafe_allow_html=True)

    # --- Mostrar animación de “bot escribiendo” ---
    with st.chat_message("assistant"):
        writing_placeholder = st.empty()
        writing_placeholder.markdown("<div class='assistant-msg pending'>Escribiendo...</div>", unsafe_allow_html=True)
        time.sleep(0.5)

    # --- Primer mensaje: saludo inicial ---
    if not st.session_state.started:
        answer = f"¡Hola! Encantado de ayudarte a planear tu viaje. {fields_info[st.session_state.current_field]['prompt']}"
        st.session_state.messages.append({"role": "assistant", "content": answer})
        writing_placeholder.markdown(
            f"<div class='assistant-msg'>{answer}</div>",
            unsafe_allow_html=True
        )
        st.session_state.started = True
    else:
        current_field = st.session_state.current_field
        rules = normalization_rules.get(current_field, "")

        messages_to_send = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": (
                f"JSON parcial: {json.dumps(st.session_state.json_data, ensure_ascii=False)}. "
                f"El usuario respondió para el campo '{current_field}': '{prompt}'. "
                f"{rules} Devuelve únicamente el valor normalizado para este campo en minúsculas."
            )}
        ]

        try:
            response = requests.post(
                f"{OLLAMA_API_URL}/v1/chat/completions",
                json={"model": MODEL, "messages": messages_to_send}
            )
            
            normalized_value = response.json()["choices"][0]["message"]["content"].strip().lower()

            if current_field == "provincia_base":
                normalized_value = validar_provincia(normalized_value, st.session_state.province_month_df["province"])

            elif current_field == "temperatura" and normalized_value == "sin preferencia":
                    normalized_value = mean_temp()
            st.text(normalized_value)
            # --- Validación de opciones ---
            field_options = fields_info[current_field]["options"]
            if callable(field_options):
                field_options = field_options()
            field_options_lower = [opt.lower() for opt in field_options] if field_options else None

            if current_field == "preferencias_experiencias":
                # Separar lo que devolvió el modelo en lista
                valores = [v.strip() for v in normalized_value.split(",") if v.strip()]
                # Filtrar solo los que están en las opciones válidas
                valid_values = [v for v in valores if v in field_options_lower]

                if not valid_values:
                    # Ningún valor válido → repreguntar inmediatamente mostrando opciones
                    options_str = ", ".join(field_options_lower)
                    answer = (
                        f"Disculpa, no pude entenderte bien. "
                        f"Estas son las opciones con las que contamos: {options_str}."
                    )
                    # Mantener el campo actual para que el usuario vuelva a responder
                    st.session_state.current_field = current_field
                    # Mostrar la repregunta en el chat inmediatamente
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    writing_placeholder.markdown(
                        f"<div class='assistant-msg'>{answer}</div>",
                        unsafe_allow_html=True
                    )
                    # Salir de la ejecución para que el flujo no avance hasta que el usuario responda
                    st.stop()

                else:
                    # Guardar los experiencias válidas como lista separada por comas
                    st.session_state.json_data[current_field] = ", ".join(valid_values[:3])
                    # --- Avanzar al siguiente campo (igual que antes) ---
                    fields_order = list(fields_info.keys())
                    next_field = None
                    for f in fields_order:
                        if f not in st.session_state.json_data:
                            next_field = f
                            break
                    st.session_state.current_field = next_field
                    if next_field:
                        next_question_prompt = (
                            f"JSON parcial: {json.dumps(st.session_state.json_data, ensure_ascii=False)}. "
                            f"El siguiente campo a llenar es '{next_field}'. "
                            "Formula una pregunta natural al usuario para obtener este campo."
                        )
                        messages_to_send = [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": next_question_prompt}
                        ]
                        response = requests.post(
                            f"{OLLAMA_API_URL}/v1/chat/completions",
                            json={"model": MODEL, "messages": messages_to_send}
                        )
                        answer = response.json()["choices"][0]["message"]["content"]
                    else:
                        answer = json.dumps(st.session_state.json_data, ensure_ascii=False, indent=2)

            elif field_options_lower and normalized_value not in field_options_lower and current_field == "provincia_base":

                options_str = ", ".join(field_options_lower)
                answer = (
                    f"Disculpa, No pude entenderte bien. "
                    f"Por favor elige una de las 52 provincias de españa. "
                )
                st.session_state.current_field = current_field
                # Mostrar inmediatamente en el chat
                st.session_state.messages.append({"role": "assistant", "content": answer})
                writing_placeholder.markdown(
                    f"<div class='assistant-msg'>{answer}</div>",
                    unsafe_allow_html=True
                )
                # Detener la ejecución para esperar la respuesta del usuario
                st.stop()

            elif field_options_lower and normalized_value not in field_options_lower:
                options_str = ", ".join(field_options_lower)
                answer = (
                    f"Disculpa, no pude entenderte bien. "
                    f"Estas son las opciones con las que contamos: {options_str}."
                )
                st.session_state.current_field = current_field
                # Mostrar inmediatamente en el chat
                st.session_state.messages.append({"role": "assistant", "content": answer})
                writing_placeholder.markdown(
                    f"<div class='assistant-msg'>{answer}</div>",
                    unsafe_allow_html=True
                )
                # Detener la ejecución para esperar la respuesta del usuario
                st.stop()

            else:

                st.session_state.json_data[current_field] = normalized_value

                anio_val = st.session_state.json_data.get("anio")
                mes_val = st.session_state.json_data.get("mes")
                geografia_val = st.session_state.json_data.get("tipo_geografia")
                temp_val = st.session_state.json_data.get("temperatura")

                if anio_val and mes_val:

                    meses_dict = {
                        "enero":1, "febrero":2, "marzo":3, "abril":4, "mayo":5, "junio":6,
                        "julio":7, "agosto":8, "septiembre":9, "octubre":10, "noviembre":11, "diciembre":12
                    }
                    mes_num = meses_dict[mes_val.lower()]

                    period = pd.Period(f"{anio_val}-{mes_num}", freq="M")

                    filtered_df = master_forecasted_df[master_forecasted_df["year_month"] == period].copy()

                    st.session_state.province_month_df = filtered_df.copy()
                    st.session_state.complete_slice_df = filtered_df.copy()

                elif geografia_val:
                    mask_geo = _geo_mask(st.session_state.province_month_df["geography"], geografia_val)
                    province_month_df = st.session_state.province_month_df[mask_geo]
                    if province_month_df.empty:
                        st.warning("EMPTY do to tourism_type") 
                    
                elif temp_val:
                    st.session_state.province_month_df = st.session_state.province_month_df[st.session_state.province_month_df["mean_temp"] <= int(temp_val)].copy()

                # --- Verificar provincia_base y asignar modo_chat ---
                if current_field == "provincia_base":
                    if normalized_value != "sin preferencia":
                        st.session_state.json_data["modo_chat"] = "2"
                    else:
                        st.session_state.json_data["modo_chat"] = "1"

                # --- Si modo_chat = 2, completar JSON y finalizar ---
                if st.session_state.json_data.get("modo_chat") == "2" and current_field == "provincia_base":
                    for field in fields_info:
                        if field not in st.session_state.json_data:
                            if field in ["tipo_geografia", "temperatura", "tolerancia_multitudes", "tolerancia_lluvia", "presupuesto"]:
                                st.session_state.json_data[field] = "sin preferencia"
                    st.session_state.current_field = None

                    st.session_state.final_json = st.session_state.json_data.copy()

                    answer = """✅ He guardado toda la información de tu viaje. Ahora puedo recomendarte experiencias personalizadas."""
                else:
                    # Avanzar al siguiente campo
                    fields_order = list(fields_info.keys())
                    next_field = None
                    for f in fields_order:
                        if f not in st.session_state.json_data:
                            next_field = f
                            break
                    st.session_state.current_field = next_field

                    if next_field:
                        next_question_prompt = (
                            f"JSON parcial: {json.dumps(st.session_state.json_data, ensure_ascii=False)}. "
                            f"El siguiente campo a llenar es '{next_field}'. "
                            "Formula una pregunta natural al usuario para obtener este campo."
                        )
                        messages_to_send = [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": next_question_prompt}
                        ]
                        response = requests.post(
                            f"{OLLAMA_API_URL}/v1/chat/completions",
                            json={"model": MODEL, "messages": messages_to_send}
                        )
                        answer = response.json()["choices"][0]["message"]["content"]
                    else:

                        st.session_state.final_json = st.session_state.json_data.copy()

                        answer = """✅ He guardado toda la información de tu viaje. 
                        Ahora puedo recomendarte experiencias personalizadas."""

            st.session_state.messages.append({"role": "assistant", "content": answer})
            writing_placeholder.markdown(
                f"<div class='assistant-msg'>{answer}</div>",
                unsafe_allow_html=True
            )

            if st.session_state.current_field is None and "final_json" in st.session_state:

                st.rerun()

        except Exception as e:
            st.error(f"❌ Ocurrió un error: {e}")
            st.text("Traceback completo:")
            st.text(traceback.format_exc())
