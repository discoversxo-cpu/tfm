import streamlit as st 
import requests
import json

OLLAMA_API_URL = "http://localhost:11434"
MODEL = "gemma3:4b"

st.set_page_config(page_title="Chat con Ollama", page_icon="💬")
st.title("💬 Chat con Ollama")

# --- Inicializar variables de sesión ---
if "json_data" not in st.session_state:
    st.session_state.json_data = {}

if "current_field" not in st.session_state:
    st.session_state.current_field = "anio"

if "messages" not in st.session_state:
    st.session_state.messages = []

if "started" not in st.session_state:
    st.session_state.started = False

# --- Definir flujo de campos y opciones ---
fields_info = {
    "anio": {"options": ["2025", "2026"], "prompt": "¿En qué año planeas tu viaje?"},
    "mes": {"options": None, "prompt": "¿En qué mes planeas viajar?"},
    "preferencias_eventos": {
        "options": ["Aficiones y Juegos", "Artes y Sociedad", "Deportes y Bienestar",
                    "Festivales", "Gastronomía", "Familia"],
        "prompt": "Elige hasta 3 eventos que te interesen"
    },
    "provincia_base": {"options": None, "prompt": "¿En qué provincia te gustaría viajar?"},
    "tipo_turismo": {"options": ["playa", "montaña", "urbano", "mixto", "n"], "prompt": "¿Qué tipo de turismo prefieres?"},
    "temperatura": {"options": ["calor", "frío", "templado", "sin preferencia"], "prompt": "¿Qué clima prefieres?"},
    "tolerancia_multitudes": {"options": ["baja", "media", "alta"], "prompt": "¿Qué tolerancia tienes a las multitudes?"},
    "tolerancia_lluvia": {"options": ["baja", "media", "alta"], "prompt": "¿Qué tolerancia tienes a la lluvia?"},
    "presupuesto": {"options": ["bajo", "medio", "alto"], "prompt": "¿Cuál es tu presupuesto?"},
}

SYSTEM_PROMPT = (
    "Eres un asistente de viaje que recopila información de manera natural, solo un campo a la vez. "
    "Campos a recopilar: provincia_base, anio, mes, tipo_turismo, temperatura, preferencias_eventos, "
    "tolerancia_multitudes, tolerancia_lluvia, presupuesto. "
    "Regla general: si la respuesta no coincide con las opciones permitidas, ajusta al valor válido más cercano. "
    "Devuelve SOLO el valor normalizado para el campo actual, sin explicaciones ni bloques de código."
)

# --- Mostrar historial ---
for msg in st.session_state.messages:
    role = msg["role"]
    content = msg["content"]
    with st.chat_message(role):
        st.markdown(
            f"<div style='background-color:{'#DCF8C6' if role=='user' else '#F1F0F0'};"
            f" padding:12px; border-radius:12px; margin:5px 0;'>{content}</div>",
            unsafe_allow_html=True
        )

# --- Input del usuario ---
if prompt := st.chat_input("Escribe tu mensaje:"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(
            f"<div style='background-color:#DCF8C6; padding:12px; border-radius:12px; margin:5px 0;'>{prompt}</div>",
            unsafe_allow_html=True
        )

    with st.chat_message("assistant"):
        writing_placeholder = st.empty()
        writing_placeholder.markdown("...")

    # --- Primer saludo ---
    if not st.session_state.started:
        answer = f"¡Hola! Encantado de ayudarte a planear tu viaje. {fields_info[st.session_state.current_field]['prompt']}"
        st.session_state.messages.append({"role": "assistant", "content": answer})
        writing_placeholder.markdown(
            f"<div style='background-color:#F1F0F0; padding:12px; border-radius:12px; margin:5px 0;'>{answer}</div>",
            unsafe_allow_html=True
        )
        st.session_state.started = True
        st.stop()

    current_field = st.session_state.current_field
    messages_to_send = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": (
            f"JSON parcial: {json.dumps(st.session_state.json_data, ensure_ascii=False)}. "
            f"El usuario respondió para el campo '{current_field}': '{prompt}'. "
            "Devuelve únicamente el valor normalizado para este campo."
        )}
    ]

    try:
        response = requests.post(
            f"{OLLAMA_API_URL}/v1/chat/completions",
            json={"model": MODEL, "messages": messages_to_send}
        )
        normalized_value = response.json()["choices"][0]["message"]["content"].strip()

        # --- Verificar si la respuesta es válida ---
        valid_options = fields_info[current_field].get("options")
        is_valid = True
        if valid_options:
            if current_field == "preferencias_eventos":
                valores = [v.strip() for v in normalized_value.split(",")]
                if not any(v in valid_options for v in valores) and normalized_value.lower() != "none":
                    is_valid = False
            else:
                if normalized_value not in valid_options and normalized_value.lower() != "none":
                    is_valid = False

        if not is_valid:
            # --- Re-pregunta natural ---
            next_question_prompt = (
                f"La respuesta para '{current_field}' no es válida. Opciones permitidas: "
                f"{valid_options if valid_options else 'cualquier valor válido'}. "
                "Formula una pregunta natural al usuario para este campo."
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
            st.session_state.messages.append({"role": "assistant", "content": answer})
            writing_placeholder.markdown(
                f"<div style='background-color:#F1F0F0; padding:12px; border-radius:12px; margin:5px 0;'>{answer}</div>",
                unsafe_allow_html=True
            )
            st.stop()

        # --- Guardar respuesta válida ---
        st.session_state.json_data[current_field] = normalized_value

        # --- Verificar provincia_base ---
        if current_field == "provincia_base":
            if normalized_value.lower() != "none":
                st.session_state.json_data["modo_chat"] = "2"
            else:
                st.session_state.json_data["modo_chat"] = "1"

        # --- Si modo_chat = 2, completar JSON y finalizar ---
        if st.session_state.json_data.get("modo_chat") == "2" and current_field == "provincia_base":
            for field in fields_info:
                if field not in st.session_state.json_data:
                    if field in ["tipo_turismo", "temperatura", "tolerancia_multitudes", "tolerancia_lluvia", "presupuesto"]:
                        st.session_state.json_data[field] = "none"
            st.session_state.current_field = None
            answer = json.dumps(st.session_state.json_data, ensure_ascii=False, indent=2)
        else:
            # --- Avanzar al siguiente campo ---
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
                    "Formula una pregunta natural al usuario para este campo."
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

        st.session_state.messages.append({"role": "assistant", "content": answer})
        writing_placeholder.markdown(
            f"<div style='background-color:#F1F0F0; padding:12px; border-radius:12px; margin:5px 0;'>{answer}</div>",
            unsafe_allow_html=True
        )

    except Exception as e:
        writing_placeholder.empty()
        st.error(f"No se pudo conectar a Ollama: {e}")

# --- Mostrar JSON final ---
if st.session_state.current_field is None:
    st.subheader("✅ Información recopilada")
    st.json(st.session_state.json_data)
    st.download_button(
        "Descargar JSON",
        data=json.dumps(st.session_state.json_data, ensure_ascii=False, indent=2),
        file_name="datos_recolectados.json",
        mime="application/json"
    )
