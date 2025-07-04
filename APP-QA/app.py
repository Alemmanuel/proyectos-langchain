# Proyecto 6 - App de Preguntas y Respuestas con LangChain y Streamlit
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

# Cargar API Key
dotenv_path = ".env"
load_dotenv(dotenv_path)

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("❌ GOOGLE_API_KEY no encontrado en .env")

# Inicializar modelo Gemini
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash-8b",
    google_api_key=api_key
)

# Configurar interfaz de Streamlit
st.set_page_config(page_title="App QA", page_icon="❓")
st.title("❓ App de Preguntas y Respuestas con Gemini")

# Entrada del usuario
pregunta = st.text_input("Haz una pregunta sobre cualquier tema")

# Procesar y mostrar respuesta
if st.button("Responder") and pregunta:
    try:
        respuesta = llm.invoke(pregunta)
        st.success(respuesta.content)
    except Exception as e:
        st.error(f"⚠️ Error al generar la respuesta: {e}")