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
st.set_page_config(page_title="IA Generativa", page_icon="✨")
st.title("✨ Creador de Texto con Gemini IA")

# Entrada del usuario
prompt_usuario = st.text_area("Escribe un prompt para generar contenido")

# Generar contenido
if st.button("Generar") and prompt_usuario:
    try:
        respuesta = llm.invoke(prompt_usuario)
        st.subheader("🧠 Resultado generado:")
        st.write(respuesta.content)
    except Exception as e:
        st.error(f"⚠️ Error al generar contenido: {e}")