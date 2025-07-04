import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import os

load_dotenv()
llm = ChatGoogleGenerativeAI(
    model="models/gemini-1.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

st.set_page_config(page_title="Pregunta Directa", page_icon="🔍")
st.title("🔍 Haz una Pregunta Directa a Gemini")

pregunta = st.text_input("¿Qué quieres saber?")
if st.button("Preguntar") and pregunta:
    respuesta = llm.invoke(pregunta)
    st.success(respuesta)