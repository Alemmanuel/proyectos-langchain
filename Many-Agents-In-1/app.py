# Proyecto 9 - Agente Analista con Herramientas (LangChain + Gemini + Streamlit)
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
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
    google_api_key=api_key,
    temperature=0.7
)

# Herramientas que simulan distintas habilidades analíticas
def resume_texto(texto: str) -> str:
    """Resume el contenido proporcionado."""
    return f"🔍 Resumen:\n{texto[:300]}... (resumen simulado)"

def detecta_sentimiento(texto: str) -> str:
    """Detecta el sentimiento del texto."""
    if any(word in texto.lower() for word in ["feliz", "bueno", "excelente"]):
        return "😊 Sentimiento positivo"
    elif any(word in texto.lower() for word in ["triste", "malo", "horrible"]):
        return "😞 Sentimiento negativo"
    else:
        return "😐 Sentimiento neutral"

def cuenta_palabras(texto: str) -> str:
    """Cuenta la cantidad de palabras en el texto."""
    num = len(texto.split())
    return f"🧮 El texto tiene {num} palabras."

# Registrar herramientas
tools = [
    Tool(
        name="Resumen de Texto",
        func=resume_texto,
        description="Resume el contenido proporcionado."
    ),
    Tool(
        name="Detección de Sentimiento",
        func=detecta_sentimiento,
        description="Detecta el sentimiento del texto (positivo, negativo, neutral)."
    ),
    Tool(
        name="Contador de Palabras",
        func=cuenta_palabras,
        description="Cuenta la cantidad de palabras en el texto."
    )
]

# Inicializar agente
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# UI en Streamlit
st.set_page_config(page_title="Agente Analista de Texto", page_icon="🧠")
st.title("🧠 Agente Analista de Texto (LangChain + Gemini)")

input_text = st.text_area("Introduce el texto para analizar:", height=200)

if input_text:
    try:
        with st.spinner("Analizando texto..."):
            resultado = agent.run(f"Analiza el siguiente texto: {input_text}")
        st.success("✅ Análisis completo:")
        st.markdown(resultado)
    except Exception as e:
        st.error(f"⚠️ Error durante el análisis: {e}")