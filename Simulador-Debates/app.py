# Proyecto 11 - Simulador de Debates (LangChain + Gemini + Streamlit)
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import Tool, initialize_agent
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

# Crear los agentes simulando dos personajes en debate
def crear_agente(nombre: str, rol: str):
    def responder(pregunta: str) -> str:
        prompt = f"Eres {nombre}, un experto con el siguiente rol: {rol}. Responde de forma convincente a la siguiente afirmación o pregunta de debate:\n\n'{pregunta}'"
        return llm.invoke(prompt)

    return Tool(
        name=f"{nombre} ({rol})",
        func=responder,
        description=f"Agente de debate con perspectiva de {rol}."
    )

agente_1 = crear_agente("Camila", "científica ambiental")
agente_2 = crear_agente("David", "economista industrial")

# Inicializar agente maestro con herramientas
agent = initialize_agent(
    tools=[agente_1, agente_2],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# UI con Streamlit
st.set_page_config(page_title="Simulador de Debate IA", page_icon="🎤")
st.title("🎤 Simulador de Debate (LangChain + Gemini)")

tema = st.text_input("🗣️ Introduce un tema de debate o una pregunta desafiante:")

if tema:
    with st.spinner("🧠 Generando argumentos de ambos lados..."):
        try:
            respuesta = agent.run(f"Debatir sobre: {tema}")
            st.success("✅ Debate generado:")
            st.markdown(respuesta)
        except Exception as e:
            st.error(f"⚠️ Error durante el debate: {e}")
