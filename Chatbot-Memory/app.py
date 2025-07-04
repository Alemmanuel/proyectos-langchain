# Proyecto 8 - Chatbot con Memoria usando LangChain, Gemini y Streamlit
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import HumanMessage, AIMessage
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

# Crear memoria persistente
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(return_messages=True, memory_key="chat_history")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Sincronizar mensajes anteriores con la memoria del agente
if not st.session_state.memory.chat_memory.messages and st.session_state.messages:
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.session_state.memory.chat_memory.add_message(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            st.session_state.memory.chat_memory.add_message(AIMessage(content=msg["content"]))

# Crear prompt personalizado
prompt = PromptTemplate(
    input_variables=["chat_history", "input"],
    template="""
Eres un asistente conversacional inteligente y amable. Usa el siguiente historial de conversación para responder con contexto.

Historial:
{chat_history}

Usuario: {input}
Asistente:
""".strip()
)

chat_chain = LLMChain(
    llm=llm,
    prompt=prompt,
    memory=st.session_state.memory,
    verbose=True
)

# Configurar Streamlit
st.set_page_config(page_title="Chat con Memoria", page_icon="🧠")
st.title("🧠 Chatbot con Memoria (Gemini + LangChain)")

# Mostrar historial visual
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Entrada del usuario
if prompt := st.chat_input("Tu mensaje"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    try:
        respuesta = chat_chain.run(prompt)
        st.session_state.messages.append({"role": "assistant", "content": respuesta})
        with st.chat_message("assistant"):
            st.markdown(respuesta)
    except Exception as e:
        st.error(f"⚠️ Error durante la conversación: {e}")