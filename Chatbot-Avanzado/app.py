import streamlit as st
from dotenv import load_dotenv
import os
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain_google_genai import ChatGoogleGenerativeAI

# Configuración de página
st.set_page_config(page_title="Chatbot Avanzado", page_icon="🤖")
st.title("🤖 Chatbot Avanzado con LangChain y Gemini")

# Cargar API Key
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.error("❌ GOOGLE_API_KEY no encontrada en .env")
    st.stop()

# Crear memoria de conversación
if "memoria" not in st.session_state:
    st.session_state.memoria = ConversationBufferMemory(return_messages=True)

if "conversacion" not in st.session_state:
    st.session_state.conversacion = []

# Inicializar modelo Gemini
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",  # Usa este formato para evitar errores
    google_api_key=GOOGLE_API_KEY,
    temperature=0.3
)

# Crear la cadena conversacional
chatbot = ConversationChain(llm=llm, memory=st.session_state.memoria)

# Inicializar variable para evitar error de acceso
if "ultima_pregunta" not in st.session_state:
    st.session_state.ultima_pregunta = ""

# Mostrar conversación previa con mejor estilo y scroll en caja grande solo si hay mensajes
if st.session_state.conversacion:
    st.markdown("""
    <style>
    .chat-container {
        max-height: 600px;
        min-height: 200px;
        overflow-y: auto;
        background: transparent;
        padding: 0px;
        border-radius: 0px;
        margin-bottom: 0px;
        border: none;
        margin-top: -200px;
    }
    .bot-bubble {
        background: #f1f8e9;
        padding: 16px;
        border-radius: 10px;
        margin-bottom: 18px;
        color: #222;
        font-size: 1.15em;
        font-weight: 500;
    }
    </style>
    <div class="chat-container" id="chat-container">
    """, unsafe_allow_html=True)
    for i, mensaje in enumerate(st.session_state.conversacion):
        if mensaje.startswith("🧍 Tú:") or mensaje.startswith("**🧍 Tú:**"):
            st.markdown(mensaje, unsafe_allow_html=True)
        elif mensaje.startswith("🤖 Gemini:") or mensaje.startswith("**🤖 Gemini:**"):
            st.markdown(f'<div class="bot-bubble">{mensaje}</div>', unsafe_allow_html=True)
        else:
            st.markdown(mensaje, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
    # Script para autoscroll abajo
    st.markdown("""
    <script>
    var chatContainer = window.parent.document.getElementById('chat-container');
    if (chatContainer) { chatContainer.scrollTop = chatContainer.scrollHeight; }
    </script>
    """, unsafe_allow_html=True)

# Campo de entrada siempre abajo del historial, sin burbuja blanca
with st.container():
    st.markdown('<div class="input-container"></div>', unsafe_allow_html=True)
    pregunta = st.text_input(
        "💬 Escribe tu mensaje:",
        key="input_pregunta"
    )

st.markdown("---")

# Procesar entrada solo si se presiona Enter y la pregunta es nueva y no vacía
if pregunta and pregunta != st.session_state.ultima_pregunta:
    with st.spinner("🧠 Pensando..."):
        try:
            respuesta = chatbot.run(pregunta)

            if not respuesta or respuesta.strip() == "":
                st.error("❌ El modelo no generó una respuesta.")
            else:
                user_msg = f"**🧍 Tú:** {pregunta}"
                bot_msg = f"**🤖 Gemini:** {respuesta}"

                st.session_state.conversacion.append(user_msg)
                st.session_state.conversacion.append(bot_msg)
                st.session_state.ultima_pregunta = pregunta  # Guardar la última pregunta procesada
                st.rerun()
        except Exception as e:
            st.error(f"❌ Error en la respuesta: {e}")
