# Proyecto 12 - Q&A desde YouTube (versión funcional full Streamlit)
import streamlit as st
from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.document_loaders import YoutubeLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import GoogleGenerativeAIEmbeddings
from langchain.chains import RetrievalQA

# Cargar variables de entorno
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Validar API Key
def check_api():
    if not GOOGLE_API_KEY:
        st.error("❌ GOOGLE_API_KEY no encontrada en .env")
        st.stop()

# Configurar Streamlit
st.set_page_config(page_title="YouTube Q&A App", page_icon="🎥")
st.title("🎥 YouTube Q&A con LangChain y Gemini")

check_api()

# Input de video
youtube_url = st.text_input("🔗 Ingresa la URL del video de YouTube")
procesado = False

# Procesar video y construir retriever
if youtube_url:
    with st.spinner("📥 Cargando video y generando embeddings..."):
        try:
            loader = YoutubeLoader.from_youtube_url(youtube_url, add_video_info=False)
            documents = loader.load()
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/embedding-001",
                google_api_key=GOOGLE_API_KEY
            )
            db = FAISS.from_documents(documents, embeddings)
            retriever = db.as_retriever()
            procesado = True
            st.success("✅ Video cargado y listo para preguntas.")
        except Exception as e:
            st.error(f"❌ Error cargando el video: {e}")

# Campo de pregunta
if procesado:
    pregunta = st.text_input("🧠 Haz una pregunta sobre el contenido del video:")
    if pregunta:
        with st.spinner("✍️ Generando respuesta..."):
            try:
                llm = ChatGoogleGenerativeAI(
                    model="gemini-1.5-flash-8b",
                    google_api_key=GOOGLE_API_KEY,
                    temperature=0.3
                )
                chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
                respuesta = chain.run(pregunta)
                st.markdown(f"### 🤖 Respuesta:\n{respuesta}")
            except Exception as e:
                st.error(f"❌ Error generando respuesta: {e}")
