import os
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document

# Cargar API KEY
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("❌ GOOGLE_API_KEY no encontrado en .env")

# Inicializar modelo Gemini
llm = ChatGoogleGenerativeAI(
    model="models/gemini-1.5-flash",
    google_api_key=api_key,
    temperature=0.3
)

# Función para obtener transcripción
def obtener_transcripcion(video_id):
    transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['es', 'en'])
    texto = " ".join([seg['text'] for seg in transcript])
    return texto

# Función para construir vectorstore
def construir_vectorstore(texto):
    docs = [Document(page_content=texto)]
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    embeddings = HuggingFaceEmbeddings()
    return FAISS.from_documents(chunks, embeddings)

# Interfaz
print("🎥📄 Sistema de Preguntas sobre Videos de YouTube")
print("Escribe 'salir' para terminar.\n")

while True:
    video_url = input("🔗 Pega la URL del video de YouTube: ").strip()
    if video_url.lower() in ["salir", "exit", "quit"]:
        print("👋 Cerrando, The Special One.")
        break

    try:
        video_id = video_url.split("v=")[-1][:11]
        texto = obtener_transcripcion(video_id)
        print("✅ Transcripción obtenida.")

        vectorstore = construir_vectorstore(texto)
        retriever = vectorstore.as_retriever()
        qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

        print("📢 Ya puedes hacer preguntas sobre el video.")
        print("Escribe 'salir' para volver al inicio.\n")

        while True:
            pregunta = input("Tú: ").strip()
            if pregunta.lower() in ["salir", "exit", "quit"]:
                break
            respuesta = qa.invoke({"query": pregunta})
            print(f"🧠 Gemini: {respuesta['result']}\n")

    except Exception as e:
        print(f"⚠️ Error: {e}\n")
