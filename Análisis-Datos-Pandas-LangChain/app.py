# Proyecto 12 - Q&A desde YouTube (versión solo terminal)
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

def check_api():
    if not GOOGLE_API_KEY:
        print("❌ GOOGLE_API_KEY no encontrada en .env")
        exit(1)

check_api()

print("=== Q&A desde YouTube con LangChain y Gemini (Terminal) ===")
youtube_url = input("🔗 Ingresa la URL del video de YouTube: ")

try:
    print("Cargando video y generando embeddings...")
    loader = YoutubeLoader.from_youtube_url(youtube_url, add_video_info=False)
    documents = loader.load()
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GOOGLE_API_KEY
    )
    db = FAISS.from_documents(documents, embeddings)
    retriever = db.as_retriever()
    print("✅ Video cargado y listo para preguntas.")
except Exception as e:
    print(f"❌ Error cargando el video: {e}")
    exit(1)

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash-8b",
    google_api_key=GOOGLE_API_KEY,
    temperature=0.3
)
chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

while True:
    pregunta = input("\n🧠 Escribe tu pregunta (o 'salir' para terminar): ")
    if pregunta.lower() == 'salir':
        print("Hasta luego!")
        break
    try:
        print("Generando respuesta...")
        respuesta = chain.run(pregunta)
        print(f"\n🤖 Respuesta:\n{respuesta}")
    except Exception as e:
        print(f"❌ Error generando respuesta: {e}")
