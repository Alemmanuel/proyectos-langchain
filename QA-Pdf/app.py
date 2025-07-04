# Proyecto 4 - Preguntas sobre PDFs con LangChain y Streamlit
import streamlit as st
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
import tempfile

load_dotenv()
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash-8b",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

st.set_page_config(page_title="PDF QA", page_icon="📄")
st.title("📄 Preguntas sobre un PDF")

pdf = st.file_uploader("Sube un archivo PDF", type=["pdf"])
if pdf:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf.read())
        loader = PyPDFLoader(tmp.name)
        documentos = loader.load()

    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documentos)
    vectorstore = FAISS.from_documents(chunks, HuggingFaceEmbeddings())
    retriever = vectorstore.as_retriever()
    chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    pregunta = st.text_input("Haz una pregunta sobre el PDF")
    if st.button("Responder") and pregunta:
        try:
            respuesta = chain.run(pregunta)
            st.success(respuesta)
        except Exception as e:
            st.error(f"Ocurrió un error al procesar la pregunta: {e}")
