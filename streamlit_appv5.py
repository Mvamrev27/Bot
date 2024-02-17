import streamlit as st
import openai
from langchain.llms import OpenAI
from langchain.vectorstores import FAISS
from langchain.vectorstores.base import VectorStore
from langchain.document_loaders import DirectoryLoader
from langchain.chains.question_answering import VectorDBQA
import PyPDF2

def create_index(pdf_dir):
    """Creates a FAISS index from PDF documents."""
    text_docs = []
    for file_path in pdf_dir.iterdir():
        if file_path.suffix == ".pdf":
            pdf_reader = PyPDF2.PdfReader(str(file_path))
            text = ""
            for page in range(len(pdf_reader.pages)):
                text += pdf_reader.pages[page].extract_text() + "\n"
            text_docs.append(text)

    loader = DirectoryLoader(text_docs, 1000)  # Adjust chunk size as needed
    docs = loader.load()
    vectorstore = FAISS.from_documents(docs, OpenAIEmbeddings())
    return vectorstore

def answer_query(query, vectorstore):
    """Uses VectorDBQA to answer questions based on the indexed documents."""
    if not isinstance(vectorstore, VectorStore): 
        raise ValueError("vectorstore must be a langchain VectorStore")
    chain = VectorDBQA.from_llm_and_vectorstore(OpenAI(temperature=0), vectorstore)
    response = chain.run(query)
    return response

# Streamlit App Design
st.title("RAG Streamlit OpenAI Bot with PDF Upload")

# Get OpenAI Key from Streamlit Secrets
openai.api_key = st.secrets["openai_api_key"]

# File Uploader
uploaded_files = st.file_uploader("Upload PDF documents", type="pdf", accept_multiple_files=True)

if uploaded_files:
    vectorstore = create_index(uploaded_files)
    st.success("Documents indexed successfully!")

    # Chat Input Area
    user_query = st.text_input("Ask a question about the documents:")

    if user_query:
        answer = answer_query(user_query, vectorstore)
        st.write("Answer:", answer)
