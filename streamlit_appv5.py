import streamlit as st
import openai
from langchain.llms import OpenAI
from langchain.vectorstores import Chroma
from langchain.document_loaders import DirectoryLoader
from langchain.chains.question_answering import ChromaDBQA
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

def answer_query(query, chromadb):
    """Uses ChromaDBQA to answer questions based on the indexed documents."""
    chain = ChromaDBQA.from_llm_and_vectorstore(OpenAI(temperature=0), chromadb)
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

    # Initialize ChromaDB (adjust path as needed)
    chromadb = Chroma.from_documents(vectorstore, collection_name="pdf_documents") 
    st.success("Documents indexed successfully!")

    # Chat Input Area
    user_query = st.text_input("Ask a question about the documents:")

    if user_query:
        answer = answer_query(user_query, chromadb)
        st.write("Answer:", answer)
