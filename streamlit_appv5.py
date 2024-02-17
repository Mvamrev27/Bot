import streamlit as st
import openai
from langchain.llms import OpenAI
from langchain.vectorstores import Chroma
from langchain.document_loaders import DirectoryLoader
import PyPDF2

def create_index(uploaded_files):
    """Creates a FAISS index from PDF documents."""
    text_docs = []
    for uploaded_file in uploaded_files:  
        with uploaded_file as file_object:  
            pdf_reader = PyPDF2.PdfReader(file_object)  
            text = ""
            for page in range(len(pdf_reader.pages)):
                text += pdf_reader.pages[page].extract_text() + "\n"
            text_docs.append(text)

    loader = DirectoryLoader(text_docs, 1000)  # Adjust chunk size as needed
    docs = loader.load()
    vectorstore = FAISS.from_documents(docs, OpenAIEmbeddings())
    return vectorstore

def answer_query(query, chromadb):
    """Retrieves relevant documents from ChromaDB and uses OpenAI for answer generation."""
    relevant_docs = chromadb.similarity_search(query) 

    # Construct a text source from relevant documents for OpenAI 
    source_text = ""
    for doc in relevant_docs:
        source_text += doc.page_content + "\n"  

    # Use OpenAI to generate an answer, specifying GPT-3.5 Turbo
    response = openai.Completion.create(
        engine="gpt-3.5-turbo",  # Use gpt-3.5-turbo model
        prompt=f"Question: {query} \n Documents: {source_text} \n Answer:",
        max_tokens=256,  # Adjust as needed
        n=1,
        stop=None,
        temperature=0, 
    )
    return response.choices[0].text.strip() 

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
