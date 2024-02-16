import streamlit as st
import pdfplumber
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA

# Function to extract text from PDF
def extract_text_from_pdf(uploaded_file):
    text_data = ''
    try:
        with pdfplumber.open(uploaded_file) as pdf:
            for page in pdf.pages:
                text_data += page.extract_text() + "\n"
    except Exception as e:
        st.error(f"Error extracting text from PDF: {e}")
    return text_data

# Page title
st.set_page_config(page_title='PDF Question Answering Bot')
st.title('📄 PDF Question Answering Bot')

# File upload
uploaded_file = st.file_uploader('Upload a PDF file', type=['pdf'])  # Specify 'pdf' as the accepted file type

# Query text input
query_text = st.text_input('Enter your question:', placeholder='Ask me anything about the PDF content.', disabled=not uploaded_file)

if uploaded_file and query_text:
    # Ask for OpenAI API Key
    openai_api_key = st.text_input('OpenAI API Key', type='password')
    if openai_api_key:
        with st.spinner('Extracting text and generating response...'):
            # Extract text from the PDF
            document_text = extract_text_from_pdf(uploaded_file)
            if document_text:
                # Split document into chunks
                text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
                texts = text_splitter.create_documents([document_text])
                
                # Use OpenAI embeddings
                embeddings = OpenAIEmbeddings(api_key=openai_api_key)
                
                # Create a vector store from document chunks
                db = Chroma.from_documents(texts, embeddings)
                
                # Create retriever interface
                retriever = db.as_retriever()
                
                # Initialize QA chain with the OpenAI model and retriever
                qa_chain = RetrievalQA(llm=OpenAI(api_key=openai_api_key), retriever=retriever)
                
                # Generate response
                response = qa_chain.run(query_text)
                st.write("Answer:", response)
