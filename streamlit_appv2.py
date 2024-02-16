import streamlit as st
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import openai
from docarray import Document, DocumentArray
from langchain.llms import OpenAI as LangChainOpenAI

# Function to extract text from PDF using PyPDF2
def extract_text_from_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    text = ''
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

# Function to embed the document and query using sentence transformers and DocArray
def embed_text(text, model_name='all-MiniLM-L6-v2'):
    model = SentenceTransformer(model_name)
    embedding = model.encode(text, convert_to_tensor=True)
    return embedding.numpy()

# Streamlit application UI
st.title("PDF Question Answering Bot with LangChain and OpenAI")

uploaded_file = st.file_uploader("Upload a PDF document", type=['pdf'])

if uploaded_file:
    document_text = extract_text_from_pdf(uploaded_file)
    st.text_area("Extracted Text:", value=document_text, height=300, max_chars=5000)
    
    query_text = st.text_input("Enter your question:")

    if st.button("Answer") and query_text:
        # Initialize OpenAI
        openai_api_key = st.text_input('OpenAI API Key', type='password')
        if openai_api_key:
            llm = LangChainOpenAI(api_key=openai_api_key)
            
            # Embed the document and query
            doc_embedding = embed_text(document_text)
            query_embedding = embed_text(query_text)
            
            # Use DocArray for similarity search (simplified for demonstration)
            docs = DocumentArray([Document(text=document_text, embedding=doc_embedding)])
            query_doc = Document(text=query_text, embedding=query_embedding)
            docs.match(query_doc, metric='cosine', use_scipy=True)
            
            # For simplicity, just send the best match or query to OpenAI (adjust based on actual use case)
            matched_text = docs[0].text  # Assuming the first document is the best match
            response = llm.invoke(matched_text + query_text)
            
            st.write("Answer:", response)
else:
    st.write("Please upload a PDF file.")
