import streamlit as st
import PyPDF2
from sentence_transformers import SentenceTransformer
import openai
from langchain.llms import OpenAI as LangChainOpenAI

# Assuming you have already set openai.api_key globally or in your environment variables

# Function to extract text from PDF using PyPDF2
def extract_text_from_pdf(uploaded_file):
    pdf_reader = PyPDF2.PdfFileReader(uploaded_file)
    text = ''
    for page_num in range(pdf_reader.numPages):
        text += pdf_reader.getPage(page_num).extractText() + "\n"
    return text

# Initialize the Sentence Transformer model for embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

# Embed a piece of text
def embed_text(text):
    return model.encode(text)

st.title("RAG Bot for PDF Documents")

uploaded_file = st.file_uploader("Upload your PDF file here", type="pdf")
question = st.text_input("Ask a question based on the PDF content:")

if st.button("Answer"):
    if uploaded_file is not None and question:
        document_text = extract_text_from_pdf(uploaded_file)
        # For demonstration, we show the document text in the app
        # In a real app, you might want to skip this step or handle it differently
        st.text_area("Extracted text from the PDF:", document_text, height=200)

        # Embed the document and the question
        document_embedding = embed_text(document_text)
        question_embedding = embed_text(question)

        # This step is placeholder for actual retrieval logic based on embeddings
        # In a complete implementation, you would use the embeddings to find the most relevant
        # parts of the text to the question, possibly using cosine similarity, etc.
        # Here, we simply pass the question to OpenAI's API via LangChain for an answer
        # assuming the document context is somehow included in the question or preprocessing step.
        
        openai_api_key = st.text_input("Enter your OpenAI API Key:", type="password")
        if openai_api_key:
            llm = LangChainOpenAI(api_key=openai_api_key)
            response = llm.invoke(question)  # This needs to be adapted based on LangChain's current API
            st.write("Response:", response)
    else:
        st.error("Please upload a PDF file and enter a question.")
