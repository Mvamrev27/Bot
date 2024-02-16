import streamlit as st
import pdfplumber
from langchain.llms import OpenAI as LangChainOpenAI
from langchain.chains import TextRetrievalChain

# Use Streamlit Secrets to access the OpenAI API key
openai_api_key = st.secrets["openai_api_key"]

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    with pdfplumber.open(pdf_file) as pdf:
        text = ''
        for page in pdf.pages:
            text += page.extract_text() + " "
    return text

# Streamlit application UI
st.title("PDF Question Answering Bot with LangChain and OpenAI")

pdf_file = st.file_uploader("Upload a PDF document", type=["pdf"])

if pdf_file is not None:
    document_text = extract_text_from_pdf(pdf_file)
    st.text_area("Extracted Text:", value=document_text, height=250, max_chars=5000)
    
    question = st.text_input("What's your question?")

    if st.button("Answer"):
        # Initialize LangChain OpenAI model with the API key from Streamlit Secrets
        llm = LangChainOpenAI(openai_api_key=openai_api_key)

        # Create a Text Retrieval Chain
        retrieval_chain = TextRetrievalChain(document=document_text, llm=llm)

        # Run the chain to get the answer
        answer = retrieval_chain.run(question)
        
        st.write("Answer:", answer)
else:
    st.write("Please upload a PDF file.")
