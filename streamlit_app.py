import streamlit as st
import pdfplumber
from langchain.llms import OpenAI as LangChainOpenAI

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    with pdfplumber.open(pdf_file) as pdf:
        text = ''
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

# Streamlit application UI
st.title("PDF Question Answering Bot with LangChain and OpenAI")

pdf_file = st.file_uploader("Upload a PDF document", type=["pdf"])

if pdf_file is not None:
    document_text = extract_text_from_pdf(pdf_file)
    st.text_area("Extracted Text:", value=document_text, height=250, max_chars=5000)
    
    question = st.text_input("What's your question?")

    if st.button("Answer"):
        # Use Streamlit Secrets to securely use OpenAI API key
        openai_api_key = st.secrets["openai_api_key"]
        
        # Initialize LangChain's OpenAI model
        llm = LangChainOpenAI(api_key=openai_api_key)
        
        # Since directly including the document text as a part of the prompt is not feasible,
        # you should adjust the question or method of incorporating document context.
        # For this example, we'll simply pass the question to the model.
        # In a more complex implementation, you might include relevant snippets of text with the question.
        response = llm.invoke(question)

        st.write("Answer:", response)
else:
    st.write("Please upload a PDF file.")
