import streamlit as st
import pdfplumber
from langchain.llms import OpenAI as LangChainOpenAI
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate

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
        
        # Set up the prompt template
        prompt_template = ChatPromptTemplate.from_messages([
            ("document", document_text),
            ("question", question)
        ])
        
        # Create an LLMChain for question answering
        chain = LLMChain(llm=llm, prompt=prompt_template)

        # Generate an answer based on the document text and the user's question
        answer = chain.run()

        st.write("Answer:", answer)
else:
    st.write("Please upload a PDF file.")
