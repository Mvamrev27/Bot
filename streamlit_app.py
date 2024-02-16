import streamlit as st
import pdfplumber
from transformers import RagTokenizer, RagRetriever, RagTokenForGeneration
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Load the RAG model components
tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-nq")
retriever = RagRetriever.from_pretrained("facebook/rag-token-nq", index_name="exact", use_dummy_dataset=True)
model = RagTokenForGeneration.from_pretrained("facebook/rag-token-nq", retriever=retriever)

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    with pdfplumber.open(pdf_file) as pdf:
        text = ''
        for page in pdf.pages:
            text += page.extract_text() + "\n" # Adding a newline character for each page
    return text

# Streamlit UI
st.title("PDF RAG Bot")
pdf_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if pdf_file is not None:
    # Extract text from PDF
    document_text = extract_text_from_pdf(pdf_file)
    st.text_area("Text extracted from PDF:", value=document_text, height=300, max_chars=5000)

    # Ask a question
    question = st.text_input("Ask a question based on the PDF:")
    
    if st.button("Get Answer"):
        # Encode the question and the retrieved documents
        inputs = tokenizer(question, return_tensors="pt")
        
        # Generate an answer
        with torch.no_grad():
            outputs = model.generate(input_ids=inputs["input_ids"])
        
        # Decode and display the answer
        answer = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        st.write("Answer:", answer[0])
else:
    st.write("Please upload a PDF file.")
