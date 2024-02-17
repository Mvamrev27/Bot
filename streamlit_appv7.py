import streamlit as st
import pandas as pd 
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA

def generate_response(uploaded_file, openai_api_key, query_text):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        # Assuming a column named 'text' contains your document content
        documents = df['text'].tolist()  

        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.create_documents(documents)

        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        db = Chroma.from_documents(texts, embeddings)
        retriever = db.as_retriever()

        qa = RetrievalQA.from_chain_type(llm=OpenAI(openai_api_key=openai_api_key), 
                                         chain_type='stuff', 
                                         retriever=retriever)
        return qa.run(query_text)

# Page title
st.set_page_config(page_title='🦜🔗 Ask the Doc App')
st.title('🦜🔗 Ask the Doc App')

# File upload
uploaded_file = st.file_uploader('Upload a CSV file', type='csv')

# Query text
query_text = st.text_input('Enter your question:', placeholder = 'Please provide a short summary.', disabled=not uploaded_file)

# Form input and query
result = []
with st.form('myform', clear_on_submit=True):
    # Retrieve OpenAI API Key from Streamlit Secrets
    openai_api_key = st.secrets["openai_api_key"] 

    submitted = st.form_submit_button('Submit', disabled=not(uploaded_file and query_text))
    if submitted: 
        with st.spinner('Calculating...'):
            response = generate_response(uploaded_file, openai_api_key, query_text)
            result.append(response)

if len(result):
    st.info(response)
