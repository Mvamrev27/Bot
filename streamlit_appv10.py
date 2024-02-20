import streamlit as st
import nest_asyncio
from pathlib import Path
from llama_hub.file.unstructured import UnstructuredReader
from llama_index.llms import OpenAI
from llama_index import GPTSimpleVectorIndex

nest_asyncio.apply()

st.title('Document Query with LLaMA')

openai_api_key = st.secrets["OPENAI_API_KEY"]  

uploaded_file = st.file_uploader("Choose a PDF file", type=['pdf'])
if uploaded_file is not None:
    try:
        file_path = Path(uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        documents = UnstructuredReader().load_data(file_path)

        # Create index (assuming text is extracted cleanly)
        index = GPTSimpleVectorIndex(documents)  

        # Example of openAI query interaction
        query_llm = OpenAI(api_key=openai_api_key, model="gpt-3.5-turbo", max_tokens=256)

        query = st.text_input("Enter your query for specific data:")
        if query:
           response = index.query(query, text_encoder=query_llm.embed_query)

           # Display the most relevant document snippet
           st.write(response.source)  

    except Exception as e:
        st.error(f"An error occurred: {e}")
