import streamlit as st
import nest_asyncio
from pathlib import Path
from llama_index import download_loader, VectorStoreIndex, download_llama_pack
from llama_index.llms import OpenAI
from llama_index.text_splitter import SentenceSplitter
from llama_hub.file.unstructured import UnstructuredReader

# Apply necessary to run async code
nest_asyncio.apply()

# Initialize Streamlit app
st.title('LLaMA Document Query App')

# File Uploader
uploaded_file = st.file_uploader("Choose a PDF file", type=['pdf'])
if uploaded_file is not None:
    file_path = Path(uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Load the PDF document using the chosen method
    documents_method = st.radio("Choose the document loading method:", ('LLaMA Index', 'LLaMA Hub'))

    if documents_method == 'LLaMA Index':
        PDFReader = download_loader("PDFReader")
        loader = PDFReader()
        documents = loader.load_data(file=file_path)
    else:
        documents = UnstructuredReader().load_data(file_path)

    # Download and initialize the DenseXRetrievalPack
    DenseXRetrievalPack = download_llama_pack("DenseXRetrievalPack", "./dense_pack")

    openai_api_key = st.secrets["OPENAI_API_KEY"]
    dense_pack = DenseXRetrievalPack(
      documents,
      proposition_llm=OpenAI(api_key=openai_api_key, model="gpt-3.5-turbo", max_tokens=750),
      query_llm=OpenAI(api_key=openai_api_key, model="gpt-3.5-turbo", max_tokens=256),
      text_splitter=SentenceSplitter(chunk_size=1024)
    )
    dense_query_engine = dense_pack.query_engine

    # Query Engine based on VectorStoreIndex (optional, demonstrating alternative)
    base_index = VectorStoreIndex.from_documents(documents)
    base_query_engine = base_index.as_query_engine()

    # User input for query
    query = st.text_input("Enter your query:")
    if query:
        response = dense_query_engine.query(query)
        st.write(response.response)
