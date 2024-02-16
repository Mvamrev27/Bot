import os
import tempfile
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain

st.set_page_config(page_title="LangChain Bot", page_icon=":earth_asia:")
st.title(":earth_asia: LangChain Bot")

@st.experimental_singleton
def configure_retriever(uploaded_files):
    # Temporary directory for saving uploaded files
    temp_dir = tempfile.TemporaryDirectory()
    docs = []
    for file in uploaded_files:
        temp_filepath = os.path.join(temp_dir.name, file.name)
        with open(temp_filepath, "wb") as f:
            f.write(file.getbuffer())
        loader = PyPDFLoader(filepath=temp_filepath)
        docs.extend(loader.load())

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    # Embed documents and create vector store
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectordb = DocArrayInMemorySearch.from_documents(splits, embeddings)

    # Setup retriever
    retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 2, "fetch_k": 4})
    return retriever

# Upload PDF files and configure retriever
uploaded_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)
if uploaded_files:
    retriever = configure_retriever(uploaded_files)
else:
    st.stop()

# Initialize chat model and memory
openai_api_key = st.text_input("OpenAI API Key", type="password")
if not openai_api_key:
    st.stop()

llm = ChatOpenAI(api_key=openai_api_key, model_name="gpt-3.5-turbo", temperature=0)
memory = ConversationBufferMemory()
msgs = StreamlitChatMessageHistory()

# Setup conversational retrieval chain
qa_chain = ConversationalRetrievalChain(llm=llm, retriever=retriever, memory=memory, verbose=True)

# Display chat history and input
for msg in msgs.messages:
    if msg.type == "human":
        st.chat_message("user").write(msg.content)
    elif msg.type == "ai":
        st.chat_message("assistant").write(msg.content)

user_query = st.chat_input("Ask me anything about the uploaded documents!")
if user_query:
    msgs.add_human_message(user_query)
    response = qa_chain.run(user_query)
    msgs.add_ai_message(response)
    st.experimental_rerun()
