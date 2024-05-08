# importing libraries
import streamlit as st
from dotenv import load_dotenv  # to load api keys
from PyPDF2 import PdfReader    # to upload pdf files as text
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template

import chromadb  # vector database
from langchain.vectorstores.chroma import Chroma
 
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings) 

from langchain.retrievers.multi_query import MultiQueryRetriever


# Function to display a single retrieved document
def display_source(content):
    # Expandable content area
    with st.expander('Sources'):
        st.write(content)
        
# extracting text from all pdf files and storing it as a string in text variable 
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# text variable is too large to process; this function is used to divide text into small chunks
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


# each chunk has its embedding (numeric representation) and we will store these chunks with embeddings in Chroma vectore store
def get_vectorstore(text_chunks, collection_name):
    
    # we need embedding models to get vector representation of the text
    embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Initialize ChromaDB
    chroma_client = chromadb.Client()

    # Check if a collection with the same name exists
    if collection_name in chroma_client.list_collections():
        # Delete the existing collection
        chroma_client.delete_collection(collection_name)
    
    # creating the new vectorstore
    vector_store = Chroma.from_texts(text_chunks, embedding=embedding_function, collection_name = collection_name)
    
    return vector_store

# The function to store chat history of the user with llm
def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})
    
    #setting two step retriever system
    # first retriever --> multi query retriever to extract possible relevant documents
    retriever_from_llm = MultiQueryRetriever.from_llm(retriever=vectorstore.as_retriever(search_kwargs = {"k":10}), llm=llm)
    
    # second retriever --> reranking extracted documents from first retriever with cross encoder model
    model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
    compressor = CrossEncoderReranker(model=model, top_n=3)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=retriever_from_llm
    )
    
    # setting conversation chain for langchain
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True, output_key = 'answer')
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=compression_retriever,
        memory=memory,
        return_source_documents = True
    )
    return conversation_chain
    


def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']
    
    source_documents = response['source_documents']
    st.session_state.retrieved_sources.append(source_documents) 
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
            
            # Display each source using the function
            st.write("Retrieved documents")
            for source in st.session_state.retrieved_sources[i//2]:
                display_source(source.page_content)
        
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)


def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if "retrieved_sources" not in st.session_state:
        st.session_state.retrieved_sources = []

    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)

                # get the text chunks
                text_chunks = get_text_chunks(raw_text)

                # create vector store
                vectorstore = get_vectorstore(text_chunks, collection_name='chat_with_pdf')

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(
                    vectorstore)


if __name__ == '__main__':
    main()