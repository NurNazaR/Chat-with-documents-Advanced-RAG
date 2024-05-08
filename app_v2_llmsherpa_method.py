import os
import tempfile

import streamlit as st 
from dotenv import load_dotenv
from langchain.chat_models.openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template, source_template 
import chromadb
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings)
from langchain.vectorstores.chroma import Chroma
from llmsherpa.readers import LayoutPDFReader
from langchain_core.documents import Document
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_openai import ChatOpenAI

from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder


# Function to display a single source
def display_source(source_name, page_idx, parent, content):
    # Expandable content area
    with st.expander(f'Sources: {source_name}           page_id: {page_idx}     parent-section: {parent}'):
        st.write(content)

def get_pdf_texts_with_chunks(pdf_docs):
    chunks = []
    with tempfile.TemporaryDirectory() as temp_dir:
        for pdf in pdf_docs:
            # Save each uploaded file to the temporary directory
            file_path = os.path.join(temp_dir, pdf.name)
            with open(file_path, "wb") as f:
                f.write(pdf.getbuffer())

            llmsherpa_api_url = "https://readers.llmsherpa.com/api/document/developer/parseDocument?renderFormat=all"
            pdf_reader = LayoutPDFReader(llmsherpa_api_url)
            doc = pdf_reader.read_pdf(file_path)  
            
            for section in doc.sections():    
                for chunk in section.chunks():
                    metadata = {}
                    metadata['parent'] = chunk.parent.to_text()
                    metadata['page_idx'] = chunk.page_idx
                    metadata['block_idx'] = chunk.block_idx
                    metadata['tag'] = chunk.tag
                    metadata['source'] = str(pdf.name)
                
                    chunks.append(Document(page_content=chunk.to_text(), metadata=metadata))
    return chunks
    

    
def get_vectorstore(text_chunks, collection_name):
    embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Initialize ChromaDB
    chroma_client = chromadb.Client()

    # Check if a collection with the same name exists
    if collection_name in chroma_client.list_collections():
        # Delete the existing collection
        chroma_client.delete_collection(collection_name)
    
    vector_store = Chroma.from_documents(text_chunks, embedding=embedding_function, collection_name = collection_name)
    return vector_store


def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})
    
    #setting retriever
    retriever_from_llm = MultiQueryRetriever.from_llm(retriever=vectorstore.as_retriever(search_kwargs = {"k":10}), llm=llm)
    
    model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
    compressor = CrossEncoderReranker(model=model, top_n=3)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=retriever_from_llm
    )
    
    
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
                display_source(source.metadata["source"], source.metadata["page_idx"], source.metadata['parent'], source.page_content)
        
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
                # get the text chunks
                text_chunks = get_pdf_texts_with_chunks(pdf_docs)

                # create vector store
                vectorstore = get_vectorstore(text_chunks, collection_name="chat_with_pdf")

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(
                    vectorstore)


if __name__ == '__main__':
    main()