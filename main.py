from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import streamlit as st
import getpass
import os

if "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter your Google AI API key: ")


def get_llm():
    return ChatGoogleGenerativeAI(model="gemini-2.0-flash",)


def get_links(url):
    docs = WebBaseLoader(url).load()
    return docs


# Split documents into chunks
def get_chunks(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(docs)
    return chunks


# Get embeddings for the chunks
def get_embeddings(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    docsearch = FAISS.from_documents(chunks, embeddings)
    return docsearch


# Get answer for the given query
def get_answer(query, docsearch, llm):
    retriever = docsearch.as_retriever()
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
    )
    answer = qa.invoke(query)
    return answer


# Streamlit app
def app():
    st.set_page_config(page_title="Website Chat", page_icon=":robot_face:")
    st.title("Chat with Website")
    link = st.sidebar.text_input("Enter website URL")
    if link:
        try:
            llm = get_llm()
            docs = get_links(link)
            chunks = get_chunks(docs)
            docsearch = get_embeddings(chunks)
            st.header("Ask a question about the website")
            query = st.text_input("Enter your question:")
            if query:
                answer = get_answer(query, docsearch, llm)
                st.write(answer)
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter a valid website URL.")


if __name__ == "__main__":
    app()
