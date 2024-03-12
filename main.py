from langchain_community.llms.huggingface_endpoint import HuggingFaceEndpoint
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.document_loaders import WebBaseLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import streamlit as st

repo_id = "mistralai/Mistral-7B-v0.1"


# Set up Hugging Face LLM
def get_llm(api_token):
    return HuggingFaceEndpoint(
        repo_id=repo_id, max_length=128, temperature=0.5, token=api_token
    )


# Get links from the provided text
def get_links(text):
    docs = WebBaseLoader(text).load()
    return docs


# Split documents into chunks
def get_chunks(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(docs)
    return chunks


# Get embeddings for the chunks
def get_embeddings(chunks):
    embeddings = HuggingFaceEmbeddings()
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
    answer = qa.run(query)
    return answer


# Streamlit app
def app():
    st.set_page_config(page_title="Website Chat", page_icon=":robot_face:")
    st.title("Chat with Website")
    api_token = st.sidebar.text_input(
        "Enter your Hugging Face API token", type="password"
    )
    if api_token:
        llm = get_llm(api_token)
        link = st.sidebar.text_input("Enter website URL")
        if link:
            try:
                docs = get_links(link)
                chunks = get_chunks(docs)
                docsearch = get_embeddings(chunks)
                st.header("Ask a question about the website")
                query = st.text_input("Enter your question:")
                if query:
                    answer = get_answer(query, docsearch, llm)
                    st.success(answer)
            except Exception as e:
                st.error(f"An error occurred: {e}")
        else:
            st.warning("Please enter a valid website URL.")
    else:
        st.warning("Please enter your Hugging Face API token.")


if __name__ == "__main__":
    app()
