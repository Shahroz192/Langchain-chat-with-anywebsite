# Website Chat

Website Chat is a Streamlit application that allows you to ask questions about a website and get answers based on the information available on the website. It uses the OLLAMA language model from Anthropic for question-answering and FAISS for document embedding and retrieval.

## Features

- Extract information from a given website URL
- Split the information into chunks for efficient processing
- Generate embeddings for the chunks using Hugging Face Embeddings
- Answer user queries using a RetrievalQA chain with the OLLAMA language model

## Requirements

- Python 3.6 or higher
- Streamlit
- langchain
- langchain-community

## Installation

1. Clone the repository:
```
git clone https://github.com/Shahroz192/chat-with-website.git
```
2. Install the required packages:
```
pip install -r requirements.txt
```

## Usage

1. Run the Streamlit app:
```
streamlit run app.py
```
2. Enter the URL of the website you want to ask questions about in the sidebar.
3. Ask a question about the website in the input field.
4. The app will display the answer based on the information available on the website.


## Code Overview

### Setting up the Local LLM
```python
from langchain_community.llms import Ollama

def get_llm():
    return ChatGoogleGenerativeAI(model="gemini-2.0-flash",)

```

### Getting Links from the Provided Text
```python
from langchain_community.document_loaders import WebBaseLoader

def get_links(url):
    docs = WebBaseLoader(url).load()
    return docs
```

### Splitting Documents into Chunks
```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

def get_chunks(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(docs)
    return chunks
```

### Getting Embeddings for the Chunks
```python
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def get_embeddings(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    docsearch = FAISS.from_documents(chunks, embeddings)
    return docsearch
```

### Getting Answer for the Given Query
```python
from langchain.chains import RetrievalQA

def get_answer(query, docsearch, llm):
    retriever = docsearch.as_retriever()
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    answer = qa.invoke(query)
    return answer
```

### Streamlit App
```python
import streamlit as st

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
```

## Limitations

- The app may not work properly for websites that require authentication or have dynamic content.
- The app may not be able to answer questions that require reasoning or common sense knowledge.

## License

This project is licensed under the MIT License.

## Acknowledgments

- [Streamlit](https://streamlit.io/) for building the app interface.
- [Anthropic](https://www.anthropic.com/) for providing the OLLAMA language model.
- [Langchain](https://github.com/hwchase17/langchain) and [Langchain-community](https://github.com/hwchase17/langchain-community) for providing the tools and utilities for building the app.


