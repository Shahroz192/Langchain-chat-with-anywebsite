# Website Chat

Website Chat is a Streamlit application that allows you to ask questions about a website and get answers based on the information available on the website. It uses Hugging Face's Mistral-7B-v0.1 language model for question-answering and FAISS for document embedding and retrieval.

## Features

- Extract information from a given website URL
- Split the information into chunks for efficient processing
- Generate embeddings for the chunks using Hugging Face Embeddings
- Answer user queries using a RetrievalQA chain with Hugging Face's Mistral-7B-v0.1 language model

## Requirements

- Python 3.6 or higher
- Streamlit
- langchain
- langchain-community
- Hugging Face API token

## Installation

1. Clone the repository:
```
git clone https://github.com/yourusername/website-chat.git
```
2. Install the required packages:
```
pip install streamlit langchain langchain-community
```
3. Obtain a Hugging Face API token from the [Hugging Face website](https://huggingface.co/settings/tokens) and provide it as an input in the Streamlit app.

## Usage

1. Run the Streamlit app:
```
streamlit run app.py
```
2. Enter your Hugging Face API token in the sidebar.
3. Enter the URL of the website you want to ask questions about.
4. Ask a question about the website in the input field.
5. The app will display the answer based on the information available on the website.

## Limitations

- The app may not work properly for websites that require authentication or have dynamic content.
- The app may not be able to answer questions that require reasoning or common sense knowledge.

## License

This project is licensed under the MIT License.

## Acknowledgments

- [Streamlit](https://streamlit.io/) for building the app interface.
- [Hugging Face](https://huggingface.co/) for providing the language model and embedding services.
- [Langchain](https://github.com/hwchase17/langchain) and [Langchain-community](https://github.com/hwchase17/langchain-community) for providing the tools and utilities for building the app.
