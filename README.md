# Website Chat

Website Chat is a Streamlit application that allows you to ask questions about a website and get answers based on the information available on the website. It uses Hugging Face's Mistral-7B-v0.1 language model for question-answering and FAISS for document embedding and retrieval.


## Why I Built This Project

As the amount of information available on the internet continues to grow, it becomes increasingly challenging to find relevant and accurate information quickly. Large Language Models (LLMs) like the one used in this application are typically trained on historical data, which means they may not have access to the latest, up-to-date information available on websites.

By combining the power of LLMs with the ability to extract and process information directly from websites in real-time, this application ensures that the answers provided are not only accurate but also reflect the most recent updates and changes to the website's content. This is particularly valuable for websites that are frequently updated or contain time-sensitive information, such as news portals, product catalogs, or government websites.

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
git clone https://github.com/Shahroz192/chat-with-website.git
```
2. Install the required packages:
```
pip install -r requirements.txt
```
3. Obtain a Hugging Face API token from the [Hugging Face website](https://huggingface.co/settings/tokens) and provide it as an input in the Streamlit app.

## Usage

1. Run the Streamlit app:
```
streamlit run app.py
```
2. Enter your Hugging Face API token in the .env file.
3. Enter the URL of the website you want to ask questions about in sidebar.
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
