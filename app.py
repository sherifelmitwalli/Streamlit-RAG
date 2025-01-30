import streamlit as st
import os
import logging
import time
from io import BytesIO
from typing import List, Optional

import PyPDF2
import pandas as pd
import numpy as np
from pptx import Presentation
from sklearn.metrics.pairwise import cosine_similarity
import openai

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Set maximum file size (e.g., 50MB)
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB

# Load secrets with error handling
try:
    openai.api_key = st.secrets["OPENAI_API_KEY"]
    MODEL_NAME = st.secrets["MODEL"]
except KeyError as e:
    st.error(f"Missing secret key: {e}. Please configure the secrets properly.")
    st.stop()

# Constants
EMBEDDING_MODEL = "text-embedding-ada-002"  # Update if newer models are available
TOP_K = 3  # Number of top relevant contexts to retrieve

# Custom styling
st.markdown("""
<style>
div.stButton > button:first-child {
    background-color: #4CAF50;
    color: white;
    font-size: 16px;
    padding: 10px 24px;
    border-radius: 8px;
}
</style>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.header("About")
st.sidebar.info("This app uses AI-assisted Retrieval-Augmented Generation (RAG) to answer questions based on uploaded files.")
st.sidebar.markdown("[View Documentation](https://example.com)")

# Title
st.title("Agentic RAG Chatbot with GPT-4o-mini")
st.subheader("Upload a file and ask questions based on its content.")

# Function to process uploaded files
def process_file(file: BytesIO) -> Optional[str]:
    try:
        if file.type == "text/plain":
            return file.read().decode("utf-8")
        elif file.type == "application/pdf":
            reader = PyPDF2.PdfReader(file)
            text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
            return text
        elif file.type == "text/csv":
            df = pd.read_csv(file)
            return df.to_string()
        elif file.type == "application/vnd.openxmlformats-officedocument.presentationml.presentation":
            presentation = Presentation(file)
            text = "\n".join(
                shape.text for slide in presentation.slides for shape in slide.shapes if hasattr(shape, "text")
            )
            return text
        else:
            st.error("Unsupported file type. Please upload a .txt, .pdf, .csv, or .pptx file.")
            return None
    except Exception as e:
        logger.error(f"Error processing file: {e}", exc_info=True)
        st.error("An error occurred while processing the file. Please try a different file or contact support.")
        return None

# Caching embeddings to avoid redundant API calls
@st.cache_data(show_spinner=False)
def generate_embeddings(text_chunks: List[str]) -> np.ndarray:
    try:
        response = openai.Embedding.create(
            input=text_chunks,
            model=EMBEDDING_MODEL
        )
        embeddings = [data['embedding'] for data in response['data']]
        return np.array(embeddings)
    except Exception as e:
        logger.error(f"Error generating embeddings: {e}", exc_info=True)
        st.error("Failed to generate embeddings. Please try again later.")
        return np.array([])

# Function to find top-k relevant contexts using cosine similarity
def find_relevant_context(query: str, text_chunks: List[str], embeddings: np.ndarray, top_k: int = TOP_K) -> List[str]:
    try:
        response = openai.Embedding.create(
            input=[query],
            model=EMBEDDING_MODEL
        )
        query_embedding = np.array(response['data'][0]['embedding']).reshape(1, -1)
        similarities = cosine_similarity(query_embedding, embeddings)[0]
        top_indices = similarities.argsort()[-top_k:][::-1]
        top_contexts = [text_chunks[idx] for idx in top_indices]
        return top_contexts
    except Exception as e:
        logger.error(f"Error finding relevant context: {e}", exc_info=True)
        st.error("Failed to retrieve relevant context. Please try again later.")
        return []

# Function to get chat response with retries
def get_chat_response(messages: List[dict], retries: int = 3, delay: int = 5) -> Optional[str]:
    for attempt in range(retries):
        try:
            response = openai.ChatCompletion.create(
                model=MODEL_NAME,
                messages=messages
            )
            return response.choices[0]["message"]["content"]
        except openai.error.RateLimitError:
            st.warning(f"Rate limit exceeded. Retrying in {delay} seconds...")
            time.sleep(delay)
        except Exception as e:
            logger.error(f"OpenAI API error: {e}", exc_info=True)
            st.error("An error occurred while generating the response. Please try again later.")
            return None
    st.error("Failed to generate response after multiple attempts.")
    return None

# File upload with size limit
uploaded_file = st.file_uploader(
    "Upload a file (.txt, .pdf, .csv, .pptx):",
    type=["txt", "pdf", "csv", "pptx"],
    accept_multiple_files=False
)

if uploaded_file:
    if uploaded_file.size > MAX_FILE_SIZE:
        st.error(f"File size exceeds the limit of {MAX_FILE_SIZE / (1024 * 1024)} MB.")
    else:
        with st.spinner("Processing file..."):
            file_text = process_file(uploaded_file)
            if file_text:
                st.success("File processed successfully!")
                
                # Clear previous session state
                if 'messages' in st.session_state:
                    del st.session_state.messages
                if 'embeddings' in st.session_state:
                    del st.session_state.embeddings
                if 'text_chunks' in st.session_state:
                    del st.session_state.text_chunks

                # Chunk the text
                text_chunks = [file_text[i:i+500] for i in range(0, len(file_text), 500)]
                st.session_state.text_chunks = text_chunks

                # Generate embeddings with progress
                with st.spinner("Generating embeddings..."):
                    embeddings = generate_embeddings(text_chunks)
                    if embeddings.size == 0:
                        st.error("Embeddings generation failed.")
                    else:
                        st.session_state.embeddings = embeddings
                        st.success("Embeddings generated successfully!")

# Initialize chat messages in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if prompt := st.chat_input("Ask a question about the uploaded file:"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if "embeddings" in st.session_state and "text_chunks" in st.session_state:
        with st.spinner("Retrieving relevant context..."):
            relevant_contexts = find_relevant_context(
                prompt,
                st.session_state.text_chunks,
                st.session_state.embeddings,
                top_k=TOP_K
            )
        if relevant_contexts:
            combined_context = "\n\n".join(relevant_contexts)
            with st.spinner("Generating response..."):
                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": f"Context: {combined_context}\n\nQuestion: {prompt}"}
                ]
                bot_response = get_chat_response(messages)
                if bot_response:
                    st.session_state.messages.append({"role": "assistant", "content": bot_response})
                    with st.chat_message("assistant"):
                        st.markdown(bot_response)
    else:
        st.warning("Please upload a file and wait for embeddings to be generated before asking questions.")

