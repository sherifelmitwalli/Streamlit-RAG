import streamlit as st
import os
import logging
import time
from io import BytesIO
from typing import List, Optional, Dict, Any
import zipfile  # For handling ZIP files
import PyPDF2
import pandas as pd
import numpy as np
from pptx import Presentation
from sklearn.metrics.pairwise import cosine_similarity
import openai
import json

# -----------------------------
# Logging & Constants
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("app.log", mode="a")
    ]
)
logger = logging.getLogger(__name__)

MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB
CHUNK_SIZE = 500                  # Size of text chunks for processing
MAX_RETRIES = 3                   # Maximum number of API call retries
RETRY_DELAY = 5                   # Delay between retries in seconds
EMBEDDING_MODEL = "text-embedding-ada-002"
TOP_K = 3                         # Number of top relevant contexts to retrieve

# -----------------------------
# OpenAI & Secrets Setup
# -----------------------------
try:
    openai.api_key = st.secrets["OPENAI_API_KEY"]
    MODEL_NAME = st.secrets["MODEL"]
    if not openai.api_key:
        raise ValueError("OPENAI_API_KEY not configured in Streamlit secrets")
    if not MODEL_NAME:
        raise ValueError("MODEL not configured in Streamlit secrets")
except (KeyError, ValueError) as e:
    error_msg = str(e)
    logger.error(f"Configuration error: {error_msg}")
    st.error("Configuration Error: Missing required secrets.")
    st.stop()

# Validate API key by making a test request
try:
    response = openai.Embedding.create(
        input="test",
        model=EMBEDDING_MODEL
    )
    if response and response.get('data'):
        logger.info("Successfully connected to OpenAI API")
    else:
        raise ValueError("Invalid API response")
except Exception as e:
    error_msg = f"Failed to connect to OpenAI API: {str(e)}"
    logger.error(error_msg)
    st.error(error_msg)
    st.stop()

# -----------------------------
# Custom Styling & Sidebar
# -----------------------------
st.markdown('''
<style>
div.stButton > button:first-child {
    background-color: #4CAF50;
    color: white;
    font-size: 16px;
    padding: 10px 24px;
    border-radius: 8px;
}
</style>
''', unsafe_allow_html=True)

st.sidebar.header("About")
st.sidebar.info("This app uses AI-assisted Retrieval-Augmented Generation (RAG) developed by TCRG to answer questions based on uploaded files or folders.")
st.sidebar.markdown("[View Documentation](https://example.com)")

st.title("Agentic RAG Chatbot")
st.subheader("Upload one or more files or a folder (as a ZIP file) and ask questions based on their content.")

# -----------------------------
# Cache/Clear Helper
# -----------------------------
def clear_cache() -> None:
    try:
        st.cache_data.clear()
        logger.info("Cache cleared successfully")
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")

# -----------------------------
# File Processing Functions
# -----------------------------
def process_file_bytes(file_bytes: BytesIO, file_name: str) -> Optional[str]:
    ext = file_name.split('.')[-1].lower()
    try:
        if ext == "txt":
            file_bytes.seek(0)
            return file_bytes.read().decode("utf-8")
        elif ext == "pdf":
            file_bytes.seek(0)
            reader = PyPDF2.PdfReader(file_bytes)
            text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
            return text
        elif ext == "csv":
            file_bytes.seek(0)
            df = pd.read_csv(file_bytes)
            return df.to_string()
        elif ext == "pptx":
            file_bytes.seek(0)
            presentation = Presentation(file_bytes)
            text = "\n".join(
                shape.text for slide in presentation.slides 
                for shape in slide.shapes if hasattr(shape, "text")
            )
            return text
        elif ext == "json":
            file_bytes.seek(0)
            data = json.load(file_bytes)
            return json.dumps(data, indent=2)
        else:
            st.error(f"Unsupported file extension: {ext} for file {file_name}")
            return None
    except Exception as e:
        logger.error(f"Error processing file {file_name}: {str(e)}", exc_info=True)
        st.error(f"Error processing file {file_name}. Error: {str(e)}")
        return None

def process_file(uploaded_file: st.runtime.uploaded_file_manager.UploadedFile) -> Optional[str]:
    try:
        file_bytes = BytesIO(uploaded_file.read())
        return process_file_bytes(file_bytes, uploaded_file.name)
    except Exception as e:
        logger.error(f"Error processing file {uploaded_file.name}: {str(e)}", exc_info=True)
        st.error(f"Error processing file {uploaded_file.name}. Error: {str(e)}")
        return None
    finally:
        file_bytes.close()

# -----------------------------
# Embedding & Chat Functions
# -----------------------------
@st.cache_data(show_spinner=False, ttl=3600)
def generate_embeddings(text_chunks: List[str]) -> np.ndarray:
    try:
        batch_size = 20
        all_embeddings = []
        for i in range(0, len(text_chunks), batch_size):
            batch = [chunk.strip() for chunk in text_chunks[i:i + batch_size] if chunk.strip()]
            if not batch:
                continue
            try:
                response = openai.Embedding.create(
                    input=batch,
                    model=EMBEDDING_MODEL
                )
                batch_embeddings = [data['embedding'] for data in response['data']]
                all_embeddings.extend(batch_embeddings)
                if i + batch_size < len(text_chunks):
                    time.sleep(0.5)
            except openai.error.RateLimitError:
                st.warning("Rate limit reached. Waiting before retrying...")
                time.sleep(20)
                response = openai.Embedding.create(
                    input=batch,
                    model=EMBEDDING_MODEL
                )
                batch_embeddings = [data['embedding'] for data in response['data']]
                all_embeddings.extend(batch_embeddings)
        if not all_embeddings:
            raise ValueError("No valid embeddings were generated")
        return np.array(all_embeddings)
    except openai.error.AuthenticationError:
        logger.error("OpenAI API authentication failed")
        st.error("Failed to authenticate with OpenAI API. Please check your API key configuration.")
        return np.array([])
    except openai.error.InvalidRequestError as e:
        logger.error(f"Invalid request to OpenAI API: {e}")
        st.error(f"Invalid request: {str(e)}")
        return np.array([])
    except Exception as e:
        logger.error(f"Error generating embeddings: {str(e)}", exc_info=True)
        st.error(f"Failed to generate embeddings: {str(e)}")
        return np.array([])

def find_relevant_context(query: str, text_chunks: List[str], embeddings: np.ndarray, top_k: int = TOP_K) -> List[str]:
    try:
        response = openai.Embedding.create(
            input=[query],
            model=EMBEDDING_MODEL
        )
        query_embedding = np.array(response['data'][0]['embedding']).reshape(1, -1)
        similarities = cosine_similarity(query_embedding, embeddings)[0]
        top_indices = similarities.argsort()[-top_k:][::-1]
        return [text_chunks[idx] for idx in top_indices]
    except Exception as e:
        logger.error(f"Error finding relevant context: {e}", exc_info=True)
        st.error("Failed to retrieve relevant context. Please try again later.")
        return []

def get_chat_response(messages: List[Dict[str, str]], retries: int = MAX_RETRIES, delay: int = RETRY_DELAY) -> Optional[str]:
    for attempt in range(retries):
        try:
            response = openai.ChatCompletion.create(
                model=MODEL_NAME,
                messages=messages
            )
            return response.choices[0]["message"]["content"]
        except openai.error.RateLimitError:
            if attempt < retries - 1:
                st.warning(f"Rate limit exceeded. Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                st.error("Rate limit exceeded. Please try again later.")
        except Exception as e:
            logger.error(f"OpenAI API error: {e}", exc_info=True)
            st.error("An error occurred while generating the response. Please try again later.")
            return None
    return None

# -----------------------------
# File Upload & Processing
# -----------------------------
uploaded_files = st.file_uploader(
    "Upload one or more files or a ZIP file containing a folder (.txt, .pdf, .csv, .pptx, .json, .zip):",
    type=["txt", "pdf", "csv", "pptx", "json", "zip"],
    accept_multiple_files=True
)

# Process files only if new files are uploaded and embeddings have not been generated yet.
if uploaded_files and "embeddings" not in st.session_state:
    if "combined_text" not in st.session_state:
        st.session_state.combined_text = ""

    for uploaded_file in uploaded_files:
        if uploaded_file.size > MAX_FILE_SIZE:
            st.error(f"File {uploaded_file.name} exceeds the size limit of {MAX_FILE_SIZE / (1024 * 1024):.1f} MB.")
            continue

        if uploaded_file.name.lower().endswith(".zip"):
            with st.spinner(f"Extracting folder from {uploaded_file.name}..."):
                try:
                    with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
                        for zip_info in zip_ref.infolist():
                            if not zip_info.is_dir():
                                ext = zip_info.filename.split('.')[-1].lower()
                                if ext in ["txt", "pdf", "csv", "pptx", "json"]:
                                    file_content = zip_ref.read(zip_info.filename)
                                    file_bytes = BytesIO(file_content)
                                    text = process_file_bytes(file_bytes, zip_info.filename)
                                    if text:
                                        st.session_state.combined_text += "\n\n" + text
                                        st.success(f"Processed file from folder: {zip_info.filename}")
                                else:
                                    st.warning(f"Skipping unsupported file {zip_info.filename} in the ZIP.")
                except Exception as e:
                    st.error(f"Failed to extract ZIP file {uploaded_file.name}: {str(e)}")
        else:
            with st.spinner(f"Processing file {uploaded_file.name}..."):
                file_text = process_file(uploaded_file)
                if file_text:
                    st.session_state.combined_text += "\n\n" + file_text
                    st.success(f"Processed file: {uploaded_file.name}")

    # Only proceed if there is accumulated text.
    if st.session_state.combined_text:
        # Clear any previous embedding-related session state.
        if 'embeddings' in st.session_state:
            del st.session_state.embeddings
        if 'text_chunks' in st.session_state:
            del st.session_state.text_chunks
        clear_cache()

        # Create text chunks.
        text_chunks = [st.session_state.combined_text[i:i+CHUNK_SIZE] 
                       for i in range(0, len(st.session_state.combined_text), CHUNK_SIZE)]
        st.session_state.text_chunks = text_chunks

        # Generate embeddings.
        with st.spinner("Generating embeddings..."):
            embeddings = generate_embeddings(text_chunks)
            if embeddings.size > 0:
                st.session_state.embeddings = embeddings
                st.success("Embeddings generated successfully!")
            else:
                st.error("Failed to generate embeddings. Please try again.")

# -----------------------------
# Chat Interface
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question about the uploaded content:"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if "embeddings" in st.session_state and "text_chunks" in st.session_state:
        with st.spinner("Retrieving relevant context..."):
            relevant_contexts = find_relevant_context(
                prompt,
                st.session_state.text_chunks,
                st.session_state.embeddings
            )
        if relevant_contexts:
            combined_context = "\n\n".join(relevant_contexts)
            with st.spinner("Generating response..."):
                messages = [
                    {"role": "system", "content": "You are a helpful assistant. Provide accurate and concise answers based on the given context."},
                    {"role": "user", "content": f"Context: {combined_context}\n\nQuestion: {prompt}"}
                ]
                bot_response = get_chat_response(messages)
                if bot_response:
                    st.session_state.messages.append({"role": "assistant", "content": bot_response})
                    with st.chat_message("assistant"):
                        st.markdown(bot_response)
    else:
        st.warning("Please upload file(s) and wait for embeddings to be generated before asking questions.")

