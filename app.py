import streamlit as st
import os
import logging
import time
from io import BytesIO
import zipfile
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
CHUNK_SIZE = 500  # Maximum characters per chunk
EMBEDDING_MODEL = "text-embedding-ada-002"
TOP_K = 3  # Number of relevant contexts to retrieve

# -----------------------------
# OpenAI Setup
# -----------------------------
try:
    openai.api_key = st.secrets["OPENAI_API_KEY"]
    MODEL_NAME = st.secrets["MODEL"]
except KeyError:
    st.error("Missing OpenAI API Key. Please check your configuration.")
    st.stop()

# -----------------------------
# File Processing Functions
# -----------------------------
def process_json_file(file_bytes: BytesIO, file_name: str):
    """
    Processes a JSON file.
    If the JSON is a dict with keys like 'page_1', 'page_2', etc.,
    it creates a separate chunk for each page; otherwise, it dumps the whole JSON.
    """
    try:
        file_bytes.seek(0)
        data = json.load(file_bytes)
        if not data:
            raise ValueError("JSON file is empty")
        chunks = []
        if isinstance(data, dict):
            for key, value in data.items():
                if key.lower().startswith("page_"):
                    page_num = key.split("_")[-1]
                    chunk_text = f"File: {file_name} | Page: {page_num}\n{value}"
                    chunks.append({"text": chunk_text, "source": {"file": file_name, "page": page_num}})
                else:
                    chunk_text = f"File: {file_name} | {key}\n{value}"
                    chunks.append({"text": chunk_text, "source": {"file": file_name}})
        else:
            # For non-dict JSON, dump the whole content.
            chunks = [{"text": json.dumps(data, indent=2), "source": {"file": file_name}}]
        return chunks
    except Exception as e:
        logger.error(f"Error processing JSON file {file_name}: {str(e)}", exc_info=True)
        return []

def process_file_bytes(file_bytes: BytesIO, file_name: str):
    """
    Processes file bytes based on the file extension.
    Returns a list of dictionaries with keys "text" and "source".
    """
    ext = file_name.split('.')[-1].lower()
    try:
        if ext == "txt":
            file_bytes.seek(0)
            text = file_bytes.read().decode("utf-8")
            return [{"text": f"File: {file_name}\n{text}", "source": {"file": file_name}}]
        elif ext == "pdf":
            file_bytes.seek(0)
            reader = PyPDF2.PdfReader(file_bytes)
            chunks = []
            for i, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text:
                    chunks.append({
                        "text": f"File: {file_name} | Page: {i+1}\n{page_text}",
                        "source": {"file": file_name, "page": i+1}
                    })
            return chunks
        elif ext == "csv":
            file_bytes.seek(0)
            df = pd.read_csv(file_bytes)
            text = df.to_string()
            return [{"text": f"File: {file_name}\n{text}", "source": {"file": file_name}}]
        elif ext == "pptx":
            file_bytes.seek(0)
            prs = Presentation(file_bytes)
            text = "\n".join(
                shape.text for slide in prs.slides for shape in slide.shapes if hasattr(shape, "text")
            )
            return [{"text": f"File: {file_name}\n{text}", "source": {"file": file_name}}]
        elif ext == "json":
            return process_json_file(file_bytes, file_name)
        else:
            st.warning(f"Unsupported file format: {file_name}")
            return None
    except Exception as e:
        logger.error(f"Error processing {file_name}: {str(e)}", exc_info=True)
        return None

def extract_zip(uploaded_file):
    """
    Extracts supported files from a ZIP archive and returns a list of chunks.
    """
    extracted_chunks = []
    try:
        with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
            for zip_info in zip_ref.infolist():
                if not zip_info.is_dir():
                    ext = zip_info.filename.split('.')[-1].lower()
                    if ext in ["txt", "pdf", "csv", "pptx", "json"]:
                        file_content = zip_ref.read(zip_info.filename)
                        file_bytes = BytesIO(file_content)
                        chunks = process_file_bytes(file_bytes, zip_info.filename)
                        if chunks:
                            extracted_chunks.extend(chunks)
    except Exception as e:
        st.error(f"Failed to extract ZIP file: {str(e)}")
    return extracted_chunks

# -----------------------------
# Embedding & Retrieval Functions
# -----------------------------
@st.cache_data(ttl=3600)
def generate_embeddings(text_chunks):
    try:
        response = openai.Embedding.create(
            input=text_chunks,
            model=EMBEDDING_MODEL
        )
        return np.array([data['embedding'] for data in response['data']])
    except Exception as e:
        logger.error(f"Error generating embeddings: {str(e)}", exc_info=True)
        return np.array([])

def find_relevant_context(query, text_chunks, embeddings):
    try:
        response = openai.Embedding.create(input=[query], model=EMBEDDING_MODEL)
        query_embedding = np.array(response['data'][0]['embedding']).reshape(1, -1)
        similarities = cosine_similarity(query_embedding, embeddings)[0]
        top_indices = similarities.argsort()[-TOP_K:][::-1]
        return [text_chunks[idx] for idx in top_indices]
    except Exception as e:
        logger.error(f"Error retrieving context: {str(e)}", exc_info=True)
        return []

def get_chat_response(messages):
    try:
        response = openai.ChatCompletion.create(model=MODEL_NAME, messages=messages)
        return response.choices[0]["message"]["content"]
    except Exception as e:
        logger.error(f"Chat response error: {str(e)}", exc_info=True)
        return None

# -----------------------------
# Streamlit UI and File Upload
# -----------------------------
st.title("Agentic RAG Chatbot")
st.subheader("Upload files and ask questions based on their content.")

uploaded_files = st.file_uploader(
    "Upload files or a ZIP archive:",
    type=["txt", "pdf", "csv", "pptx", "json", "zip"],
    accept_multiple_files=True
)

if uploaded_files:
    if "chunks" not in st.session_state:
        st.session_state.chunks = []

    for uploaded_file in uploaded_files:
        if uploaded_file.size > MAX_FILE_SIZE:
            st.error(f"{uploaded_file.name} exceeds size limit!")
            continue

        if uploaded_file.name.lower().endswith(".zip"):
            st.session_state.chunks.extend(extract_zip(uploaded_file))
        else:
            file_content = uploaded_file.read()
            chunks = process_file_bytes(BytesIO(file_content), uploaded_file.name)
            if chunks:
                st.session_state.chunks.extend(chunks)

    # Optional: Split very large chunks into smaller sub-chunks
    final_chunks = []
    for chunk in st.session_state.chunks:
        text = chunk["text"]
        if len(text) > CHUNK_SIZE:
            for i in range(0, len(text), CHUNK_SIZE):
                sub_text = text[i:i+CHUNK_SIZE]
                final_chunks.append({"text": sub_text, "source": chunk["source"]})
        else:
            final_chunks.append(chunk)
    st.session_state.chunks = final_chunks

    text_for_embedding = [chunk["text"] for chunk in st.session_state.chunks]
    if text_for_embedding:
        st.session_state.embeddings = generate_embeddings(text_for_embedding)
        st.success("Embeddings generated successfully!")

# -----------------------------
# Chat Interface
# -----------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about the uploaded content:"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if "embeddings" in st.session_state and st.session_state.embeddings.size > 0:
        relevant_contexts = find_relevant_context(
            prompt,
            [chunk["text"] for chunk in st.session_state.chunks],
            st.session_state.embeddings
        )
        if relevant_contexts:
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Context: {relevant_contexts}\n\nQuestion: {prompt}"}
            ]
            response = get_chat_response(messages)
            if response:
                st.session_state.messages.append({"role": "assistant", "content": response})
                with st.chat_message("assistant"):
                    st.markdown(response)
    else:
        st.warning("Upload files first and wait for processing.")

