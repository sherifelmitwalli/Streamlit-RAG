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
import re  # For regex extraction

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
CHUNK_SIZE = 500                  # Maximum characters per text chunk
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
st.sidebar.info("This app uses AI-assisted Retrieval-Augmented Generation (RAG) to answer questions based on uploaded files or folders.")
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
def process_json_file(file_bytes: BytesIO, file_name: str) -> List[Dict[str, Any]]:
    """
    Process a JSON file and convert it to text chunks with metadata.
    Supports dictionaries (with keys like "page_1", etc.), lists, or other JSON types.
    """
    chunks = []
    try:
        file_bytes.seek(0)
        data = json.load(file_bytes)
    except json.JSONDecodeError as jde:
        error_msg = f"JSON decoding error in file {file_name}: {str(jde)}"
        logger.error(error_msg, exc_info=True)
        st.error(error_msg)
        return []
    except Exception as e:
        error_msg = f"Unexpected error reading JSON file {file_name}: {str(e)}"
        logger.error(error_msg, exc_info=True)
        st.error(error_msg)
        return []
    
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(key, str) and key.lower().startswith("page_"):
                page_num = key.split("_")[-1]
                chunk_text = f"File: {file_name} | Page: {page_num}\n{value}"
                chunks.append({"text": chunk_text, "source": {"file": file_name, "page": page_num}})
            else:
                chunk_text = f"File: {file_name} | {key}\n{value}"
                chunks.append({"text": chunk_text, "source": {"file": file_name, "key": key}})
    elif isinstance(data, list):
        for index, item in enumerate(data, start=1):
            chunk_text = f"File: {file_name} | Item {index}\n{json.dumps(item, indent=2)}"
            chunks.append({"text": chunk_text, "source": {"file": file_name, "item": index}})
    else:
        chunk_text = f"File: {file_name}\n{json.dumps(data, indent=2)}"
        chunks.append({"text": chunk_text, "source": {"file": file_name}})
    
    return chunks

def process_file_bytes(file_bytes: BytesIO, file_name: str) -> Optional[List[Dict[str, Any]]]:
    """
    Processes file bytes and returns a list of chunk dictionaries, each with text and source metadata.
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
                    chunks.append({"text": f"File: {file_name} | Page: {i+1}\n{page_text}", "source": {"file": file_name, "page": i+1}})
            return chunks
        elif ext == "csv":
            file_bytes.seek(0)
            df = pd.read_csv(file_bytes)
            text = df.to_string()
            return [{"text": f"File: {file_name}\n{text}", "source": {"file": file_name}}]
        elif ext == "pptx":
            file_bytes.seek(0)
            presentation = Presentation(file_bytes)
            text = "\n".join(
                shape.text for slide in presentation.slides 
                for shape in slide.shapes if hasattr(shape, "text")
            )
            return [{"text": f"File: {file_name}\n{text}", "source": {"file": file_name}}]
        elif ext == "json":
            return process_json_file(file_bytes, file_name)
        else:
            st.error(f"Unsupported file extension: {ext} for file {file_name}")
            return None
    except Exception as e:
        logger.error(f"Error processing file {file_name}: {str(e)}", exc_info=True)
        st.error(f"Error processing file {file_name}. Error: {str(e)}")
        return None

def process_file(uploaded_file: st.runtime.uploaded_file_manager.UploadedFile) -> Optional[List[Dict[str, Any]]]:
    """
    Wraps process_file_bytes to read the uploaded file and returns a list of chunk dictionaries.
    """
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

def find_relevant_context_indices(query: str, text_chunks: List[str], embeddings: np.ndarray, top_k: int = TOP_K) -> List[int]:
    """
    Returns the indices of the top_k text_chunks most similar to the query.
    """
    try:
        response = openai.Embedding.create(
            input=[query],
            model=EMBEDDING_MODEL
        )
        query_embedding = np.array(response['data'][0]['embedding']).reshape(1, -1)
        similarities = cosine_similarity(query_embedding, embeddings)[0]
        top_indices = similarities.argsort()[-top_k:][::-1]
        return list(top_indices)
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

def sort_chunks(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Sort chunks by file name (alphabetically) and by page number (numerically, if available).
    """
    def chunk_sort_key(chunk):
        file_name = chunk["source"].get("file", "").lower()
        page = chunk["source"].get("page")
        try:
            page_num = int(page) if page is not None and str(page).strip().isdigit() else 0
        except Exception:
            page_num = 0
        return (file_name, page_num)
    return sorted(chunks, key=chunk_sort_key)

def extract_exact_mentions(chunks: List[Dict[str, Any]], search_term: str,
                           context_words: int = 3) -> List[Dict[str, Any]]:
    """
    Extract all exact occurrences of the search_term (with a small context window)
    from the provided chunks. Returns a list of dictionaries with file, page, and snippet.
    """
    # Build a regex pattern using the provided search_term.
    pattern = rf'((?:\S+\s+){{0,{context_words}}}{re.escape(search_term)}(?:\s+\S+){{0,{context_words}}})'
    results = []
    for chunk in chunks:
        text = chunk.get("text", "")
        for match in re.finditer(pattern, text, re.IGNORECASE):
            snippet = ' '.join(match.group(0).split())  # collapse extra whitespace and newlines
            file_name = chunk["source"].get("file", "unknown file")
            page = chunk["source"].get("page")
            if page:
                page = str(page).strip()
                m = re.search(r'\d+', page)
                if m:
                    page = m.group(0)
                else:
                    page = "N/A"
            else:
                page = "N/A"
            results.append({
                "file": file_name,
                "page": page,
                "snippet": snippet
            })
    def result_sort_key(item):
        f = item.get("file", "").lower()
        p = item.get("page")
        try:
            p_num = int(p) if p != "N/A" and str(p).isdigit() else 0
        except Exception:
            p_num = 0
        return (f, p_num)
    return sorted(results, key=result_sort_key)

# -----------------------------
# File Upload & Processing
# -----------------------------
uploaded_files = st.file_uploader(
    "Upload one or more files or a ZIP file containing a folder (.txt, .pdf, .csv, .pptx, .json, .zip):",
    type=["txt", "pdf", "csv", "pptx", "json", "zip"],
    accept_multiple_files=True
)

if uploaded_files and "embeddings" not in st.session_state:
    if "chunks" not in st.session_state:
        st.session_state.chunks = []
    
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
                                    chunks = process_file_bytes(file_bytes, zip_info.filename)
                                    if chunks:
                                        st.session_state.chunks.extend(chunks)
                                        st.success(f"Processed file from folder: {zip_info.filename}")
                                else:
                                    st.warning(f"Skipping unsupported file {zip_info.filename} in the ZIP.")
                except Exception as e:
                    st.error(f"Failed to extract ZIP file {uploaded_file.name}: {str(e)}")
        else:
            with st.spinner(f"Processing file {uploaded_file.name}..."):
                chunks = process_file(uploaded_file)
                if chunks:
                    st.session_state.chunks.extend(chunks)
                    st.success(f"Processed file: {uploaded_file.name}")

    # Optionally split large chunks further.
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
    with st.spinner("Generating embeddings..."):
        embeddings = generate_embeddings(text_for_embedding)
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

    # Check if the query asks for exact matches.
    if "exact match" in prompt.lower():
        search_match = re.search(r'exact matches? of (?:the name|the word)?\s*[\'"]?(\w+)[\'"]?', prompt, re.IGNORECASE)
        if search_match:
            search_term = search_match.group(1)
        else:
            st.error("Could not determine the search term from your query. Please include it (e.g., exact matches of the word 'mcculloch').")
            search_term = None

        if search_term:
            exact_results = extract_exact_mentions(st.session_state.chunks, search_term)
            if exact_results:
                response_lines = []
                for idx, res in enumerate(exact_results, start=1):
                    response_lines.append(f"{idx}. File: {res['file']}, Page: {res['page']}\n   {res['snippet']}")
                bot_response = "\n\n".join(response_lines)
            else:
                bot_response = "No exact matches found."
            st.session_state.messages.append({"role": "assistant", "content": bot_response})
            with st.chat_message("assistant"):
                st.markdown(bot_response)
    else:
        file_match = re.search(r'file\s+([A-Za-z0-9\-]+)', prompt, re.IGNORECASE)
        if file_match:
            file_id = file_match.group(1)
            filtered_indices = [
                i for i, chunk in enumerate(st.session_state.chunks)
                if file_id.lower() in chunk["source"].get("file", "").lower()
            ]
            if filtered_indices:
                filtered_chunks = [st.session_state.chunks[i] for i in filtered_indices]
                filtered_text_chunks = [chunk["text"] for chunk in filtered_chunks]
                filtered_embeddings = st.session_state.embeddings[filtered_indices]
            else:
                filtered_chunks = st.session_state.chunks
                filtered_text_chunks = [chunk["text"] for chunk in st.session_state.chunks]
                filtered_embeddings = st.session_state.embeddings
        else:
            filtered_chunks = st.session_state.chunks
            filtered_text_chunks = [chunk["text"] for chunk in st.session_state.chunks]
            filtered_embeddings = st.session_state.embeddings

        with st.spinner("Retrieving relevant context..."):
            top_indices = find_relevant_context_indices(prompt, filtered_text_chunks, filtered_embeddings)
        relevant_chunks = [filtered_chunks[i] for i in top_indices]
        relevant_chunks_sorted = sort_chunks(relevant_chunks)
        formatted_contexts = []
        for chunk in relevant_chunks_sorted:
            file_ref = chunk["source"].get("file", "unknown file")
            page_ref = chunk["source"].get("page")
            if page_ref and str(page_ref).strip().isdigit():
                ref_str = f"File: {file_ref}, Page: {page_ref}"
            else:
                ref_str = f"File: {file_ref}, Page: N/A"
            formatted_context = f"{ref_str}\n{chunk['text']}"
            formatted_contexts.append(formatted_context)
        combined_context = "\n\n".join(formatted_contexts)

        with st.spinner("Generating response..."):
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Context:\n{combined_context}\n\nQuestion: {prompt}"}
            ]
            bot_response = get_chat_response(messages)
            if bot_response:
                st.session_state.messages.append({"role": "assistant", "content": bot_response})
                with st.chat_message("assistant"):
                    st.markdown(bot_response)
else:
    st.warning("Please upload file(s) and wait for embeddings to be generated before asking questions.")

