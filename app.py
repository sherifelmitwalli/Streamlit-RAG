import streamlit as st
import os
import logging
import time
from io import BytesIO
from typing import List, Optional, Dict, Any

import PyPDF2
import pandas as pd
import numpy as np
from pptx import Presentation
from sklearn.metrics.pairwise import cosine_similarity
import openai

# Initialize logging with more detailed formatting and file output
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("app.log", mode="a")
    ]
)
logger = logging.getLogger(__name__)

# Memory management
def clear_cache() -> None:
    """Clear all Streamlit cache to free up memory."""
    try:
        st.cache_data.clear()
        logger.info("Cache cleared successfully")
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")

# Constants
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB
CHUNK_SIZE = 500  # Size of text chunks for processing
MAX_RETRIES = 3  # Maximum number of API call retries
RETRY_DELAY = 5  # Delay between retries in seconds
EMBEDDING_MODEL = "text-embedding-ada-002"
TOP_K = 3  # Number of top relevant contexts to retrieve

# Load secrets with enhanced error handling
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
    st.error(f"""
    Configuration Error: Missing required secrets.
    
    Please ensure you have configured in Streamlit Cloud:
    1. OPENAI_API_KEY
    2. MODEL (should be "gpt-4o-mini")
    
    Go to your app settings in Streamlit Cloud:
    1. Click on "Settings" ⚙️
    2. Select "Secrets"
    3. Add these secrets:
    ```toml
    MODEL = "gpt-4o-mini"
    OPENAI_API_KEY = "your-openai-api-key"
    ```
    Note: Do not include any section headers like [openai] in the secrets.
    """)
    st.stop()

# Validate API key by making a test request
try:
    # Simple API test using embeddings endpoint
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

# Custom styling
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

# Application information
st.sidebar.header("About")
st.sidebar.info("This app uses AI-assisted Retrieval-Augmented Generation (RAG) to answer questions based on uploaded files.")
st.sidebar.markdown("[View Documentation](https://example.com)")

st.title("Agentic RAG Chatbot")
st.subheader("Upload a file and ask questions based on its content.")

def process_file(uploaded_file: st.runtime.uploaded_file_manager.UploadedFile) -> Optional[str]:
    """
    Process uploaded file and extract text content.
    
    Args:
        uploaded_file: Streamlit UploadedFile object
        
    Returns:
        Optional[str]: Extracted text content or None if processing fails
    """
    try:
        # Create a BytesIO object from the uploaded file's bytes
        file_bytes = BytesIO(uploaded_file.read())
        
        if uploaded_file.type == "text/plain":
            # Reset pointer for text files
            file_bytes.seek(0)
            return file_bytes.read().decode("utf-8")
            
        elif uploaded_file.type == "application/pdf":
            # Reset pointer for PDF
            file_bytes.seek(0)
            reader = PyPDF2.PdfReader(file_bytes)
            text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
            return text
            
        elif uploaded_file.type == "text/csv":
            # Reset pointer for CSV
            file_bytes.seek(0)
            df = pd.read_csv(file_bytes)
            return df.to_string()
            
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.presentationml.presentation":
            # Reset pointer for PPTX
            file_bytes.seek(0)
            presentation = Presentation(file_bytes)
            text = "\n".join(
                shape.text for slide in presentation.slides 
                for shape in slide.shapes if hasattr(shape, "text")
            )
            return text
            
        else:
            st.error(f"Unsupported file type: {uploaded_file.type}. Please upload a .txt, .pdf, .csv, or .pptx file.")
            return None
    except Exception as e:
        logger.error(f"Error processing file {uploaded_file.name}: {str(e)}", exc_info=True)
        st.error(f"Error processing file {uploaded_file.name}. Error: {str(e)}")
        return None
    finally:
        # Clean up
        file_bytes.close()

@st.cache_data(show_spinner=False, ttl=3600)  # Cache for 1 hour
def generate_embeddings(text_chunks: List[str]) -> np.ndarray:
    """
    Generate embeddings for text chunks using OpenAI's API.
    
    Args:
        text_chunks (List[str]): List of text segments to embed
        
    Returns:
        np.ndarray: Array of embeddings or empty array if generation fails
    """
    try:
        # Process chunks in smaller batches to avoid rate limits
        batch_size = 20
        all_embeddings = []
        
        for i in range(0, len(text_chunks), batch_size):
            batch = text_chunks[i:i + batch_size]
            
            # Filter out empty chunks and ensure text is clean
            batch = [chunk.strip() for chunk in batch if chunk.strip()]
            if not batch:
                continue
                
            try:
                response = openai.Embedding.create(
                    input=batch,
                    model=EMBEDDING_MODEL
                )
                batch_embeddings = [data['embedding'] for data in response['data']]
                all_embeddings.extend(batch_embeddings)
                
                # Add a small delay to avoid rate limits
                if i + batch_size < len(text_chunks):
                    time.sleep(0.5)
                    
            except openai.error.RateLimitError:
                st.warning("Rate limit reached. Waiting before retrying...")
                time.sleep(20)  # Wait longer on rate limit
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

def find_relevant_context(
    query: str,
    text_chunks: List[str],
    embeddings: np.ndarray,
    top_k: int = TOP_K
) -> List[str]:
    """
    Find most relevant context for a query using cosine similarity.
    
    Args:
        query (str): User's question
        text_chunks (List[str]): Available text segments
        embeddings (np.ndarray): Pre-computed embeddings
        top_k (int): Number of relevant chunks to retrieve
        
    Returns:
        List[str]: Most relevant text chunks
    """
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

def get_chat_response(
    messages: List[Dict[str, str]],
    retries: int = MAX_RETRIES,
    delay: int = RETRY_DELAY
) -> Optional[str]:
    """
    Get response from chat model with retry mechanism.
    
    Args:
        messages (List[Dict[str, str]]): Conversation messages
        retries (int): Number of retry attempts
        delay (int): Delay between retries in seconds
        
    Returns:
        Optional[str]: Model's response or None if all retries fail
    """
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

# File upload with size limit
uploaded_file = st.file_uploader(
    "Upload a file (.txt, .pdf, .csv, .pptx):",
    type=["txt", "pdf", "csv", "pptx"],
    accept_multiple_files=False
)

if uploaded_file:
    if uploaded_file.size > MAX_FILE_SIZE:
        st.error(f"File size exceeds the limit of {MAX_FILE_SIZE / (1024 * 1024):.1f} MB.")
    else:
        with st.spinner("Processing file..."):
            file_text = process_file(uploaded_file)
            if file_text:
                st.success("File processed successfully!")
                
                # Clear previous session state and cache
                if 'messages' in st.session_state:
                    del st.session_state.messages
                if 'embeddings' in st.session_state:
                    del st.session_state.embeddings
                if 'text_chunks' in st.session_state:
                    del st.session_state.text_chunks
                clear_cache()

                # Chunk the text
                text_chunks = [file_text[i:i+CHUNK_SIZE] for i in range(0, len(file_text), CHUNK_SIZE)]
                st.session_state.text_chunks = text_chunks

                # Generate embeddings with progress
                with st.spinner("Generating embeddings..."):
                    embeddings = generate_embeddings(text_chunks)
                    if embeddings.size > 0:
                        st.session_state.embeddings = embeddings
                        st.success("Embeddings generated successfully!")
                    else:
                        st.error("Failed to generate embeddings. Please try again.")

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
        st.warning("Please upload a file and wait for embeddings to be generated before asking questions.")

