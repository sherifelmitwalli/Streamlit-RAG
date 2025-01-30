import streamlit as st
import os
import PyPDF2
import pandas as pd
import numpy as np
from pptx import Presentation
from sklearn.metrics.pairwise import cosine_similarity
import openai

# Load secrets
openai.api_key = st.secrets["OPENAI_API_KEY"]
MODEL_NAME = st.secrets["MODEL"]

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
st.sidebar.info("This app uses AI-assisted RAG to answer questions based on uploaded files.")
st.sidebar.markdown("[View Documentation](https://example.com)")

# Title
st.title("Agentic RAG Chatbot with GPT-4o-mini")
st.subheader("Upload a file and ask questions based on its content.")

# Function to process uploaded files
def process_file(file):
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
        st.error(f"Error processing file: {str(e)}")
        return None

# Function to generate embeddings
def generate_embeddings(text_chunks):
    embeddings = []
    for chunk in text_chunks:
        response = openai.embeddings.create(
            input=chunk,
            model="text-embedding-ada-002"
        )
        embeddings.append(response.data[0].embedding)
    return np.array(embeddings)

# Function to find relevant context using cosine similarity
def find_relevant_context(query, text_chunks, embeddings):
    query_embedding = openai.embeddings.create(
        input=query,
        model="text-embedding-ada-002"
    ).data[0].embedding

    similarities = cosine_similarity([query_embedding], embeddings)[0]
    best_match_idx = np.argmax(similarities)
    return text_chunks[best_match_idx]

# File upload
uploaded_file = st.file_uploader("Upload a file (.txt, .pdf, .csv, .pptx):", type=["txt", "pdf", "csv", "pptx"])

if uploaded_file:
    with st.spinner("Processing file..."):
        file_text = process_file(uploaded_file)
        if file_text:
            st.success("File processed successfully!")

            # Chunk the text
            text_chunks = [file_text[i:i+500] for i in range(0, len(file_text), 500)]

            # Generate embeddings and store in session state
            if "embeddings" not in st.session_state:
                st.session_state.embeddings = generate_embeddings(text_chunks)
                st.session_state.text_chunks = text_chunks

# Chatbot interface
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if prompt := st.chat_input("Ask a question about the uploaded file:"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if "embeddings" in st.session_state:
        with st.spinner("Finding relevant context..."):
            relevant_context = find_relevant_context(
                prompt,
                st.session_state.text_chunks,
                st.session_state.embeddings
            )

        with st.spinner("Generating response..."):
            try:
                response = openai.ChatCompletion.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": f"Context: {relevant_context}\n\nQuestion: {prompt}"}
                    ]
                )
                bot_response = response.choices[0]["message"]["content"]
            except Exception as e:
                bot_response = f"Error generating response: {str(e)}"

        st.session_state.messages.append({"role": "assistant", "content": bot_response})
        with st.chat_message("assistant"):
            st.markdown(bot_response)
    else:
        st.warning("Please upload a file before asking questions.")
