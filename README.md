# Agentic RAG Chatbot

A Streamlit application that uses Retrieval-Augmented Generation (RAG) with GPT-4 to answer questions based on uploaded documents.

## Features

- Support for multiple file formats (TXT, PDF, CSV, PPTX)
- AI-powered document analysis and Q&A
- Memory-efficient processing with text chunking
- Robust error handling and retry mechanisms
- Cache management for better performance

## Deployment on Streamlit Cloud

1. Fork or copy this repository to your GitHub account.

2. Visit [Streamlit Cloud](https://streamlit.io/cloud) and sign in.

3. Click "New app" and select this repository.

4. Configure the following secrets in your Streamlit Cloud dashboard:
   ```toml
   OPENAI_API_KEY = "your-openai-api-key"
   MODEL = "gpt-4"  # or your preferred OpenAI model
   ```

5. Deploy the application by clicking "Deploy!"

## System Requirements

The application requires the following system packages:
```
libgl1-mesa-glx
poppler-utils
```

These are already configured in `packages.txt` for Streamlit Cloud deployment.

## Python Dependencies

All required Python packages are listed in `requirements.txt`:
```
streamlit>=1.27.0
openai>=1.1.1
PyPDF2>=3.0.1
pandas>=2.0.3
numpy>=1.24.3
python-pptx>=0.6.21
scikit-learn>=1.3.0
```

## Local Development

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.streamlit/secrets.toml` file with your configuration:
   ```toml
   OPENAI_API_KEY = "your-openai-api-key"
   MODEL = "gpt-4"
   ```

5. Run the application:
   ```bash
   streamlit run app.py
   ```

## File Structure

- `app.py`: Main application code
- `requirements.txt`: Python package dependencies
- `packages.txt`: System package dependencies
- `.gitignore`: Git ignore configuration
- `.streamlit/secrets.toml`: Configuration secrets (not in repository)

## Limitations

- Maximum file size: 50MB
- Supported file types: .txt, .pdf, .csv, .pptx
- Rate limiting based on OpenAI API constraints

## Error Handling

The application implements comprehensive error handling:
- File processing errors
- API rate limiting with retries
- Memory management
- Invalid configuration detection
- Logging to both console and file

## Logging

Logs are written to:
- Console (stdout)
- `app.log` file

## Cache Management

The application implements efficient cache management:
- Embeddings are cached for 1 hour
- Cache is cleared when processing new files
- Memory-efficient text chunking

## Security

- API keys are managed through Streamlit secrets
- File size limits are enforced
- Input validation is implemented
- Secure error messages (no sensitive information exposed)
