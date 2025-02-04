# DocuChat - Open Source Document Assistant 📄

DocuChat is an AI-powered document assistant that allows you to upload PDF documents and interact with them through natural language queries. Built with Streamlit and leveraging state-of-the-art AI models from Groq and Google, it combines text extraction with OCR capabilities to handle both digital and scanned documents.

## Features ✨
- **PDF Text Extraction**: Direct text extraction from digital PDFs
- **OCR Integration**: Extract text from scanned PDFs/images using Tesseract OCR
- **AI-Powered Q&A**: Conversational interface powered by Groq's Llama-3-70b model
- **Document Embeddings**: Google's Generative AI embeddings for semantic understanding
- **Conversational Memory**: Remembers context within each session

## Installation 🛠️

### Prerequisites
- Python 3.8+
- Tesseract OCR ([Windows install](https://github.com/UB-Mannheim/tesseract/wiki))
- Poppler ([Windows install](https://blog.alivate.com.au/poppler-windows/))

### Steps
1. Clone the repository
2. Install dependencies:
```bash
pip install streamlit pytesseract python-dotenv PyPDF2 langchain pypdfium2 matplotlib pillow langchain-google-genai langchain-groq

