# DocuChat - Document Assistant 📄

DocuChat is an AI-powered document assistant that allows you to upload PDF documents and interact with them through natural language queries. Built with Streamlit and leveraging state-of-the-art AI models from Groq and Google, it combines text extraction with OCR capabilities to handle both digital and scanned documents.

## Features ✨
- **PDF Text Extraction**: Direct text extraction from digital PDFs
- **OCR Integration**: Extract text from scanned PDFs/images using Tesseract OCR
- **AI-Powered Q&A**: Conversational interface powered by Groq's Llama-3-70b model
- **Document Embeddings**: Google's Generative AI embeddings for semantic understanding
- **Conversational Memory**: Remembers context within each session

## Installation 🛠️

### Prerequisites :-
- Python 3.8+
- Tesseract OCR ([Windows install](https://github.com/UB-Mannheim/tesseract/wiki))
- Poppler ([Windows install](https://github.com/oschwartz10612/poppler-windows/releases))

### Steps :-
1. Clone the repository
```bash
git clone https://github.com/Alecxender1402/Document_Chatbot_OCR.git
```
2. Install dependencies:
```bash
pip install -r requirement.txt 
```

### Set environment variables in .env file :-
1. GOOGLE_API_KEY=your_google_api_key
2. GROQ_API_KEY=your_groq_api_key

### Configuration ⚙️

1. Update Tesseract and Poppler paths in code if needed (Windows default paths included)
2. Replace "your_api_key" in get_conversion_chain() with your actual Groq API key

### Usage 🚀
1. Start the app:-
```bash
streamlit run test2.py
```
2. Upload PDF documents through the sidebar
3. Click "Process" to analyze documents
4. Ask questions in the main chat interface

### Dependencies 📦

- **Core**: Streamlit, LangChain, PyPDF2
- **OCR**: Tesseract, Poppler, pypdfium2
- **AI Models**: Google Generative AI, Groq API
- **Embeddings**: FAISS vector store


