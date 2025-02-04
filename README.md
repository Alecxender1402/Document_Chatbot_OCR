DocuChat - Open Source Document Assistant

DocuChat is an intelligent document assistant designed to process and analyze PDF documents efficiently. It leverages AI-powered OCR and conversational retrieval models to extract text, generate embeddings, and enable seamless interaction with document content.

Features

Extract text from PDFs using PyPDF2 and OCR via pytesseract

Convert PDFs into images for enhanced text extraction

Chunk large text data for efficient processing

Utilize FAISS vector store for semantic search

Employ ChatGroq for conversational AI interactions

Store chat history using ConversationBufferMemory

Interactive Streamlit interface for user-friendly experience

Installation

Clone this repository:

git clone https://github.com/Alecxender1402/Document_Chatbot_OCR.git

Install the required dependencies:

pip install -r requirements.txt

Set up environment variables:

Create a .env file in the project root and add:

GOOGLE_API_KEY=your_google_api_key
GROQ_API_KEY=your_groq_api_key

Install Tesseract OCR (if not already installed):

Download and install Tesseract OCR

Update the path in pytesseract.pytesseract.tesseract_cmd

Install Poppler (for PDF processing):

Download and install Poppler

Update the POPPLER_PATH variable in the script

Usage

Run the Streamlit app:

streamlit run test2.py

Upload your PDF documents via the sidebar.

Ask questions about the document in the text input field.

The assistant will extract, process, and respond based on document content.

Technologies Used

Python - Core language

Streamlit - Web application framework

PyPDF2 & pypdfium2 - PDF processing

pytesseract - Optical Character Recognition (OCR)

FAISS - Vector database for semantic search

Google Generative AI Embeddings - Text embeddings

ChatGroq - Conversational AI model

Contributing

Contributions are welcome! Feel free to submit pull requests or raise issues.

License

This project is licensed under the MIT License.

Developed with ❤️ by Abhi_Bhingradiya

