import streamlit as st
import os
import pytesseract
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import pypdfium2 as pdfium
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from PIL import Image
from io import BytesIO
from langchain_groq import ChatGroq
import base64

# Configure Tesseract path (Windows)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
POPPLER_PATH = r'C:\Program Files\poppler-24.08.0\Library\bin'

load_dotenv()
os.getenv("GOOGLE_API_KEY")

user_icon = "👤"
doc_icon = "📄"

def preprocess_image(img):
    img = img.convert('L')  # Grayscale
    img = img.point(lambda x: 0 if x < 180 else 255, '1')  # Simple binarization
    return img

def show_pdf(file_path):
    try:
        bytes_data = file_path.getvalue()
        base64_pdf = base64.b64encode(bytes_data).decode('utf-8')
        pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'
        st.markdown(pdf_display, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error displaying PDF: {str(e)}")

def get_pdf_text(pdf_io):
    text = ""
    pdf_reader = PdfReader(pdf_io)
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text

def convert_image_to_text(img):
    try:
        osd = pytesseract.image_to_osd(img, output_type=pytesseract.Output.DICT)
        rotate_angle = osd.get('rotate', 0)
        if rotate_angle:
            img = img.rotate(-rotate_angle, expand=True)
    except Exception:
        pass
    img = preprocess_image(img)
    text = pytesseract.image_to_string(img, lang='eng', config='--psm 6')
    text = text.strip()
    return text

def clean_text(text):
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines)

def get_chunks(text):
    text_splitter = CharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=256,
        separator="\n",
        length_function=len
    )
    return text_splitter.split_text(text)

def get_vectorstore(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversion_chain(vectorstore):
    chat = ChatGroq(
        temperature=0.7,
        groq_api_key="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",
        model_name="llama-3.3-70b-versatile"
    )
    memory = ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=True
    )
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=chat,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        verbose=False
    )
    return conversation_chain

def handle_user_input(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.markdown(f"{user_icon} **User:** {message.content}")
        else:
            st.markdown(f"{doc_icon} **DocuChat:** {message.content}")

def display_pdf(pdf_docs):
    if pdf_docs:
        pdf_tabs = st.tabs([f"PDF {i+1}: {pdf.name}" for i, pdf in enumerate(pdf_docs)])
        for i, tab in enumerate(pdf_tabs):
            with tab:
                pdf_display = pdf_docs[i]
                st.write(f"Filename: {pdf_display.name}")
                st.write(f"File size: {pdf_display.size/1024:.2f} KB")
                show_pdf(pdf_display)

def process_pdf_with_logs(pdf_file, log_area):
    pdf_bytes = pdf_file.read()
    pdf_io = BytesIO(pdf_bytes)
    pdf_io.seek(0)
    pdfium_doc = pdfium.PdfDocument(pdf_io)
    total_pages = len(pdfium_doc)
    all_text = ""
    for i in range(total_pages):
        pdf_io.seek(0)
        text = ""
        try:
            reader = PdfReader(pdf_io)
            page = reader.pages[i]
            page_text = page.extract_text()
            if page_text:
                text += page_text
        except Exception:
            pass
        if not text.strip():
            image = next(pdfium_doc.render(
                pdfium.PdfBitmap.to_pil,
                page_indices=[i],
                scale=300/72,
            ))
            if image.convert('L').getextrema()[0] > 240:
                log_area.write(f"Page {i+1}: Skipped (blank page)")
                continue
            text = convert_image_to_text(image)
        text = clean_text(text)
        if text.strip():
            log_area.write(f"Page {i+1}: Successfully converted and embedded")
        else:
            log_area.write(f"Page {i+1}: No text found")
        all_text += text + "\n"
    return all_text


def main():
    st.set_page_config(page_title="DocuChat", page_icon="📄", layout="wide")
    st.title("DocuChat - PDF Q&A with Logs")

    with st.sidebar:
        st.header("Upload & Process")
        pdf_file = st.file_uploader("Upload a PDF", type=['pdf'])
        process_btn = st.button("Process")

    log_area = st.empty()

    if process_btn and pdf_file:
        with st.spinner("Processing PDF..."):
            log_area.write("Starting PDF processing...\n")
            all_text = process_pdf_with_logs(pdf_file, log_area)
            text_chunks = get_chunks(all_text)
            vectorstore = get_vectorstore(text_chunks)
            st.session_state.conversation = get_conversion_chain(vectorstore)
            st.success("Processing complete! See logs above.")

    # Chat UI
    st.header("DocuChat - Assistant")
    user_question = st.text_input("Ask questions about your documents:")
    submit_button = st.button("Submit")
    if submit_button and user_question:
        if st.session_state.get("conversation") is not None:
            handle_user_input(user_question)
        else:
            st.warning("Please process the uploaded documents first.")

if __name__ == "__main__":
    main()
