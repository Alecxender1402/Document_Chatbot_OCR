# Use an official Python base image
FROM python:3.10-slim

# Install system dependencies: tesseract, poppler, and others
RUN apt-get update && \
    apt-get install -y tesseract-ocr poppler-utils libgl1-mesa-glx && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set environment variables for Tesseract and Poppler (Linux paths)
ENV TESSERACT_CMD=/usr/bin/tesseract
ENV POPPLER_PATH=/usr/bin

# Set the working directory
WORKDIR /app

# Copy your app code
COPY . /app

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirement.txt

# Expose Streamlit port
EXPOSE 8501

# Run the Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
