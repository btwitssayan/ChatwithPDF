import fitz  # PyMuPDF
import os
from PIL import Image
from mistralai import Mistral
import streamlit as st

def convert_pdf_to_images(pdf_path, zoom=2.0):
    # Open the PDF
    pdf_document = fitz.open(pdf_path)
    
    # Create output folder if it doesn't exist
        
    images=[]
    
    # Iterate through pages
    for page_num in range(len(pdf_document)):
        # Get the page
        page = pdf_document.load_page(page_num)
        
        # Create matrix for better resolution
        mat = fitz.Matrix(zoom, zoom)
        
        # Get the pixmap (image)
        pix = page.get_pixmap(matrix=mat)
        
        images.append(Image.frombytes("RGB", [pix.width, pix.height], pix.samples))
    
    pdf_document.close()
    return images

def extract_text_from_pdf(pdf_path: str) -> str:
    # Retrieve the API key from environment variables
    api_key = st.secrets["MISTRAL_API_KEY"]
    
    # Initialize the Mistral client
    client = Mistral(api_key=api_key)
    
    # Upload the PDF file for OCR
    with open(pdf_path, "rb") as file_obj:
        uploaded_pdf = client.files.upload(
            file={
                "file_name": os.path.basename(pdf_path),
                "content": file_obj,
            },
            purpose="ocr"
        )
    
    # Get the signed URL for the uploaded PDF
    signed_url = client.files.get_signed_url(file_id=uploaded_pdf.id)
    
    # Define the chat message for text extraction
    model = "mistral-small-latest"
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Extract and present the complete text from the document while preserving "
                        "its original structure. Ensure that the output includes only meaningful "
                        "and coherent content."
                    )
                },
                {
                    "type": "document_url",
                    "document_url": signed_url.url
                }
            ]
        }
    ]
    
    # Request the text extraction via chat completion
    chat_response = client.chat.complete(
        model=model,
        messages=messages
    )
    
    # Return the extracted text from the response
    return chat_response.choices[0].message.content
