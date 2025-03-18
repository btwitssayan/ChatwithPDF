import fitz  # PyMuPDF
import os
from PIL import Image

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
