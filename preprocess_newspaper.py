"""
Script to preprocess Loksatta newspaper PDF data in Marathi.
This script extracts Marathi text from PDF, preprocesses it, and saves clean data for clustering.
"""

import os
import sys
import pandas as pd
import numpy as np
import PyPDF2
import pytesseract
from pdf2image import convert_from_path
import re
import tempfile
import time
import argparse
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

# Add the project directory to the path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

# Set up paths
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
TEMP_DIR = os.path.join(BASE_DIR, 'temp')

# Create directories if they don't exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, TEMP_DIR]:
    os.makedirs(directory, exist_ok=True)

# Download NLTK resources if needed
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Marathi stopwords (common words to filter out)
MARATHI_STOPWORDS = [
    'आहे', 'आणि', 'तर', 'ते', 'ती', 'या', 'व', 'असून', 'मध्ये', 'आज', 'होते', 
    'झाले', 'होती', 'केले', 'केली', 'करण्यात', 'हे', 'सर्व', 'कोणत्याही', 'काही',
    'असे', 'करून', 'केला', 'होता', 'आला', 'आले', 'आली', 'असा', 'अशी', 'असलेल्या',
    'आहेत', 'त्या', 'त्यांनी', 'त्यांना', 'त्यांच्या', 'त्याच्या', 'त्याची', 'त्याचे',
    'त्यांचे', 'त्यांची', 'त्याला', 'त्याने', 'त्यामुळे', 'त्यावर', 'त्यातून', 'त्यात',
    'येथे', 'सोबत', 'द्वारे', 'परंतु', 'मात्र', 'किंवा', 'अथवा', 'म्हणून', 'म्हणजे',
    'कारण', 'दरम्यान', 'नंतर', 'पूर्वी', 'आता', 'येथील', 'तसेच', 'असतो', 'असते',
    'असतात', 'म्हणाले', 'म्हणाला', 'म्हणाली', 'म्हटले', 'केल्याने', 'केल्यास',
    'करताना', 'करतात', 'करतो', 'करते', 'झाला', 'झाली', 'होत', 'होणार', 'झालेल्या',
    'झालेला', 'झालेली', 'झालेले', 'येत', 'येणार', 'गेले', 'गेला', 'गेली', 'घेतले',
    'घेतला', 'घेतली', 'दिले', 'दिला', 'दिली', 'आले', 'आला', 'आली', 'असलेला',
    'असलेली', 'असलेले', 'असतील', 'असेल', 'होईल', 'करील', 'करेल', 'येईल', 'जाईल'
]

def check_tesseract_installation():
    """Check if Tesseract OCR is installed and available."""
    try:
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False

def perform_ocr_on_pdf(pdf_path, lang='mar'):
    """
    Perform OCR on a PDF file to extract Marathi text.
    
    Args:
        pdf_path: Path to the PDF file
        lang: The language for OCR (default: 'mar' for Marathi)
        
    Returns:
        List of extracted text from each page
    """
    try:
        # Create a unique temp directory for this operation
        temp_dir = os.path.join(TEMP_DIR, f"ocr_{int(time.time())}")
        os.makedirs(temp_dir, exist_ok=True)
        
        print(f"Converting PDF to images...")
        # Convert PDF to images
        images = convert_from_path(pdf_path)
        
        # Process each image with OCR
        extracted_texts = []
        
        print(f"Performing OCR on {len(images)} pages with Marathi language settings...")
        for i, image in enumerate(images):
            print(f"Processing page {i+1}/{len(images)}...")
            
            # Save image temporarily
            image_path = os.path.join(temp_dir, f"page_{i}.png")
            image.save(image_path, "PNG")
            
            # Perform OCR with Tesseract
            try:
                # Try with Marathi language
                text = pytesseract.image_to_string(image_path, lang=lang)
                
                # If no text is found, try with English as fallback
                if not text.strip() and lang != 'eng':
                    text = pytesseract.image_to_string(image_path, lang='eng')
                
                if text.strip():
                    extracted_texts.append(text.strip())
            except Exception as e:
                print(f"OCR error on page {i+1}: {e}. Trying with default settings.")
                # Try with default settings
                try:
                    text = pytesseract.image_to_string(image_path)
                    if text.strip():
                        extracted_texts.append(text.strip())
                except:
                    pass
        
        print("OCR completed!")
        
        # Clean up temp files
        for file in os.listdir(temp_dir):
            os.remove(os.path.join(temp_dir, file))
        os.rmdir(temp_dir)
        
        return extracted_texts
    
    except Exception as e:
        print(f"Error during OCR: {e}")
        return []

def extract_text_from_pdf(pdf_path):
    """
    Extract Marathi text from a PDF file using PyPDF2 and OCR if needed.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        List of extracted text paragraphs
    """
    print(f"Extracting Marathi text from {pdf_path}...")
    
    # Check if Tesseract is installed
    tesseract_available = check_tesseract_installation()
    if not tesseract_available:
        print("Warning: Tesseract OCR is not installed. PDF processing will be limited.")
        print("Installation instructions: https://github.com/tesseract-ocr/tesseract")
        print("For Marathi language support, you also need to install the Marathi language data.")
    
    # Extract text using PyPDF2
    text_content = []
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            
            # Check if PDF has text content
            has_text = False
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text = page.extract_text()
                if text.strip():  # Only add non-empty text
                    has_text = True
                    text_content.append(text.strip())
            
            # If no text was found, it might be a scanned PDF
            if not has_text:
                print("The PDF appears to be scanned or doesn't contain extractable text.")
                
                # Try OCR if Tesseract is available
                if tesseract_available:
                    print("Attempting to extract text using OCR with Marathi language settings...")
                    ocr_texts = perform_ocr_on_pdf(pdf_path, lang='mar')
                    if ocr_texts:
                        text_content = ocr_texts
                        print(f"Successfully extracted text from {len(ocr_texts)} pages using OCR!")
                    else:
                        print("OCR could not extract text from the PDF.")
                else:
                    print("Tesseract OCR is not available. Cannot extract text from scanned PDF.")
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
    
    # Process the extracted text
    if text_content:
        # Join all text content and then split by newlines to better handle paragraphs
        all_text = "\n".join(text_content)
        paragraphs = [p.strip() for p in all_text.split("\n") if p.strip()]
        
        # Filter out very short paragraphs (likely not actual content)
        paragraphs = [p for p in paragraphs if len(p) > 10]
        
        if paragraphs:
            return paragraphs
        else:
            print("No substantial text content found in the PDF file.")
            return []
    else:
        print("No text content found in the PDF file.")
        return []

def segment_into_articles(paragraphs, min_length=50):
    """
    Segment the extracted Marathi paragraphs into articles.
    This uses heuristics specific to Marathi news articles.
    
    Args:
        paragraphs: List of text paragraphs
        min_length: Minimum character length for an article
        
    Returns:
        List of articles
    """
    print("Segmenting text into Marathi news articles...")
    
    articles = []
    current_article = []
    
    for paragraph in paragraphs:
        # Skip very short paragraphs (likely headers, page numbers, etc.)
        if len(paragraph) < 15:
            continue
            
        # Check for Marathi headline patterns (often ends with colon)
        if paragraph.strip().endswith(':') and len(paragraph) < 100:
            # This might be a headline, start a new article
            if current_article:
                current_text = ' '.join(current_article)
                if len(current_text) >= min_length:
                    articles.append(current_text)
            current_article = [paragraph]
            continue
            
        # If paragraph starts with a capital letter and previous paragraph ended with period or Marathi danda (।),
        # it might be the start of a new article
        if (current_article and 
            (paragraph[0].isupper() or ord(paragraph[0]) > 128) and  # Check for uppercase or non-ASCII (likely Devanagari)
            (current_article[-1].strip().endswith(('.', '।', '?', '!')))):
            
            # Only consider it a new article if the current one is long enough
            current_text = ' '.join(current_article)
            if len(current_text) >= min_length:
                articles.append(current_text)
                current_article = [paragraph]
            else:
                current_article.append(paragraph)
        else:
            current_article.append(paragraph)
    
    # Add the last article if it exists
    if current_article:
        current_text = ' '.join(current_article)
        if len(current_text) >= min_length:
            articles.append(current_text)
    
    print(f"Segmented into {len(articles)} potential Marathi news articles")
    return articles

def preprocess_marathi_text(text):
    """
    Preprocess Marathi text for NLP tasks.
    
    Args:
        text: Marathi text string
        
    Returns:
        List of preprocessed tokens
    """
    # Convert to lowercase (though less important for Devanagari)
    text = text.lower()
    
    # Remove punctuation and special characters, but keep Devanagari characters
    text = re.sub(r'[^\u0900-\u097F\s]', ' ', text)  # Keep Devanagari Unicode range
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Tokenize (basic word tokenization)
    tokens = text.split()
    
    # Remove stopwords
    tokens = [token for token in tokens if token not in MARATHI_STOPWORDS]
    
    return tokens

def preprocess_articles(articles):
    """
    Preprocess the Marathi articles.
    
    Args:
        articles: List of article texts
        
    Returns:
        DataFrame with original and preprocessed text
    """
    print("Preprocessing Marathi articles...")
    
    # Create DataFrame
    df = pd.DataFrame({'text': articles})
    
    # Preprocess each article
    tokens_list = []
    for text in articles:
        tokens = preprocess_marathi_text(text)
        tokens_list.append(tokens)
    
    # Add preprocessed text to DataFrame
    df['preprocessed_tokens'] = tokens_list
    df['preprocessed_text'] = [' '.join(tokens) for tokens in tokens_list]
    
    return df

def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Preprocess Loksatta newspaper PDF in Marathi')
    
    parser.add_argument('--input_file', type=str, default='data/Loksatta-Mumbai 05-04.pdf',
                        help='Path to the Loksatta PDF file')
    parser.add_argument('--output_file', type=str, default='processed_loksatta.csv',
                        help='Name of the output CSV file (will be saved in data/processed)')
    parser.add_argument('--min_article_length', type=int, default=50,
                        help='Minimum character length for an article')
    
    args = parser.parse_args()
    
    # Resolve input file path
    if os.path.isabs(args.input_file):
        input_file = args.input_file
    else:
        input_file = os.path.join(BASE_DIR, args.input_file)
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} does not exist.")
        return
    
    # Extract text from PDF
    paragraphs = extract_text_from_pdf(input_file)
    
    if not paragraphs:
        print("No text extracted from the PDF. Exiting.")
        return
    
    # Segment into articles
    articles = segment_into_articles(paragraphs, args.min_article_length)
    
    if not articles:
        print("No articles segmented from the text. Exiting.")
        return
    
    # Preprocess articles
    df = preprocess_articles(articles)
    
    # Save to CSV
    output_path = os.path.join(PROCESSED_DATA_DIR, args.output_file)
    df.to_csv(output_path, index=False)
    
    print(f"Successfully processed {len(df)} Marathi articles.")
    print(f"Saved processed data to {output_path}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()