"""
Preprocessing module for Marathi text data.
This module handles text cleaning, tokenization, and other preprocessing steps
specifically optimized for Marathi language.
"""

import re
import string
import pandas as pd
import numpy as np
from tqdm import tqdm
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Download required NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Define Marathi stopwords (extend as needed)
MARATHI_STOPWORDS = set([
    'आणि', 'आहे', 'ते', 'तो', 'ती', 'या', 'व', 'असे', 'असा', 'असेल', 
    'असलेल्या', 'आहेत', 'केला', 'केली', 'केले', 'करण्यात', 'झाली', 'झाले', 
    'झाला', 'होते', 'होता', 'होती', 'म्हणून', 'म्हणाले', 'म्हणाला', 'म्हणाली',
    'हे', 'ही', 'हा', 'तर', 'सर्व', 'कोणत्याही', 'काही', 'येथे', 'सर्व', 'परंतु',
    'पण', 'मात्र', 'किंवा', 'आता', 'त्यामुळे', 'त्याच्या', 'त्याची', 'त्यांच्या',
    'त्यांची', 'त्याचे', 'त्यांचे', 'त्यांना', 'त्याला', 'त्याने', 'त्यांनी',
    'त्याचा', 'त्यांचा', 'त्याच्यावर', 'त्यावर', 'त्यांच्यावर', 'त्यांच्यात',
    'त्यात', 'त्याच्यात', 'त्याच्याकडे', 'त्याकडे', 'त्यांच्याकडे', 'त्यांच्याकडून',
    'त्याकडून', 'त्याच्याकडून', 'त्याच्यापासून', 'त्यापासून', 'त्यांच्यापासून',
    'त्यांच्यासाठी', 'त्यासाठी', 'त्याच्यासाठी', 'त्याच्याबरोबर', 'त्याबरोबर',
    'त्यांच्याबरोबर', 'त्यांच्यामुळे', 'त्यामुळे', 'त्याच्यामुळे', 'त्यांच्यापर्यंत',
    'त्यापर्यंत', 'त्याच्यापर्यंत', 'त्यांच्यापैकी', 'त्यापैकी', 'त्याच्यापैकी',
    'त्यांच्यातील', 'त्यातील', 'त्याच्यातील', 'त्यांच्यातून', 'त्यातून', 'त्याच्यातून'
])

def normalize_marathi_text(text):
    """
    Simple normalization for Marathi text.
    
    Args:
        text (str): Input Marathi text
        
    Returns:
        str: Normalized text
    """
    if not isinstance(text, str):
        return ""
    # Simple normalization - remove extra spaces
    return re.sub(r'\s+', ' ', text).strip()

def remove_punctuation(text):
    """
    Remove punctuation from text.
    
    Args:
        text (str): Input text
        
    Returns:
        str: Text without punctuation
    """
    if not isinstance(text, str):
        return ""
    translator = str.maketrans('', '', string.punctuation)
    return text.translate(translator)

def remove_special_characters(text):
    """
    Remove special characters, URLs, and other non-textual content.
    
    Args:
        text (str): Input text
        
    Returns:
        str: Cleaned text
    """
    if not isinstance(text, str):
        return ""
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove special characters but keep Marathi Unicode range
    text = re.sub(r'[^\u0900-\u097F\s]', ' ', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenize_marathi_text(text):
    """
    Tokenize Marathi text using simple space-based tokenization.
    
    Args:
        text (str): Input Marathi text
        
    Returns:
        list: List of tokens
    """
    if not isinstance(text, str):
        return []
    # Simple space-based tokenization for Marathi
    return text.split()

def remove_stopwords(tokens):
    """
    Remove Marathi stopwords from a list of tokens.
    
    Args:
        tokens (list): List of tokens
        
    Returns:
        list: List of tokens without stopwords
    """
    return [token for token in tokens if token.lower() not in MARATHI_STOPWORDS]

def preprocess_text(text):
    """
    Complete preprocessing pipeline for Marathi text.
    
    Args:
        text (str): Input Marathi text
        
    Returns:
        list: List of preprocessed tokens
    """
    if not isinstance(text, str) or not text:
        return []
    
    # Normalize text
    text = normalize_marathi_text(text)
    
    # Remove special characters
    text = remove_special_characters(text)
    
    # Remove punctuation
    text = remove_punctuation(text)
    
    # Tokenize
    tokens = tokenize_marathi_text(text)
    
    # Remove stopwords
    tokens = remove_stopwords(tokens)
    
    return tokens

def preprocess_dataframe(df, text_column):
    """
    Preprocess a dataframe containing Marathi text.
    
    Args:
        df (pandas.DataFrame): Input dataframe
        text_column (str): Name of the column containing text
        
    Returns:
        pandas.DataFrame: Dataframe with preprocessed text
    """
    # Create a copy to avoid modifying the original dataframe
    processed_df = df.copy()
    
    # Add a column with preprocessed tokens
    tqdm.pandas(desc="Preprocessing text")
    processed_df['preprocessed_tokens'] = processed_df[text_column].progress_apply(preprocess_text)
    
    # Add a column with preprocessed text (tokens joined)
    processed_df['preprocessed_text'] = processed_df['preprocessed_tokens'].apply(lambda x: ' '.join(x))
    
    return processed_df

def load_and_preprocess_data(file_path, text_column):
    """
    Load data from a file and preprocess it.
    
    Args:
        file_path (str): Path to the data file (CSV or TXT)
        text_column (str): Name of the column containing text
        
    Returns:
        pandas.DataFrame: Preprocessed dataframe
    """
    # Determine file type and load accordingly
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith('.txt'):
        # For TXT files, assume one article per line
        with open(file_path, 'r', encoding='utf-8') as f:
            articles = f.readlines()
        df = pd.DataFrame({text_column: articles})
    else:
        raise ValueError(f"Unsupported file format: {file_path}")
    
    # Preprocess the dataframe
    return preprocess_dataframe(df, text_column)

if __name__ == "__main__":
    # Example usage
    sample_text = "मुंबई: महाराष्ट्रात कोरोनाचा प्रादुर्भाव वाढत असताना राज्य सरकारने नवीन निर्बंध जाहीर केले आहेत."
    print("Original text:", sample_text)
    tokens = preprocess_text(sample_text)
    print("Preprocessed tokens:", tokens)
    print("Preprocessed text:", " ".join(tokens))
