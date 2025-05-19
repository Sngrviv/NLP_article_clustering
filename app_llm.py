"""
Streamlit application for Marathi News Article Clustering with LLM-enhanced capabilities.
This app allows users to upload articles, process them with Ollama, and visualize clustering results.
"""

import os
import sys
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import joblib
import io
import tempfile
import time
import PyPDF2
import pytesseract
from pdf2image import convert_from_path, convert_from_bytes

# Add the project directory to the path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

# Import project modules
from src.preprocessing import preprocess_text
from src.embeddings import MarathiEmbeddings
from src.clustering import cluster_articles, MarathiNewsClustering
from src.entity_extraction import extract_entities, analyze_corpus
from src.llm_metadata import OllamaMetadataExtractor, LLMEnhancedEmbeddings, cluster_with_llm_metadata

# Set up paths
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
TEMP_DIR = os.path.join(BASE_DIR, 'temp')

# Create directories if they don't exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, RESULTS_DIR, TEMP_DIR]:
    os.makedirs(directory, exist_ok=True)

# Set page configuration
st.set_page_config(
    page_title="Marathi News Article Clustering with LLM",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define functions for the app
def check_ollama_status():
    """Check if Ollama is installed and running."""
    import subprocess
    try:
        result = subprocess.run(
            ["ollama", "list"], 
            capture_output=True, 
            text=True, 
            check=False,
            encoding='utf-8',
            errors='ignore'
        )
        
        if result.returncode != 0:
            return False, "Ollama is not running or not installed. Please install from https://ollama.com/"
        
        available_models = result.stdout.strip().split('\n')
        if not available_models or len(available_models) <= 1:  # Header line only
            return False, "No Ollama models found. Please pull a model using 'ollama pull llama3:8b'"
        
        return True, available_models
    except Exception as e:
        return False, f"Error checking Ollama: {e}"

def parse_csv_or_text(uploaded_file):
    """Parse uploaded file as CSV, plain text, or PDF."""
    content_type = uploaded_file.type
    
    if 'csv' in content_type:
        # Parse as CSV
        try:
            df = pd.read_csv(uploaded_file)
            return df, "csv"
        except Exception as e:
            st.error(f"Error parsing CSV file: {e}")
            return None, None
    elif 'pdf' in content_type:
        # Parse as PDF
        try:
            # Check if Tesseract is installed
            tesseract_available = check_tesseract_installation()
            
            # Get PDF bytes
            pdf_bytes = uploaded_file.getvalue()
            
            # Create a temporary file to save the PDF
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
                temp_file.write(pdf_bytes)
                temp_path = temp_file.name
            
            # Extract text from PDF
            text_content = []
            with open(temp_path, 'rb') as file:
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
                    st.warning("The PDF appears to be scanned or doesn't contain extractable text.")
                    
                    # Try OCR if Tesseract is available
                    if tesseract_available:
                        st.info("Attempting to extract text using OCR (Optical Character Recognition)...")
                        ocr_texts = perform_ocr_on_pdf(pdf_bytes)
                        if ocr_texts:
                            text_content = ocr_texts
                            st.success(f"Successfully extracted text from {len(ocr_texts)} pages using OCR!")
                        else:
                            st.error("OCR could not extract text from the PDF.")
                    else:
                        st.error("Tesseract OCR is not installed. PDF processing will be limited.")
                        st.info("Installation instructions: https://github.com/tesseract-ocr/tesseract")
                        st.info("For Marathi language support, you also need to install the Marathi language data.")
                        
                        # Try alternative extraction methods if OCR is not available
                        st.info("Trying alternative extraction methods...")
                        
                        # Try to extract text with a more aggressive approach
                        for page_num in range(len(pdf_reader.pages)):
                            page = pdf_reader.pages[page_num]
                            
                            # Try to extract text from each object in the page
                            if '/Contents' in page:
                                try:
                                    content = page['/Contents'].get_data()
                                    # Try to decode with different encodings
                                    for encoding in ['utf-8', 'utf-16', 'latin-1']:
                                        try:
                                            decoded = content.decode(encoding, errors='ignore')
                                            if len(decoded.strip()) > 0:
                                                text_content.append(decoded.strip())
                                                break
                                        except:
                                            continue
                                except:
                                    pass
                        
                        # If still no content, try to extract raw bytes and decode
                        if not text_content:
                            for page_num in range(len(pdf_reader.pages)):
                                try:
                                    page = pdf_reader.pages[page_num]
                                    raw_content = page.extract_text(0).encode('utf-8', errors='ignore').decode('utf-8', errors='ignore')
                                    if raw_content.strip():
                                        text_content.append(raw_content.strip())
                                except:
                                    pass
            
            # Clean up the temporary file
            os.unlink(temp_path)
            
            # Create DataFrame
            if text_content:
                # Join all text content and then split by newlines to better handle paragraphs
                all_text = "\n".join(text_content)
                paragraphs = [p.strip() for p in all_text.split("\n") if p.strip()]
                
                # Filter out very short paragraphs (likely not actual content)
                paragraphs = [p for p in paragraphs if len(p) > 10]
                
                if paragraphs:
                    df = pd.DataFrame({'text': paragraphs})
                    return df, "pdf"
                else:
                    st.error("No substantial text content found in the PDF file.")
                    return None, None
            else:
                st.error("No text content found in the PDF file. The PDF might be scanned or contain images only.")
                if not tesseract_available:
                    st.info("For scanned PDFs, you'll need to install Tesseract OCR to extract text.")
                return None, None
        except Exception as e:
            st.error(f"Error parsing PDF file: {e}")
            return None, None
    else:
        # Parse as text
        try:
            content = uploaded_file.getvalue().decode('utf-8')
            # Split by lines and create a dataframe
            lines = [line.strip() for line in content.split('\n') if line.strip()]
            df = pd.DataFrame({'text': lines})
            return df, "text"
        except Exception as e:
            st.error(f"Error parsing text file: {e}")
            return None, None

def perform_ocr_on_pdf(pdf_bytes, lang='mar'):
    """
    Perform OCR on a PDF file to extract text, optimized for Marathi.
    
    Args:
        pdf_bytes: The PDF file as bytes
        lang: The language for OCR (default: 'mar' for Marathi)
        
    Returns:
        List of extracted text from each page
    """
    try:
        # Create a unique temp directory for this operation
        temp_dir = os.path.join(TEMP_DIR, f"ocr_{int(time.time())}")
        os.makedirs(temp_dir, exist_ok=True)
        
        # Convert PDF to images
        images = convert_from_bytes(pdf_bytes)
        
        # Process each image with OCR
        extracted_texts = []
        
        # Show progress
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, image in enumerate(images):
            status_text.text(f"Performing OCR on page {i+1}/{len(images)}...")
            
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
                st.warning(f"OCR error on page {i+1}: {e}. Trying with default settings.")
                # Try with default settings
                try:
                    text = pytesseract.image_to_string(image_path)
                    if text.strip():
                        extracted_texts.append(text.strip())
                except:
                    pass
            
            # Update progress
            progress_bar.progress(min(1.0, (i + 1) / len(images)))
        
        status_text.text("OCR completed!")
        
        # Clean up temp files
        for file in os.listdir(temp_dir):
            os.remove(os.path.join(temp_dir, file))
        os.rmdir(temp_dir)
        
        return extracted_texts
    
    except Exception as e:
        st.error(f"Error during OCR: {e}")
        return []

def check_tesseract_installation():
    """Check if Tesseract OCR is installed and available."""
    try:
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False

def preprocess_data(df, text_column='text'):
    """Preprocess the data."""
    # Add a progress bar for preprocessing
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Process in batches to update progress
    tokens_list = []
    batch_size = max(1, len(df) // 10)  # 10 updates
    
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i+batch_size]
        status_text.text(f"Preprocessing batch {i//batch_size + 1}/{(len(df) + batch_size - 1)//batch_size}...")
        batch_tokens = [preprocess_text(text) for text in batch[text_column]]
        tokens_list.extend(batch_tokens)
        progress_bar.progress(min(1.0, (i + batch_size) / len(df)))
    
    df['preprocessed_tokens'] = tokens_list
    df['preprocessed_text'] = [' '.join(tokens) if tokens else "dummy_token" for tokens in tokens_list]
    
    # Check if we have any meaningful text after preprocessing
    if all(text == "dummy_token" for text in df['preprocessed_text']):
        st.warning("No meaningful text was found after preprocessing. Adding dummy content to avoid empty vocabulary error.")
        # Add some dummy content to avoid empty vocabulary error
        df['preprocessed_text'] = df['preprocessed_text'].apply(lambda x: x + " dummy_content_for_processing")
    
    progress_bar.progress(1.0)
    status_text.text("Preprocessing completed!")
    
    return df

def extract_metadata_with_llm(df, text_column, model_name):
    """Extract metadata using LLM."""
    # Initialize the metadata extractor
    metadata_extractor = OllamaMetadataExtractor(model_name=model_name)
    
    # Add a progress bar for metadata extraction
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Process in batches
    batch_size = 1  # Process one article at a time due to LLM processing time
    result_df = df.copy()
    
    # Initialize metadata columns
    result_df['llm_title'] = ""
    result_df['llm_category'] = ""
    result_df['llm_entities'] = result_df.apply(lambda x: [], axis=1)
    result_df['llm_emotions'] = ""
    result_df['llm_severity'] = ""
    result_df['llm_summary'] = ""
    result_df['llm_keywords'] = result_df.apply(lambda x: [], axis=1)
    
    total_batches = (len(df) + batch_size - 1) // batch_size
    
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i+batch_size]
        status_text.text(f"Extracting metadata for article {i+1}/{len(df)}...")
        
        for j, (idx, row) in enumerate(batch.iterrows()):
            try:
                # Extract metadata
                metadata = metadata_extractor.extract_metadata(row[text_column])
                
                # Update the result dataframe
                result_df.at[idx, 'llm_title'] = metadata.get('title', '')
                result_df.at[idx, 'llm_category'] = metadata.get('category', 'Other')
                result_df.at[idx, 'llm_entities'] = metadata.get('entities', [])
                result_df.at[idx, 'llm_emotions'] = metadata.get('emotions', 'neutral')
                result_df.at[idx, 'llm_severity'] = metadata.get('severity', 'low')
                result_df.at[idx, 'llm_summary'] = metadata.get('summary', '')
                result_df.at[idx, 'llm_keywords'] = metadata.get('keywords', [])
                
                # Map categories to ensure proper assignment
                category_map = {
                    'politics': 'Politics',
                    'sports': 'Sports',
                    'entertainment': 'Entertainment',
                    'business': 'Business',
                    'technology': 'Technology',
                    'health': 'Health',
                    'education': 'Education',
                    'environment': 'Environment',
                    'crime': 'Crime',
                    'agriculture': 'Agriculture',
                    'religion': 'Religion',
                    'transportation': 'Transportation',
                    'weather': 'Weather',
                    'science': 'Science',
                    'social issues': 'Social Issues',
                    'other': 'Other'
                }
                
                # Handle both string and list formats for emotions
                if isinstance(result_df.at[idx, 'llm_emotions'], list):
                    result_df.at[idx, 'llm_emotions'] = ', '.join(result_df.at[idx, 'llm_emotions'])
                
                # Normalize category name and apply mapping
                category = str(result_df.at[idx, 'llm_category']).lower().strip()
                result_df.at[idx, 'llm_category'] = category_map.get(category, result_df.at[idx, 'llm_category'])
                
                # Add a small delay to avoid overwhelming the LLM
                if j < len(batch) - 1:
                    time.sleep(0.5)
                    
            except Exception as e:
                st.error(f"Error processing article at index {idx}: {e}")
        
        progress_bar.progress(min(1.0, (i + batch_size) / len(df)))
    
    progress_bar.progress(1.0)
    status_text.text("Metadata extraction completed!")
    
    return result_df

def generate_embeddings(texts):
    """Generate TF-IDF embeddings for texts."""
    # Check if texts are valid for embedding generation
    if not texts or all(not text.strip() for text in texts):
        st.error("No valid text content for generating embeddings.")
        # Create dummy embeddings to avoid errors
        dummy_embeddings = np.zeros((len(texts), 1))
        dummy_embedder = None
        return dummy_embeddings, dummy_embedder
    
    try:
        embedder = MarathiEmbeddings(method='tfidf')
        embedder.train(texts)
        embeddings = embedder.get_embeddings_for_corpus(texts)
        return embeddings, embedder
    except ValueError as e:
        st.error(f"Error generating embeddings: {e}")
        st.info("Creating dummy embeddings to continue processing.")
        # Create dummy embeddings to avoid breaking the app flow
        dummy_embeddings = np.zeros((len(texts), 1))
        dummy_embedder = None
        return dummy_embeddings, dummy_embedder

def display_metadata_table(df):
    """
    Display a table with LLM-extracted metadata.
    
    Args:
        df (pd.DataFrame): Dataframe with LLM-extracted metadata
    """
    if df is None or len(df) == 0:
        st.warning("No metadata to display.")
        return
    
    st.subheader("LLM-Extracted Metadata")
    
    # Create a copy of the dataframe for display
    display_df = df.copy()
    
    # Process each column to ensure it's compatible with Streamlit's dataframe display
    for col in display_df.columns:
        if col not in display_df:
            continue
            
        # Handle list-type columns (convert to string)
        display_df[col] = display_df[col].apply(
            lambda x: ', '.join(map(str, x)) if isinstance(x, list) else x
        )
        
        # Handle NaN values
        display_df[col] = display_df[col].fillna('')
        
        # Convert all columns to string to avoid mixed type issues
        display_df[col] = display_df[col].astype(str)
    
    # Display the dataframe
    st.dataframe(display_df)

def display_cluster_distribution(df_clustered):
    """Display the distribution of articles across clusters."""
    cluster_counts = df_clustered['cluster'].value_counts().sort_index()
    
    # Create a DataFrame for plotting
    plot_df = pd.DataFrame({
        'Cluster': cluster_counts.index.astype(str),
        'Count': cluster_counts.values
    })
    
    # Create a bar plot
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Cluster', y='Count', data=plot_df, ax=ax)
    ax.set_title('Cluster Distribution')
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Number of Articles')
    
    # Rotate x-axis labels if there are many clusters
    if len(cluster_counts) > 5:
        plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Display the counts as a table
    st.write("Cluster Distribution:")
    st.dataframe(plot_df)

def display_cluster_samples(df_clustered, text_column):
    """Display sample articles from each cluster with metadata if available."""
    clusters = sorted(df_clustered['cluster'].unique())
    
    for cluster in clusters:
        cluster_df = df_clustered[df_clustered['cluster'] == cluster]
        sample_article = cluster_df.iloc[0]
        
        with st.expander(f"Cluster {cluster} ({len(cluster_df)} articles)"):
            # Display sample article
            st.write("### Sample article:")
            st.write(sample_article[text_column])
            
            # Display metadata if available
            if 'llm_title' in sample_article:
                st.write("### Metadata:")
                st.write(f"**Title:** {sample_article['llm_title']}")
                st.write(f"**Category:** {sample_article['llm_category']}")
                st.write(f"**Emotion:** {sample_article['llm_emotions']}")
                st.write(f"**Severity:** {sample_article['llm_severity']}")
                st.write(f"**Keywords:** {', '.join(sample_article['llm_keywords']) if isinstance(sample_article['llm_keywords'], list) else sample_article['llm_keywords']}")
                st.write(f"**Summary:** {sample_article['llm_summary']}")
            
            # Show all articles in this cluster
            if st.checkbox(f"Show all articles in Cluster {cluster}", key=f"show_all_{cluster}"):
                for i, row in cluster_df.iterrows():
                    st.write(f"Article {i}:")
                    st.write(row[text_column])
                    st.write("---")

def display_category_distribution(df):
    """Display the distribution of categories from LLM metadata."""
    if 'llm_category' not in df.columns:
        st.warning("No category metadata available.")
        return
    
    # Count categories
    category_counts = df['llm_category'].value_counts()
    
    # Check if all categories are "Other"
    if len(category_counts) == 1 and category_counts.index[0] == 'Other':
        st.warning("All articles were classified as 'Other'. This may indicate an issue with the metadata extraction.")
        st.info("Try adjusting the LLM model or prompt to improve category classification.")
    
    # Create a DataFrame for plotting
    plot_df = pd.DataFrame({
        'Category': category_counts.index,
        'Count': category_counts.values,
        'Percentage': (category_counts.values / category_counts.values.sum() * 100).round(1)
    })
    
    # Sort by count descending
    plot_df = plot_df.sort_values('Count', ascending=False).reset_index(drop=True)
    
    # Display the categories as a table first for clarity
    st.subheader("Category Distribution:")
    st.dataframe(plot_df)
    
    # Create a pie chart
    if len(category_counts) > 1:  # Only show pie chart if there's more than one category
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.pie(plot_df['Count'], labels=plot_df['Category'], 
               autopct='%1.1f%%', startangle=90, 
               wedgeprops={'edgecolor': 'white', 'linewidth': 1})
        ax.set_title('Category Distribution')
        plt.tight_layout()
        st.pyplot(fig)
    
    # Create a bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x='Category', y='Count', data=plot_df, ax=ax)
    ax.set_title('Category Distribution')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig)

def display_emotion_distribution(df):
    """Display the distribution of emotions from LLM metadata."""
    if 'llm_emotions' not in df.columns:
        st.warning("No emotion metadata available.")
        return
    
    # Count emotions
    emotion_counts = df['llm_emotions'].value_counts()
    
    # Create a DataFrame for plotting
    plot_df = pd.DataFrame({
        'Emotion': emotion_counts.index,
        'Count': emotion_counts.values
    })
    
    # Create a pie chart
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.pie(plot_df['Count'], labels=plot_df['Emotion'], autopct='%1.1f%%')
    ax.set_title('Emotion Distribution')
    plt.tight_layout()
    st.pyplot(fig)
    
    # Display the emotions as a table
    st.write("Emotion Distribution:")
    st.dataframe(plot_df)

# Main app
def main():
    st.title("Marathi News Article Clustering with LLM")
    
    # Check Ollama status
    ollama_status, ollama_message = check_ollama_status()
    
    if not ollama_status:
        st.error(ollama_message)
        st.info("You can still use the app without LLM features, but metadata extraction will not be available.")
    else:
        st.success("Ollama is running and ready!")
        # Extract model names from the message
        available_models = [line.split()[0] for line in ollama_message if line.strip() and not line.startswith("NAME")]
    
    # Check Tesseract OCR status
    tesseract_status = check_tesseract_installation()
    if tesseract_status:
        st.success("Tesseract OCR is installed and ready for PDF processing!")
    else:
        st.warning("Tesseract OCR is not installed. PDF processing will be limited.")
        st.info("To enable OCR for scanned PDFs with Marathi text, please install Tesseract OCR with Marathi language support.")
        st.info("Installation instructions are available in the README_LLM_APP.md file.")
    
    # Sidebar
    st.sidebar.header("Configuration")
    
    # File upload
    uploaded_file = st.sidebar.file_uploader("Upload Marathi News Articles", type=["csv", "txt", "pdf"])
    
    # Text column selection (for CSV files)
    text_column = st.sidebar.text_input("Text Column Name (for CSV files)", "text")
    
    # LLM options
    use_llm = st.sidebar.checkbox("Use LLM for Metadata Extraction", value=ollama_status)
    
    llm_model = "llama3:8b"  # Default model
    if use_llm and ollama_status:
        llm_model = st.sidebar.selectbox(
            "Select Ollama Model", 
            available_models if isinstance(ollama_message, list) else ["llama3:8b"]
        )
    
    # Clustering options
    n_clusters = st.sidebar.slider("Number of Clusters", 2, 20, 5)
    
    # Process button
    process_button = st.sidebar.button("Process and Cluster Articles")
    
    # Main content
    if uploaded_file is not None:
        st.write(f"Uploaded file: {uploaded_file.name}")
        
        if process_button:
            # Parse the uploaded file
            df, file_type = parse_csv_or_text(uploaded_file)
            
            if df is not None:
                st.success(f"Loaded {len(df)} articles from {uploaded_file.name}")
                
                # Display data statistics
                st.header("Data Statistics")
                st.write(f"Number of articles: {len(df)}")
                st.write(f"Sample articles:")
                st.dataframe(df[[text_column]].head())
                
                # Preprocess data
                with st.spinner("Preprocessing data..."):
                    df = preprocess_data(df, text_column)
                
                # Extract metadata with LLM if enabled
                if use_llm and ollama_status:
                    with st.spinner(f"Extracting metadata using {llm_model}..."):
                        df = extract_metadata_with_llm(df, text_column, llm_model)
                    
                    # Display metadata
                    st.header("LLM-Extracted Metadata")
                    display_metadata_table(df)
                
                # Generate embeddings
                with st.spinner("Generating embeddings..."):
                    embeddings, embedder = generate_embeddings(df['preprocessed_text'].tolist())
                
                st.success(f"Generated embeddings with shape {embeddings.shape}")
                
                # Perform clustering
                with st.spinner(f"Clustering articles into {n_clusters} clusters..."):
                    if use_llm and ollama_status:
                        # Use LLM-enhanced clustering
                        df_clustered, _ = cluster_with_llm_metadata(df, embeddings, n_clusters)
                    else:
                        # Use standard clustering
                        df_clustered, _ = cluster_articles(df, embeddings, n_clusters)
                
                st.success(f"Clustered {len(df_clustered)} articles into {n_clusters} clusters")
                
                # Display results
                st.header("Clustering Results")
                
                # Display cluster distribution
                st.subheader("Cluster Distribution")
                display_cluster_distribution(df_clustered)
                
                # Display sample articles from each cluster
                st.subheader("Sample Articles from Each Cluster")
                display_cluster_samples(df_clustered, text_column)
                
                # Display metadata distributions if available
                if use_llm and ollama_status:
                    st.header("Metadata Analysis")
                    
                    # Display category distribution
                    st.subheader("Category Distribution")
                    display_category_distribution(df_clustered)
                    
                    # Display emotion distribution
                    st.subheader("Emotion Distribution")
                    display_emotion_distribution(df_clustered)
                
                # Save results
                output_file = os.path.join(RESULTS_DIR, f"clustered_data_{uploaded_file.name}.csv")
                df_clustered.to_csv(output_file, index=False)
                st.success(f"Saved clustered data to {output_file}")
                
                # Download button for results
                csv_buffer = io.StringIO()
                df_clustered.to_csv(csv_buffer, index=False)
                csv_str = csv_buffer.getvalue()
                
                st.download_button(
                    label="Download Clustered Data",
                    data=csv_str,
                    file_name=f"clustered_data_{uploaded_file.name}",
                    mime="text/csv"
                )
    else:
        st.info("Please upload a file to get started.")
        
        # Show sample data format
        st.header("Sample Data Format")
        st.write("For CSV files, include a column with article text (default column name: 'text').")
        st.write("For text files, each line will be treated as a separate article.")
        st.write("For PDF files, the text will be extracted and used for clustering.")
        
        # Show example
        st.code("""
        text
        मुंबई: महाराष्ट्रात कोरोनाचा प्रादुर्भाव वाढत असताना राज्य सरकारने नवीन निर्बंध जाहीर केले आहेत.
        पुणे: पुण्यात मोठ्या प्रमाणात पाऊस झाल्याने अनेक भागात पूरस्थिती निर्माण झाली आहे.
        नागपूर: नागपूर विद्यापीठाने परीक्षांचे वेळापत्रक जाहीर केले आहे.
        """, language="csv")

if __name__ == "__main__":
    main()
