"""
Main execution script for Marathi News Article Clustering.
This script ties together all the modules to create a complete pipeline.
"""

import os
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm
import joblib
import time
import sys
import io
import locale

# Set up proper encoding for console output
try:
    # Try to set console to UTF-8 mode
    if sys.platform == 'win32':
        # For Windows, use UTF-8 encoding for stdout
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
except Exception as e:
    print(f"Warning: Could not set UTF-8 encoding for console: {e}")

# Import project modules
from preprocessing import load_and_preprocess_data, preprocess_text
from embeddings import MarathiEmbeddings
from clustering import cluster_articles
from llm_metadata import OllamaMetadataExtractor, LLMEnhancedEmbeddings, cluster_with_llm_metadata

# Set up paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

# Create directories if they don't exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, RESULTS_DIR]:
    os.makedirs(directory, exist_ok=True)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Marathi News Article Clustering')
    
    # Input data
    parser.add_argument('--input_file', type=str, default=None,
                        help='Path to input file (CSV or TXT)')
    parser.add_argument('--text_column', type=str, default='text',
                        help='Name of the column containing text (for CSV files)')
    
    # Clustering
    parser.add_argument('--n_clusters', type=int, default=None,
                        help='Number of clusters for K-means (default: auto-determined based on data size)')
    
    # LLM options
    parser.add_argument('--use_llm', action='store_true',
                        help='Use LLM for metadata extraction and enhanced clustering')
    parser.add_argument('--llm_model', type=str, default='llama3:8b',
                        help='Ollama model to use for metadata extraction')
    
    # Output
    parser.add_argument('--output_file', type=str, default=None,
                        help='Path to output file for clustered data')
    
    return parser.parse_args()

def safe_print(text):
    """Print text safely, handling encoding issues."""
    try:
        print(text)
    except UnicodeEncodeError:
        # If we can't print the text, try to print a simplified version
        try:
            print(text.encode('ascii', 'replace').decode('ascii'))
        except:
            print("[Text contains characters that cannot be displayed]")

def main():
    """Main execution function."""
    # Parse arguments
    args = parse_arguments()
    
    print("=" * 80)
    print("Marathi News Article Clustering")
    print("=" * 80)
    print(f"Current directory: {os.getcwd()}")
    print(f"Python version: {sys.version}")
    print(f"Console encoding: {sys.stdout.encoding}")
    print("-" * 80)
    
    # Step 1: Load and preprocess data
    if not args.input_file:
        # Use sample data if no input file is provided
        print("No input file provided. Using sample data...")
        sample_texts = [
            "मुंबई: महाराष्ट्रात कोरोनाचा प्रादुर्भाव वाढत असताना राज्य सरकारने नवीन निर्बंध जाहीर केले आहेत.",
            "पुणे: पुण्यात मोठ्या प्रमाणात पाऊस झाल्याने अनेक भागात पूरस्थिती निर्माण झाली आहे.",
            "नागपूर: नागपूर विद्यापीठाने परीक्षांचे वेळापत्रक जाहीर केले आहे.",
            "मुंबई: कोरोना विषाणूच्या नव्या प्रकाराचा पहिला रुग्ण मुंबईत आढळला आहे.",
            "पुणे: पुण्यातील रस्त्यांची दुरुस्ती लवकरच होणार आहे."
        ]
        df = pd.DataFrame({args.text_column: sample_texts})
        
        # Preprocess the sample data
        print("Preprocessing sample data...")
        tokens_list = []
        for text in tqdm(df[args.text_column], desc="Preprocessing"):
            tokens = preprocess_text(text)
            tokens_list.append(tokens)
        
        df['preprocessed_tokens'] = tokens_list
        df['preprocessed_text'] = [' '.join(tokens) for tokens in tokens_list]
    else:
        print(f"Loading and preprocessing data from {args.input_file}...")
        df = load_and_preprocess_data(args.input_file, args.text_column)
    
    print(f"Data shape: {df.shape}")
    print("Sample data:")
    for i, row in df.head().iterrows():
        try:
            # Try to safely print the text
            safe_print(f"[{i}] {row[args.text_column][:80]}...")
        except Exception as e:
            print(f"[{i}] [Error displaying text: {e}]")
    print("-" * 80)
    
    # Step 2: Generate TF-IDF embeddings
    print("Training TF-IDF embeddings model...")
    embedder = MarathiEmbeddings(method='tfidf')
    embedder.train(df['preprocessed_text'].tolist())
    
    # Step 3: LLM metadata extraction (if enabled)
    if args.use_llm:
        print("\nExtracting metadata using LLM...")
        try:
            # Initialize the metadata extractor
            metadata_extractor = OllamaMetadataExtractor(model_name=args.llm_model)
            
            # Extract metadata
            df = metadata_extractor.process_dataframe(df, args.text_column)
            
            print("Metadata extraction completed.")
            print("Sample metadata:")
            for i, row in df.head().iterrows():
                try:
                    safe_print(f"[{i}] Category: {row.get('llm_category', 'N/A')}, "
                              f"Emotions: {row.get('llm_emotions', 'N/A')}, "
                              f"Keywords: {str(row.get('llm_keywords', []))[:50]}")
                except Exception as e:
                    print(f"[{i}] [Error displaying metadata: {e}]")
            
            # Create enhanced embeddings
            print("\nGenerating LLM-enhanced embeddings...")
            enhanced_embedder = LLMEnhancedEmbeddings(embedder, metadata_extractor)
            embeddings = enhanced_embedder.get_embeddings_for_corpus(df['preprocessed_text'].tolist(), df)
        except Exception as e:
            print(f"Error in LLM processing: {e}")
            print("Falling back to standard embeddings...")
            # Generate regular embeddings if LLM processing fails
            embeddings = embedder.get_embeddings_for_corpus(df['preprocessed_text'].tolist())
    else:
        # Generate regular embeddings
        print("Generating document embeddings...")
        embeddings = embedder.get_embeddings_for_corpus(df['preprocessed_text'].tolist())
    
    print(f"Embeddings shape: {embeddings.shape}")
    print("-" * 80)
    
    # Step 4: Determine appropriate number of clusters
    n_samples = df.shape[0]
    if args.n_clusters is None:
        # Auto-determine number of clusters based on data size
        if n_samples <= 10:
            n_clusters = min(2, n_samples - 1)  # For very small datasets
        elif n_samples <= 100:
            n_clusters = min(5, n_samples // 2)
        else:
            n_clusters = min(10, n_samples // 10)
    else:
        # Ensure the requested number of clusters is valid
        n_clusters = min(args.n_clusters, n_samples - 1)
        if n_clusters < 2:
            n_clusters = 2
    
    # Step 5: Cluster articles
    print(f"Clustering articles using K-means with {n_clusters} clusters...")
    if args.use_llm:
        try:
            # Use LLM-enhanced clustering
            df_clustered, clusterer = cluster_with_llm_metadata(df, embeddings, n_clusters)
        except Exception as e:
            print(f"Error in LLM-enhanced clustering: {e}")
            print("Falling back to standard clustering...")
            # Use standard clustering if LLM-enhanced clustering fails
            df_clustered, clusterer = cluster_articles(df, embeddings, n_clusters)
    else:
        # Use standard clustering
        df_clustered, clusterer = cluster_articles(df, embeddings, n_clusters)
    
    # Save clustered data
    output_file = args.output_file or os.path.join(RESULTS_DIR, 'clustered_data.csv')
    df_clustered.to_csv(output_file, index=False)
    print(f"Saved clustered data to {output_file}")
    print("-" * 80)
    
    # Print cluster distribution
    cluster_counts = df_clustered['cluster'].value_counts().sort_index()
    print("\nCluster distribution:")
    for cluster_id, count in cluster_counts.items():
        print(f"Cluster {cluster_id}: {count} articles")
    print("-" * 80)
    
    # Print sample articles from each cluster
    print("\nSample articles from each cluster:")
    for cluster_id in sorted(df_clustered['cluster'].unique()):
        cluster_df = df_clustered[df_clustered['cluster'] == cluster_id]
        sample_article = cluster_df.iloc[0][args.text_column]
        print(f"\nCluster {cluster_id}:")
        try:
            safe_print(f"{sample_article[:150]}...")
            
            # Print metadata if available
            if args.use_llm and 'llm_category' in cluster_df.columns:
                sample_metadata = cluster_df.iloc[0]
                safe_print(f"Category: {sample_metadata.get('llm_category', 'N/A')}")
                safe_print(f"Keywords: {str(sample_metadata.get('llm_keywords', []))[:50]}")
        except Exception as e:
            print(f"[Error displaying text: {e}]")
    print("-" * 80)
    
    print("\nClustering pipeline completed successfully!")
    print(f"Results saved to {RESULTS_DIR}")
    print("=" * 80)

if __name__ == "__main__":
    start_time = time.time()
    try:
        main()
        end_time = time.time()
        print(f"Total execution time: {end_time - start_time:.2f} seconds")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
