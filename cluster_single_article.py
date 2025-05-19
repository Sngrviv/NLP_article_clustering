"""
Script to demonstrate clustering a single new article using the LLM-enhanced model.
"""

import os
import sys
import pandas as pd
import numpy as np
import argparse
from src.preprocessing import preprocess_text
from src.embeddings import MarathiEmbeddings
from src.llm_metadata import OllamaMetadataExtractor, LLMEnhancedEmbeddings
from src.clustering import MarathiNewsClustering

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Cluster a single Marathi article')
    
    parser.add_argument('--article', type=str, required=False,
                        help='Article text to cluster')
    parser.add_argument('--article_file', type=str, required=False,
                        help='File containing the article text')
    parser.add_argument('--model_dir', type=str, default='models',
                        help='Directory containing trained models')
    parser.add_argument('--llm_model', type=str, default='llama3:8b',
                        help='Ollama model to use for metadata extraction')
    
    return parser.parse_args()

def main():
    """Main execution function."""
    args = parse_arguments()
    
    # Get article text
    if args.article:
        article_text = args.article
    elif args.article_file:
        try:
            with open(args.article_file, 'r', encoding='utf-8') as f:
                article_text = f.read()
        except Exception as e:
            print(f"Error reading article file: {e}")
            return
    else:
        print("Please provide either --article or --article_file")
        return
    
    print("=" * 80)
    print("Marathi Article Clustering")
    print("=" * 80)
    
    # Set up paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, args.model_dir)
    
    # Step 1: Preprocess the article
    print("Preprocessing article...")
    tokens = preprocess_text(article_text)
    preprocessed_text = ' '.join(tokens)
    
    # Step 2: Load the TF-IDF model
    print("Loading TF-IDF model...")
    try:
        embedder = MarathiEmbeddings(method='tfidf')
        embedder.load_model(os.path.join(models_dir, 'tfidf_model.joblib'))
    except Exception as e:
        print(f"Error loading TF-IDF model: {e}")
        print("Training a new model...")
        # Train a new model with just this article
        embedder = MarathiEmbeddings(method='tfidf')
        embedder.train([preprocessed_text])
    
    # Step 3: Extract metadata using LLM
    print("Extracting metadata using LLM...")
    metadata_extractor = OllamaMetadataExtractor(model_name=args.llm_model)
    metadata = metadata_extractor.extract_metadata(article_text)
    
    print("\nExtracted Metadata:")
    print(f"Title: {metadata.get('title', 'N/A')}")
    print(f"Category: {metadata.get('category', 'N/A')}")
    print(f"Emotions: {metadata.get('emotions', 'N/A')}")
    print(f"Severity: {metadata.get('severity', 'N/A')}")
    print(f"Entities: {', '.join(metadata.get('entities', []))}")
    print(f"Keywords: {', '.join(metadata.get('keywords', []))}")
    print(f"Summary: {metadata.get('summary', 'N/A')}")
    print("-" * 80)
    
    # Step 4: Generate embedding for the article
    print("Generating embedding...")
    embedding = embedder.get_embeddings_for_document(preprocessed_text)
    
    # Step 5: Load clustering model
    print("Loading clustering model...")
    try:
        clusterer = MarathiNewsClustering()
        clusterer.load_model(os.path.join(models_dir, 'kmeans_clustering_model.joblib'))
    except Exception as e:
        print(f"Error loading clustering model: {e}")
        print("No existing clustering model found. Cannot assign to a cluster.")
        print("You need to train a model first using the main.py script.")
        return
    
    # Step 6: Predict cluster
    print("Predicting cluster...")
    cluster = clusterer.predict(embedding.reshape(1, -1))[0]
    
    print(f"\nArticle assigned to Cluster {cluster}")
    print("-" * 80)
    
    # Step 7: Print article summary
    print("Article Summary:")
    print(f"Text: {article_text[:200]}...")
    print(f"Category: {metadata.get('category', 'N/A')}")
    print(f"Cluster: {cluster}")
    print("=" * 80)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
