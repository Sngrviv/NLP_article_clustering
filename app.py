"""
Streamlit application for Marathi News Article Clustering.
This app allows users to cluster articles based on different criteria.
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

# Add the project directory to the path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

# Import project modules
from src.preprocessing import load_and_preprocess_data, preprocess_text
from src.embeddings import MarathiEmbeddings
from src.clustering import cluster_articles, MarathiNewsClustering
from src.entity_extraction import extract_entities, cluster_by_entity, analyze_corpus

# Set up paths
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

# Create directories if they don't exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, RESULTS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Set page configuration
st.set_page_config(
    page_title="Marathi News Article Clustering",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define functions for the app
@st.cache_data
def load_data(file_path, text_column='text'):
    """Load and preprocess data."""
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    else:
        st.error(f"Unsupported file format: {file_path}")
        return None
    
    # Check if the text column exists
    if text_column not in df.columns:
        st.error(f"Column '{text_column}' not found in the data.")
        return None
    
    # Preprocess the data
    with st.spinner("Preprocessing data..."):
        # Add a progress bar for preprocessing
        progress_bar = st.progress(0)
        
        # Process in batches to update progress
        tokens_list = []
        batch_size = max(1, len(df) // 10)  # 10 updates
        
        for i in range(0, len(df), batch_size):
            batch = df.iloc[i:i+batch_size]
            batch_tokens = [preprocess_text(text) for text in batch[text_column]]
            tokens_list.extend(batch_tokens)
            progress_bar.progress(min(1.0, (i + batch_size) / len(df)))
        
        df['preprocessed_tokens'] = tokens_list
        df['preprocessed_text'] = [' '.join(tokens) for tokens in tokens_list]
        
        progress_bar.progress(1.0)
    
    return df

@st.cache_resource
def generate_embeddings(texts):
    """Generate TF-IDF embeddings for texts."""
    with st.spinner("Generating embeddings..."):
        embedder = MarathiEmbeddings(method='tfidf')
        embedder.train(texts)
        embeddings = embedder.get_embeddings_for_corpus(texts)
    return embeddings, embedder

def perform_clustering(df, embeddings, n_clusters, cluster_method):
    """Perform clustering based on the selected method."""
    if cluster_method == "K-means":
        df_clustered, clusterer = cluster_articles(df, embeddings, n_clusters=n_clusters)
    elif cluster_method == "Places":
        df_clustered = cluster_by_entity(df, 'places')
    elif cluster_method == "Emotions":
        df_clustered = cluster_by_entity(df, 'emotions')
    elif cluster_method == "Severity":
        df_clustered = cluster_by_entity(df, 'severity')
    elif cluster_method == "Category":
        df_clustered = cluster_by_entity(df, 'category')
    else:
        st.error(f"Unsupported clustering method: {cluster_method}")
        return None
    
    return df_clustered

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
    """Display sample articles from each cluster."""
    clusters = sorted(df_clustered['cluster'].unique())
    
    for cluster in clusters:
        cluster_df = df_clustered[df_clustered['cluster'] == cluster]
        sample_article = cluster_df.iloc[0][text_column]
        
        with st.expander(f"Cluster {cluster} ({len(cluster_df)} articles)"):
            st.write("Sample article:")
            st.write(sample_article)
            
            # Show all articles in this cluster
            if st.checkbox(f"Show all articles in Cluster {cluster}", key=f"show_all_{cluster}"):
                for i, row in cluster_df.iterrows():
                    st.write(f"Article {i}:")
                    st.write(row[text_column])
                    st.write("---")

def display_entity_analysis(df):
    """Display entity analysis for the corpus."""
    with st.spinner("Analyzing entities..."):
        analysis = analyze_corpus(df['text'].tolist())
    
    # Display places analysis
    st.subheader("Places Analysis")
    places_df = pd.DataFrame({
        'Place': list(analysis['places'].keys()),
        'Count': list(analysis['places'].values())
    }).sort_values('Count', ascending=False)
    
    if not places_df.empty:
        # Create a bar plot for places
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='Count', y='Place', data=places_df.head(10), ax=ax)
        ax.set_title('Top 10 Places Mentioned')
        plt.tight_layout()
        st.pyplot(fig)
        
        # Display the places as a table
        st.write("Places Mentioned:")
        st.dataframe(places_df)
    else:
        st.write("No places detected in the corpus.")
    
    # Display emotions analysis
    st.subheader("Emotions Analysis")
    emotions_df = pd.DataFrame({
        'Emotion': list(analysis['emotions'].keys()),
        'Count': list(analysis['emotions'].values())
    })
    
    # Create a pie chart for emotions
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.pie(emotions_df['Count'], labels=emotions_df['Emotion'], autopct='%1.1f%%')
    ax.set_title('Emotion Distribution')
    plt.tight_layout()
    st.pyplot(fig)
    
    # Display the emotions as a table
    st.write("Emotion Distribution:")
    st.dataframe(emotions_df)
    
    # Display severity analysis
    st.subheader("Severity Analysis")
    severity_df = pd.DataFrame({
        'Severity': list(analysis['severity'].keys()),
        'Count': list(analysis['severity'].values())
    })
    
    # Create a pie chart for severity
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.pie(severity_df['Count'], labels=severity_df['Severity'], autopct='%1.1f%%')
    ax.set_title('Severity Distribution')
    plt.tight_layout()
    st.pyplot(fig)
    
    # Display the severity as a table
    st.write("Severity Distribution:")
    st.dataframe(severity_df)
    
    # Display category analysis
    st.subheader("Category Analysis")
    categories_df = pd.DataFrame({
        'Category': list(analysis['categories'].keys()),
        'Count': list(analysis['categories'].values())
    }).sort_values('Count', ascending=False)
    
    # Create a pie chart for categories
    fig, ax = plt.subplots(figsize=(10, 10))
    # Filter out categories with zero count for cleaner visualization
    non_zero_categories = categories_df[categories_df['Count'] > 0]
    ax.pie(non_zero_categories['Count'], labels=non_zero_categories['Category'], autopct='%1.1f%%')
    ax.set_title('Category Distribution')
    plt.tight_layout()
    st.pyplot(fig)
    
    # Create a bar chart for categories
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x='Category', y='Count', data=categories_df, ax=ax)
    ax.set_title('Category Distribution')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig)
    
    # Display the categories as a table
    st.write("Category Distribution:")
    st.dataframe(categories_df)

# Main app
def main():
    st.title("Marathi News Article Clustering")
    
    # Sidebar
    st.sidebar.header("Configuration")
    
    # File selection
    file_options = ["Sample Data"] + [f for f in os.listdir(RAW_DATA_DIR) if f.endswith('.csv')]
    selected_file = st.sidebar.selectbox("Select Data Source", file_options)
    
    if selected_file == "Sample Data":
        file_path = os.path.join(RAW_DATA_DIR, "sample_marathi_news.csv")
    else:
        file_path = os.path.join(RAW_DATA_DIR, selected_file)
    
    # Text column selection
    text_column = st.sidebar.text_input("Text Column Name", "text")
    
    # Clustering method
    cluster_method = st.sidebar.selectbox(
        "Clustering Method",
        ["K-means", "Places", "Emotions", "Severity", "Category"]
    )
    
    # Number of clusters (only for K-means)
    n_clusters = None
    if cluster_method == "K-means":
        n_clusters = st.sidebar.slider("Number of Clusters", 2, 20, 5)
    
    # Load data button
    if st.sidebar.button("Load Data and Cluster"):
        # Load and preprocess data
        df = load_data(file_path, text_column)
        
        if df is not None:
            st.success(f"Loaded {len(df)} articles from {os.path.basename(file_path)}")
            
            # Display data statistics
            st.header("Data Statistics")
            st.write(f"Number of articles: {len(df)}")
            st.write(f"Sample articles:")
            st.dataframe(df[[text_column]].head())
            
            # Generate embeddings (only needed for K-means)
            embeddings = None
            if cluster_method == "K-means":
                embeddings, _ = generate_embeddings(df['preprocessed_text'].tolist())
                st.success(f"Generated embeddings with shape {embeddings.shape}")
            
            # Perform clustering
            with st.spinner(f"Clustering articles using {cluster_method}..."):
                df_clustered = perform_clustering(df, embeddings, n_clusters, cluster_method)
            
            if df_clustered is not None:
                st.success(f"Clustered {len(df_clustered)} articles using {cluster_method}")
                
                # Display results
                st.header("Clustering Results")
                
                # Display cluster distribution
                st.subheader("Cluster Distribution")
                display_cluster_distribution(df_clustered)
                
                # Display sample articles from each cluster
                st.subheader("Sample Articles from Each Cluster")
                display_cluster_samples(df_clustered, text_column)
                
                # Display entity analysis
                st.header("Entity Analysis")
                display_entity_analysis(df)
                
                # Save results
                output_file = os.path.join(RESULTS_DIR, f"clustered_data_{cluster_method}.csv")
                df_clustered.to_csv(output_file, index=False)
                st.success(f"Saved clustered data to {output_file}")

if __name__ == "__main__":
    main()
