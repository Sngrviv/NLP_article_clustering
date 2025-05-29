import os
import sys
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import joblib
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from transformers import AutoTokenizer, AutoModel
import torch

# Add the project directory to the path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

# Import project modules (assuming these exist in your project)
from src.preprocessing import preprocess_text
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

# Set font for Marathi text display
st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Noto+Sans+Devanagari&display=swap" rel="stylesheet">
    <style>
    body {
        font-family: 'Noto Sans Devanagari', sans-serif;
    }
    </style>
    """, unsafe_allow_html=True)

# Define functions for the app
@st.cache_data
def load_data(data_source, text_column='text'):
    """Load and preprocess data from a file path or uploaded file."""
    if isinstance(data_source, str):  # File path
        if data_source.endswith('.csv'):
            df = pd.read_csv(data_source, encoding='utf-8')
        else:
            st.error(f"Unsupported file format: {data_source}")
            return None
    else:  # File-like object (uploaded file)
        try:
            df = pd.read_csv(data_source, encoding='utf-8')
        except Exception as e:
            st.error(f"Error reading the uploaded file: {e}")
            return None
    
    if text_column not in df.columns:
        st.error(f"Column '{text_column}' not found in the data.")
        return None
    
    # Preprocess the data
    with st.spinner("Preprocessing data..."):
        progress_bar = st.progress(0)
        tokens_list = []
        batch_size = max(1, len(df) // 10)
        
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
    """Generate embeddings using XLM-Roberta."""
    with st.spinner("Generating embeddings using XLM-Roberta (this may take a while on first run)..."):
        tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
        model = AutoModel.from_pretrained("xlm-roberta-base")
        
        embeddings = []
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                outputs = model(**inputs)
            embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
            embeddings.append(embedding)
        
        embeddings = np.array(embeddings)
    return embeddings, None

def perform_clustering(df, embeddings, n_clusters, cluster_method):
    """Perform clustering based on the selected method."""
    if cluster_method == "K-means":
        with st.spinner("Reducing dimensions with PCA..."):
            pca = PCA(n_components=50)
            embeddings_reduced = pca.fit_transform(embeddings)
        df_clustered, clusterer = cluster_articles(df, embeddings_reduced, n_clusters=n_clusters)
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
    
    plot_df = pd.DataFrame({
        'Cluster': cluster_counts.index.astype(str),
        'Count': cluster_counts.values
    })
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Cluster', y='Count', data=plot_df, ax=ax)
    ax.set_title('Cluster Distribution')
    ax.set_xlabel('Cluster')
    ax.set_ylabel('Number of Articles')
    if len(cluster_counts) > 5:
        plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig)
    
    st.write("Cluster Distribution:")
    st.dataframe(plot_df)

def display_cluster_plot(df_clustered, embeddings):
    """Display a 2D visualization of clusters using t-SNE."""
    with st.spinner("Generating 2D cluster visualization..."):
        tsne = TSNE(n_components=2, random_state=42)
        embeddings_2d = tsne.fit_transform(embeddings)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        scatter = ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=df_clustered['cluster'], cmap='viridis')
        legend = ax.legend(*scatter.legend_elements(), title="Clusters")
        ax.add_artist(legend)
        ax.set_title('2D Visualization of Clusters (t-SNE)')
        plt.tight_layout()
        st.pyplot(fig)

def display_cluster_samples(df_clustered, text_column):
    """Display sample articles from each cluster."""
    clusters = sorted(df_clustered['cluster'].unique())
    
    for cluster in clusters:
        cluster_df = df_clustered[df_clustered['cluster'] == cluster]
        sample_article = cluster_df.iloc[0][text_column]
        
        with st.expander(f"Cluster {cluster} ({len(cluster_df)} articles)"):
            st.write("Sample article:")
            st.write(sample_article)
            if st.checkbox(f"Show all articles in Cluster {cluster}", key=f"show_all_{cluster}"):
                for i, row in cluster_df.iterrows():
                    st.write(f"Article {i}:")
                    st.write(row[text_column])
                    st.write("---")

def display_entity_analysis(df):
    """Display entity analysis for the corpus."""
    with st.spinner("Analyzing entities..."):
        analysis = analyze_corpus(df['text'].tolist())
    
    st.subheader("Places Analysis")
    places_df = pd.DataFrame({
        'Place': list(analysis['places'].keys()),
        'Count': list(analysis['places'].values())
    }).sort_values('Count', ascending=False)
    
    if not places_df.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='Count', y='Place', data=places_df.head(10), ax=ax)
        ax.set_title('Top 10 Places Mentioned')
        plt.tight_layout()
        st.pyplot(fig)
        st.write("Places Mentioned:")
        st.dataframe(places_df)
    else:
        st.write("No places detected in the corpus.")
    
    st.subheader("Emotions Analysis")
    emotions_df = pd.DataFrame({
        'Emotion': list(analysis['emotions'].keys()),
        'Count': list(analysis['emotions'].values())
    })
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.pie(emotions_df['Count'], labels=emotions_df['Emotion'], autopct='%1.1f%%')
    ax.set_title('Emotion Distribution')
    plt.tight_layout()
    st.pyplot(fig)
    st.write("Emotion Distribution:")
    st.dataframe(emotions_df)
    
    st.subheader("Severity Analysis")
    severity_df = pd.DataFrame({
        'Severity': list(analysis['severity'].keys()),
        'Count': list(analysis['severity'].values())
    })
    
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.pie(severity_df['Count'], labels=severity_df['Severity'], autopct='%1.1f%%')
    ax.set_title('Severity Distribution')
    plt.tight_layout()
    st.pyplot(fig)
    st.write("Severity Distribution:")
    st.dataframe(severity_df)
    
    st.subheader("Category Analysis")
    categories_df = pd.DataFrame({
        'Category': list(analysis['categories'].keys()),
        'Count': list(analysis['categories'].values())
    }).sort_values('Count', ascending=False)
    
    non_zero_categories = categories_df[categories_df['Count'] > 0]
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.pie(non_zero_categories['Count'], labels=non_zero_categories['Category'], autopct='%1.1f%%')
    ax.set_title('Category Distribution')
    plt.tight_layout()
    st.pyplot(fig)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(x='Category', y='Count', data=categories_df, ax=ax)
    ax.set_title('Category Distribution')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    plt.tight_layout()
    st.pyplot(fig)
    st.write("Category Distribution:")
    st.dataframe(categories_df)

# Main app
def main():
    st.title("Marathi News Article Clustering")
    
    st.sidebar.header("Configuration")
    
    # Data source selection
    data_source = st.sidebar.selectbox("Select Data Source", ["Use Existing Data", "Upload New Data"])
    
    if data_source == "Use Existing Data":
        file_options = [f for f in os.listdir(RAW_DATA_DIR) if f.endswith('.csv')]
        if not file_options:
            st.sidebar.warning("No CSV files found in the raw data directory.")
            return
        selected_file = st.sidebar.selectbox("Select File", file_options)
        file_path = os.path.join(RAW_DATA_DIR, selected_file)
    else:
        uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])
    
    text_column = st.sidebar.text_input("Text Column Name", "text")
    cluster_method = st.sidebar.selectbox(
        "Clustering Method",
        ["K-means", "Places", "Emotions", "Severity", "Category"]
    )
    
    n_clusters = None
    if cluster_method == "K-means":
        n_clusters = st.sidebar.slider("Number of Clusters", 2, 20, 5)
    
    if st.sidebar.button("Load Data and Cluster"):
        if data_source == "Use Existing Data":
            df = load_data(file_path, text_column)
        elif data_source == "Upload New Data":
            if uploaded_file is not None:
                df = load_data(uploaded_file, text_column)
            else:
                st.warning("Please upload a CSV file.")
                return
        else:
            st.warning("Please select a data source.")
            return
        
        if df is not None:
            st.success(f"Loaded {len(df)} articles")
            st.header("Data Statistics")
            st.write(f"Number of articles: {len(df)}")
            st.write(f"Sample articles:")
            st.dataframe(df[[text_column]].head())
            
            embeddings = None
            if cluster_method == "K-means":
                embeddings, _ = generate_embeddings(df['preprocessed_text'].tolist())
                st.success(f"Generated embeddings with shape {embeddings.shape}")
            
            with st.spinner(f"Clustering articles using {cluster_method}..."):
                df_clustered = perform_clustering(df, embeddings, n_clusters, cluster_method)
            
            if df_clustered is not None:
                st.success(f"Clustered {len(df_clustered)} articles using {cluster_method}")
                st.header("Clustering Results")
                
                st.subheader("Cluster Distribution")
                display_cluster_distribution(df_clustered)
                
                if cluster_method == "K-means" and embeddings is not None:
                    st.subheader("Cluster Visualization")
                    display_cluster_plot(df_clustered, embeddings)
                
                st.subheader("Sample Articles from Each Cluster")
                display_cluster_samples(df_clustered, text_column)
                
                st.header("Entity Analysis")
                display_entity_analysis(df)
                
                output_file = os.path.join(RESULTS_DIR, f"clustered_data_{cluster_method}.csv")
                df_clustered.to_csv(output_file, index=False)
                st.success(f"Saved clustered data to {output_file}")

if __name__ == "__main__":
    main()  