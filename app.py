import os
import sys
import streamlit as st

# Set matplotlib backend before importing
import matplotlib
matplotlib.use('Agg')

try:
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from collections import Counter
    import joblib
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
except ImportError as e:
    st.error(f"Required package not found: {e}")
    st.stop()

# Try importing transformers (optional for basic functionality)
try:
    from transformers import AutoTokenizer, AutoModel
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    st.warning("Transformers not available. Using fallback methods.")
    TRANSFORMERS_AVAILABLE = False

# Add the project directory to the path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

# Try importing project modules with fallback
try:
    from src.preprocessing import preprocess_text
    from src.embeddings import MarathiEmbeddings
    from src.clustering import cluster_articles, MarathiNewsClustering
    from src.entity_extraction import extract_entities, cluster_by_entity, analyze_corpus
    CUSTOM_MODULES_AVAILABLE = True
except ImportError as e:
    st.warning(f"Custom modules not available: {e}. Using basic functionality.")
    CUSTOM_MODULES_AVAILABLE = False

# Set up paths
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

# Create directories if they don't exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, RESULTS_DIR]:
    try:
        os.makedirs(directory, exist_ok=True)
    except:
        pass

# Set page configuration
st.set_page_config(
    page_title="Marathi News Article Clustering",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Basic preprocessing function (fallback)
def basic_preprocess_text(text):
    """Basic text preprocessing when custom modules are not available"""
    if pd.isna(text):
        return []
    text = str(text).lower()
    # Basic tokenization
    import re
    tokens = re.findall(r'\b\w+\b', text)
    return tokens

# Basic clustering function (fallback)
def basic_cluster_articles(df, n_clusters=5):
    """Basic clustering using TF-IDF when custom modules are not available"""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans
    
    texts = df['preprocessed_text'].fillna('').tolist()
    
    # Create TF-IDF vectors
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(texts)
    
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(tfidf_matrix)
    
    df_clustered = df.copy()
    df_clustered['cluster'] = clusters
    
    return df_clustered, kmeans

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
            if CUSTOM_MODULES_AVAILABLE:
                batch_tokens = [preprocess_text(text) for text in batch[text_column]]
            else:
                batch_tokens = [basic_preprocess_text(text) for text in batch[text_column]]
            tokens_list.extend(batch_tokens)
            progress_bar.progress(min(1.0, (i + batch_size) / len(df)))
        
        df['preprocessed_tokens'] = tokens_list
        df['preprocessed_text'] = [' '.join(tokens) for tokens in tokens_list]
        progress_bar.progress(1.0)
    
    return df

@st.cache_resource
def generate_embeddings(texts):
    """Generate embeddings using available methods."""
    if TRANSFORMERS_AVAILABLE:
        with st.spinner("Generating embeddings using XLM-Roberta..."):
            try:
                tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
                model = AutoModel.from_pretrained("xlm-roberta-base")
                
                embeddings = []
                for text in texts[:100]:  # Limit for demo
                    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
                    with torch.no_grad():
                        outputs = model(**inputs)
                    embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
                    embeddings.append(embedding)
                
                return np.array(embeddings), None
            except Exception as e:
                st.warning(f"Error with transformers, using TF-IDF: {e}")
    
    # Fallback to TF-IDF
    with st.spinner("Generating TF-IDF embeddings..."):
        from sklearn.feature_extraction.text import TfidfVectorizer
        vectorizer = TfidfVectorizer(max_features=1000)
        embeddings = vectorizer.fit_transform(texts).toarray()
        return embeddings, vectorizer

def perform_clustering(df, embeddings, n_clusters, cluster_method):
    """Perform clustering based on the selected method."""
    if cluster_method == "K-means":
        if embeddings.shape[1] > 50:
            with st.spinner("Reducing dimensions with PCA..."):
                pca = PCA(n_components=50)
                embeddings_reduced = pca.fit_transform(embeddings)
        else:
            embeddings_reduced = embeddings
            
        if CUSTOM_MODULES_AVAILABLE:
            df_clustered, clusterer = cluster_articles(df, embeddings_reduced, n_clusters=n_clusters)
        else:
            df_clustered, clusterer = basic_cluster_articles(df, n_clusters=n_clusters)
    else:
        if CUSTOM_MODULES_AVAILABLE:
            if cluster_method == "Places":
                df_clustered = cluster_by_entity(df, 'places')
            elif cluster_method == "Emotions":
                df_clustered = cluster_by_entity(df, 'emotions')
            elif cluster_method == "Severity":
                df_clustered = cluster_by_entity(df, 'severity')
            elif cluster_method == "Category":
                df_clustered = cluster_by_entity(df, 'category')
        else:
            st.warning("Entity-based clustering not available. Using K-means instead.")
            df_clustered, _ = basic_cluster_articles(df, n_clusters=5)
    
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
        # Limit data for visualization
        sample_size = min(500, len(df_clustered))
        sample_indices = np.random.choice(len(df_clustered), sample_size, replace=False)
        
        embeddings_sample = embeddings[sample_indices]
        clusters_sample = df_clustered.iloc[sample_indices]['cluster']
        
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, sample_size-1))
        embeddings_2d = tsne.fit_transform(embeddings_sample)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        scatter = ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=clusters_sample, cmap='viridis')
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

def display_entity_analysis(df):
    """Display basic analysis when entity extraction is not available."""
    if CUSTOM_MODULES_AVAILABLE:
        with st.spinner("Analyzing entities..."):
            analysis = analyze_corpus(df['text'].tolist())
        # ... (rest of the original entity analysis code)
    else:
        st.info("Entity analysis requires custom modules. Showing basic text statistics instead.")
        
        # Basic text statistics
        texts = df['text'].fillna('').astype(str)
        word_counts = texts.str.split().str.len()
        char_counts = texts.str.len()
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Average Words per Article", f"{word_counts.mean():.1f}")
            st.metric("Total Articles", len(df))
        with col2:
            st.metric("Average Characters per Article", f"{char_counts.mean():.1f}")
            st.metric("Longest Article", f"{word_counts.max()} words")

# Main app
def main():
    st.title("Marathi News Article Clustering")
    
    if not CUSTOM_MODULES_AVAILABLE:
        st.warning("⚠️ Running in basic mode. Some features may be limited.")
    
    st.sidebar.header("Configuration")
    
    # Data source selection
    data_source = st.sidebar.selectbox("Select Data Source", ["Upload New Data", "Use Sample Data"])
    
    if data_source == "Use Sample Data":
        # Create sample data for demo
        sample_data = {
            'text': [
                'मुंबईत मोठा अपघात झाला आहे',
                'दिल्लीत नवीन धोरण जाहीर केले',
                'पुण्यात शिक्षण क्षेत्रात सुधारणा',
                'नागपूरमध्ये औद्योगिक विकास',
                'कोल्हापूरात कृषी नवीन तंत्रज्ञान'
            ]
        }
        df_sample = pd.DataFrame(sample_data)
        st.sidebar.success("Using sample data")
    else:
        uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])
    
    text_column = st.sidebar.text_input("Text Column Name", "text")
    
    available_methods = ["K-means"]
    if CUSTOM_MODULES_AVAILABLE:
        available_methods.extend(["Places", "Emotions", "Severity", "Category"])
    
    cluster_method = st.sidebar.selectbox("Clustering Method", available_methods)
    
    n_clusters = None
    if cluster_method == "K-means":
        n_clusters = st.sidebar.slider("Number of Clusters", 2, 10, 3)
    
    if st.sidebar.button("Load Data and Cluster"):
        if data_source == "Use Sample Data":
            df = df_sample.copy()
            # Basic preprocessing for sample data
            df['preprocessed_tokens'] = [basic_preprocess_text(text) for text in df[text_column]]
            df['preprocessed_text'] = [' '.join(tokens) for tokens in df['preprocessed_tokens']]
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
                
                st.header("Analysis")
                display_entity_analysis(df)

if __name__ == "__main__":
    main()