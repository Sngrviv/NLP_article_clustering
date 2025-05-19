"""
Visualization module for Marathi news article clustering.
This module provides functions for visualizing clustering results.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from collections import Counter
from tqdm import tqdm

# Set up paths for saving visualizations
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# Set style for matplotlib
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")

def plot_cluster_distribution(df, cluster_col='cluster', title='Cluster Distribution', save_path=None):
    """
    Plot the distribution of articles across clusters.
    
    Args:
        df (pandas.DataFrame): Dataframe with cluster labels
        cluster_col (str): Name of the column containing cluster labels
        title (str): Plot title
        save_path (str): Path to save the plot. If None, display only.
    """
    plt.figure(figsize=(12, 6))
    
    # Count articles in each cluster
    cluster_counts = df[cluster_col].value_counts().sort_index()
    
    # Plot
    ax = sns.barplot(x=cluster_counts.index, y=cluster_counts.values)
    
    # Add count labels on top of bars
    for i, count in enumerate(cluster_counts.values):
        ax.text(i, count + 0.1, str(count), ha='center')
    
    # Set labels and title
    plt.xlabel('Cluster')
    plt.ylabel('Number of Articles')
    plt.title(title)
    plt.xticks(rotation=0)
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Saved cluster distribution plot to {save_path}")
    
    plt.tight_layout()
    plt.show()

def plot_2d_clusters(embeddings, labels, title='Article Clusters', save_path=None):
    """
    Plot 2D visualization of clusters using t-SNE.
    
    Args:
        embeddings (numpy.ndarray): Document embeddings matrix
        labels (numpy.ndarray): Cluster labels
        title (str): Plot title
        save_path (str): Path to save the plot. If None, display only.
    """
    # Reduce to 2D using t-SNE
    if embeddings.shape[1] > 2:
        print("Reducing dimensions using t-SNE...")
        reducer = TSNE(n_components=2, random_state=42)
        embeddings_2d = reducer.fit_transform(embeddings)
    else:
        embeddings_2d = embeddings
    
    # Create a dataframe for plotting
    plot_df = pd.DataFrame({
        'x': embeddings_2d[:, 0],
        'y': embeddings_2d[:, 1],
        'cluster': labels
    })
    
    # Plot
    plt.figure(figsize=(12, 10))
    scatter = sns.scatterplot(data=plot_df, x='x', y='y', hue='cluster', palette='Set2', 
                             s=100, alpha=0.7)
    
    # Add cluster centers
    centers = plot_df.groupby('cluster')[['x', 'y']].mean().reset_index()
    for i, row in centers.iterrows():
        plt.text(row['x'], row['y'], f"Cluster {row['cluster']}", 
                fontsize=12, fontweight='bold', ha='center', va='center',
                bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.5'))
    
    # Set labels and title
    plt.title(title, fontsize=16)
    plt.xlabel('Dimension 1', fontsize=12)
    plt.ylabel('Dimension 2', fontsize=12)
    
    # Remove legend title
    scatter.legend_.set_title(None)
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Saved 2D cluster plot to {save_path}")
    
    plt.tight_layout()
    plt.show()

def plot_top_terms_per_cluster(df, n_terms=10, text_col='preprocessed_text', 
                              cluster_col='cluster', save_path=None):
    """
    Plot top terms for each cluster.
    
    Args:
        df (pandas.DataFrame): Dataframe with articles and cluster labels
        n_terms (int): Number of top terms to show
        text_col (str): Name of the column containing text
        cluster_col (str): Name of the column containing cluster labels
        save_path (str): Path to save the plot. If None, display only.
    """
    # Get unique clusters
    clusters = sorted(df[cluster_col].unique())
    n_clusters = len(clusters)
    
    # Set up the plot
    fig, axes = plt.subplots(nrows=n_clusters, ncols=1, figsize=(12, 4 * n_clusters))
    if n_clusters == 1:
        axes = [axes]
    
    # For each cluster
    for i, cluster_id in enumerate(clusters):
        # Get texts for this cluster
        cluster_texts = df[df[cluster_col] == cluster_id][text_col].tolist()
        
        # Count term frequencies
        all_words = []
        for text in cluster_texts:
            words = text.split()
            all_words.extend(words)
        
        # Get top terms
        word_counts = Counter(all_words)
        top_terms = word_counts.most_common(n_terms)
        
        # Create dataframe for plotting
        terms_df = pd.DataFrame(top_terms, columns=['term', 'count'])
        
        # Plot
        ax = axes[i]
        sns.barplot(data=terms_df, x='count', y='term', ax=ax, palette='Blues_d')
        
        # Set title and labels
        ax.set_title(f'Top {n_terms} Terms in Cluster {cluster_id} (n={len(cluster_texts)})', fontsize=14)
        ax.set_xlabel('Count', fontsize=12)
        ax.set_ylabel('Term', fontsize=12)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Saved top terms plot to {save_path}")
    
    plt.show()

def visualize_clustering_results(df, embeddings, text_col='text', cluster_col='cluster'):
    """
    Generate a comprehensive set of visualizations for clustering results.
    
    Args:
        df (pandas.DataFrame): Dataframe with articles and cluster labels
        embeddings (numpy.ndarray): Document embeddings matrix
        text_col (str): Name of the column containing text
        cluster_col (str): Name of the column containing cluster labels
    """
    print("Generating visualizations for clustering results...")
    
    # Create results directory if it doesn't exist
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # 1. Plot cluster distribution
    print("\n1. Plotting cluster distribution...")
    plot_cluster_distribution(
        df, 
        cluster_col=cluster_col,
        title='Distribution of Articles Across Clusters',
        save_path=os.path.join(RESULTS_DIR, 'cluster_distribution.png')
    )
    
    # 2. Plot 2D visualization of clusters
    print("\n2. Plotting 2D visualization of clusters...")
    plot_2d_clusters(
        embeddings, 
        df[cluster_col],
        title='2D Visualization of Article Clusters',
        save_path=os.path.join(RESULTS_DIR, 'clusters_2d.png')
    )
    
    # 3. Plot top terms per cluster
    print("\n3. Plotting top terms per cluster...")
    plot_top_terms_per_cluster(
        df,
        n_terms=15,
        text_col='preprocessed_text' if 'preprocessed_text' in df.columns else text_col,
        cluster_col=cluster_col,
        save_path=os.path.join(RESULTS_DIR, 'top_terms_per_cluster.png')
    )
    
    print("\nAll visualizations saved to:", RESULTS_DIR)

if __name__ == "__main__":
    # Example usage
    from preprocessing import preprocess_text
    from embeddings import MarathiEmbeddings
    from clustering import cluster_articles
    
    # Sample data
    sample_texts = [
        "मुंबई: महाराष्ट्रात कोरोनाचा प्रादुर्भाव वाढत असताना राज्य सरकारने नवीन निर्बंध जाहीर केले आहेत.",
        "पुणे: पुण्यात मोठ्या प्रमाणात पाऊस झाल्याने अनेक भागात पूरस्थिती निर्माण झाली आहे.",
        "नागपूर: नागपूर विद्यापीठाने परीक्षांचे वेळापत्रक जाहीर केले आहे.",
        "मुंबई: कोरोना विषाणूच्या नव्या प्रकाराचा पहिला रुग्ण मुंबईत आढळला आहे.",
        "पुणे: पुण्यातील रस्त्यांची दुरुस्ती लवकरच होणार आहे."
    ]
    
    # Create a sample dataframe
    df = pd.DataFrame({'text': sample_texts})
    
    # Preprocess texts
    tokens_list = [preprocess_text(text) for text in sample_texts]
    preprocessed_texts = [' '.join(tokens) for tokens in tokens_list]
    df['preprocessed_text'] = preprocessed_texts
    
    # Generate TF-IDF embeddings
    embedder = MarathiEmbeddings(method='tfidf')
    embedder.train(preprocessed_texts)
    embeddings = embedder.get_embeddings_for_corpus(preprocessed_texts)
    
    # Cluster articles
    df_with_clusters, clusterer = cluster_articles(df, embeddings, n_clusters=2)
    
    # Visualize clustering results
    visualize_clustering_results(df_with_clusters, embeddings, text_col='text', cluster_col='cluster')
