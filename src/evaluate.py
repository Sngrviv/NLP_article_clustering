"""
Evaluation module for Marathi news article clustering.
This module provides functions for evaluating clustering quality.
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

# Set up paths for saving evaluation results
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

def evaluate_clustering(embeddings, labels, true_labels=None):
    """
    Evaluate clustering quality using various metrics.
    
    Args:
        embeddings (numpy.ndarray): Document embeddings matrix
        labels (numpy.ndarray): Predicted cluster labels
        true_labels (numpy.ndarray): True cluster labels (if available)
        
    Returns:
        dict: Dictionary of evaluation metrics
    """
    # Skip evaluation if all samples are in the same cluster
    if len(set(labels)) <= 1 or -1 in labels:
        print("Cannot evaluate: All samples in one cluster or contains noise points")
        return {
            "silhouette_score": None,
            "davies_bouldin_score": None,
            "calinski_harabasz_score": None,
            "cluster_distribution": Counter(labels),
            "adjusted_rand_score": None if true_labels is None else None,
            "normalized_mutual_info_score": None if true_labels is None else None
        }
    
    # Calculate internal metrics (no ground truth needed)
    metrics = {
        "silhouette_score": silhouette_score(embeddings, labels),
        "davies_bouldin_score": davies_bouldin_score(embeddings, labels),
        "calinski_harabasz_score": calinski_harabasz_score(embeddings, labels),
        "cluster_distribution": Counter(labels)
    }
    
    # Calculate external metrics (if ground truth is available)
    if true_labels is not None:
        metrics["adjusted_rand_score"] = adjusted_rand_score(true_labels, labels)
        metrics["normalized_mutual_info_score"] = normalized_mutual_info_score(true_labels, labels)
    
    # Print metrics
    print(f"Silhouette Score: {metrics['silhouette_score']:.4f}")
    print(f"Davies-Bouldin Index: {metrics['davies_bouldin_score']:.4f}")
    print(f"Calinski-Harabasz Index: {metrics['calinski_harabasz_score']:.4f}")
    print(f"Cluster Distribution: {metrics['cluster_distribution']}")
    
    if true_labels is not None:
        print(f"Adjusted Rand Index: {metrics['adjusted_rand_score']:.4f}")
        print(f"Normalized Mutual Information: {metrics['normalized_mutual_info_score']:.4f}")
    
    return metrics

def plot_evaluation_metrics(metrics_list, method_names, save_path=None):
    """
    Plot comparison of evaluation metrics for different clustering methods.
    
    Args:
        metrics_list (list): List of metrics dictionaries
        method_names (list): List of method names
        save_path (str): Path to save the plot. If None, display only.
    """
    # Extract metrics for comparison
    metric_names = ['silhouette_score', 'davies_bouldin_score', 'calinski_harabasz_score']
    metric_display_names = ['Silhouette Score', 'Davies-Bouldin Index', 'Calinski-Harabasz Index']
    
    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot each metric
    for i, (metric_name, display_name) in enumerate(zip(metric_names, metric_display_names)):
        values = [metrics[metric_name] for metrics in metrics_list]
        
        # For Davies-Bouldin, lower is better
        if metric_name == 'davies_bouldin_score':
            best_idx = np.argmin(values)
        else:
            best_idx = np.argmax(values)
        
        # Create bar colors (highlight the best method)
        colors = ['lightgray'] * len(method_names)
        colors[best_idx] = 'lightblue'
        
        # Plot
        ax = axes[i]
        bars = ax.bar(method_names, values, color=colors)
        
        # Add value labels on top of bars
        for j, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{values[j]:.4f}', ha='center', va='bottom', fontsize=10)
        
        # Set labels and title
        ax.set_title(display_name)
        ax.set_ylabel('Score')
        
        # Highlight the best method
        ax.text(best_idx, values[best_idx] / 2, 'BEST', ha='center', 
               fontweight='bold', color='blue', fontsize=12)
    
    # Set overall title
    plt.suptitle('Clustering Evaluation Metrics Comparison', fontsize=16)
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Saved evaluation metrics plot to {save_path}")
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for suptitle
    plt.show()

def compute_cluster_coherence(df, text_col='preprocessed_text', cluster_col='cluster'):
    """
    Compute coherence of each cluster based on term overlap.
    
    Args:
        df (pandas.DataFrame): Dataframe with articles and cluster labels
        text_col (str): Name of the column containing text
        cluster_col (str): Name of the column containing cluster labels
        
    Returns:
        dict: Dictionary with cluster coherence scores
    """
    # Get unique cluster IDs
    cluster_ids = sorted(df[cluster_col].unique())
    
    # Initialize coherence scores
    coherence_scores = {}
    
    # Compute coherence for each cluster
    for cluster_id in cluster_ids:
        # Filter for the specific cluster
        cluster_texts = df[df[cluster_col] == cluster_id][text_col]
        
        if len(cluster_texts) <= 1:
            coherence_scores[cluster_id] = 1.0  # Perfect coherence for single-document clusters
            continue
        
        # Tokenize texts (assuming space-separated tokens)
        tokenized_texts = [text.split() for text in cluster_texts]
        
        # Compute pairwise Jaccard similarity
        similarities = []
        for i in range(len(tokenized_texts)):
            for j in range(i + 1, len(tokenized_texts)):
                set1 = set(tokenized_texts[i])
                set2 = set(tokenized_texts[j])
                
                if not set1 or not set2:
                    continue
                
                # Jaccard similarity
                similarity = len(set1.intersection(set2)) / len(set1.union(set2))
                similarities.append(similarity)
        
        # Average similarity as coherence
        coherence_scores[cluster_id] = np.mean(similarities) if similarities else 0.0
    
    # Overall coherence
    coherence_scores['overall'] = np.mean(list(coherence_scores.values()))
    
    return coherence_scores

def plot_cluster_coherence(coherence_scores, save_path=None):
    """
    Plot coherence scores for each cluster.
    
    Args:
        coherence_scores (dict): Dictionary with cluster coherence scores
        save_path (str): Path to save the plot. If None, display only.
    """
    # Extract cluster IDs and scores (excluding 'overall')
    cluster_ids = [cid for cid in coherence_scores.keys() if cid != 'overall']
    scores = [coherence_scores[cid] for cid in cluster_ids]
    
    # Create figure
    plt.figure(figsize=(12, 6))
    
    # Plot
    bars = plt.bar(cluster_ids, scores, color='lightblue')
    
    # Add overall coherence line
    plt.axhline(y=coherence_scores['overall'], color='r', linestyle='-', 
               label=f'Overall: {coherence_scores["overall"]:.4f}')
    
    # Add value labels on top of bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
               f'{scores[i]:.4f}', ha='center', va='bottom', fontsize=10)
    
    # Set labels and title
    plt.xlabel('Cluster')
    plt.ylabel('Coherence Score')
    plt.title('Cluster Coherence Scores')
    plt.legend()
    
    # Save if requested
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Saved cluster coherence plot to {save_path}")
    
    plt.tight_layout()
    plt.show()

def identify_outliers(df, embeddings, cluster_col='cluster', threshold=0.5):
    """
    Identify potential outliers in each cluster.
    
    Args:
        df (pandas.DataFrame): Dataframe with articles and cluster labels
        embeddings (numpy.ndarray): Document embeddings matrix
        cluster_col (str): Name of the column containing cluster labels
        threshold (float): Distance threshold for outlier detection
        
    Returns:
        pandas.DataFrame: Dataframe with outlier information
    """
    # Get cluster labels
    labels = df[cluster_col].values
    
    # Compute cluster centers
    cluster_centers = {}
    for cluster_id in set(labels):
        if cluster_id == -1:  # Skip noise points
            continue
        
        # Get indices of documents in this cluster
        indices = np.where(labels == cluster_id)[0]
        
        # Compute center
        cluster_centers[cluster_id] = np.mean(embeddings[indices], axis=0)
    
    # Compute distances to cluster centers
    distances = []
    outlier_flags = []
    
    for i, (idx, row) in enumerate(df.iterrows()):
        cluster_id = row[cluster_col]
        
        if cluster_id == -1:  # Noise points are already outliers
            distances.append(np.nan)
            outlier_flags.append(True)
            continue
        
        # Compute Euclidean distance to cluster center
        center = cluster_centers[cluster_id]
        distance = np.linalg.norm(embeddings[i] - center)
        distances.append(distance)
        
        # Flag as outlier if distance exceeds threshold
        outlier_flags.append(distance > threshold)
    
    # Add information to dataframe
    result_df = df.copy()
    result_df['distance_to_center'] = distances
    result_df['is_outlier'] = outlier_flags
    
    return result_df

def compare_clustering_methods(df, embeddings, methods, n_clusters=None):
    """
    Compare different clustering methods.
    
    Args:
        df (pandas.DataFrame): Dataframe with articles
        embeddings (numpy.ndarray): Document embeddings matrix
        methods (list): List of clustering methods to compare
        n_clusters (int): Number of clusters. If None, find optimal number.
        
    Returns:
        tuple: (results_df, metrics_list, method_names)
    """
    from clustering import cluster_articles
    
    # Initialize results
    results = []
    metrics_list = []
    method_names = []
    
    # Run each clustering method
    for method in methods:
        print(f"\nRunning {method} clustering...")
        
        # Cluster articles
        df_clustered, clusterer = cluster_articles(df, embeddings, method=method, n_clusters=n_clusters)
        
        # Evaluate clustering
        metrics = evaluate_clustering(embeddings, df_clustered['cluster'].values)
        
        # Compute cluster coherence
        coherence_scores = compute_cluster_coherence(
            df_clustered, 
            text_col='preprocessed_text' if 'preprocessed_text' in df_clustered.columns else 'text',
            cluster_col='cluster'
        )
        
        # Store results
        results.append({
            'method': method,
            'df_clustered': df_clustered,
            'clusterer': clusterer,
            'metrics': metrics,
            'coherence_scores': coherence_scores
        })
        
        metrics_list.append(metrics)
        method_names.append(method)
    
    # Plot comparison
    plot_evaluation_metrics(
        metrics_list, 
        method_names,
        save_path=os.path.join(RESULTS_DIR, 'clustering_methods_comparison.png')
    )
    
    return results, metrics_list, method_names

def evaluate_and_visualize(df, embeddings, text_col='text', cluster_col='cluster'):
    """
    Perform comprehensive evaluation and visualization of clustering results.
    
    Args:
        df (pandas.DataFrame): Dataframe with articles and cluster labels
        embeddings (numpy.ndarray): Document embeddings matrix
        text_col (str): Name of the column containing text
        cluster_col (str): Name of the column containing cluster labels
    """
    from visualize import visualize_clustering_results
    
    # Evaluate clustering
    metrics = evaluate_clustering(embeddings, df[cluster_col].values)
    
    # Compute cluster coherence
    coherence_scores = compute_cluster_coherence(
        df, 
        text_col='preprocessed_text' if 'preprocessed_text' in df.columns else text_col,
        cluster_col=cluster_col
    )
    
    # Plot cluster coherence
    plot_cluster_coherence(
        coherence_scores,
        save_path=os.path.join(RESULTS_DIR, 'cluster_coherence.png')
    )
    
    # Identify outliers
    outliers_df = identify_outliers(df, embeddings, cluster_col)
    
    # Save outliers information
    outliers_df.to_csv(os.path.join(RESULTS_DIR, 'outliers.csv'), index=False)
    
    # Visualize clustering results
    visualize_clustering_results(df, embeddings, text_col, cluster_col)
    
    # Print summary
    print("\nClustering Evaluation Summary:")
    print(f"Number of clusters: {len(set(df[cluster_col].values))}")
    print(f"Silhouette Score: {metrics['silhouette_score']:.4f}")
    print(f"Overall Coherence: {coherence_scores['overall']:.4f}")
    print(f"Number of outliers: {outliers_df['is_outlier'].sum()}")

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
    
    # Compare different clustering methods
    methods = ['kmeans', 'agglomerative', 'hdbscan']
    results, metrics_list, method_names = compare_clustering_methods(df, embeddings, methods, n_clusters=2)
    
    # Get the best method based on silhouette score
    best_idx = np.argmax([metrics['silhouette_score'] for metrics in metrics_list])
    best_method = method_names[best_idx]
    best_df = results[best_idx]['df_clustered']
    
    print(f"\nBest clustering method: {best_method}")
    
    # Evaluate and visualize the best clustering
    evaluate_and_visualize(best_df, embeddings)
