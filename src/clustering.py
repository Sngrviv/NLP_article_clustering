"""
Clustering module for Marathi news articles.
This module implements K-means clustering to group similar articles.
"""

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import joblib
from collections import Counter

# Set up paths for saving models
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

class MarathiNewsClustering:
    """Class for clustering Marathi news articles."""
    
    def __init__(self, n_clusters=5, random_state=42):
        """
        Initialize the clustering model.
        
        Args:
            n_clusters (int): Number of clusters
            random_state (int): Random seed for reproducibility
        """
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.model = None
    
    def fit(self, embeddings):
        """
        Fit the clustering model on the embeddings.
        
        Args:
            embeddings (numpy.ndarray): Document embeddings matrix
            
        Returns:
            self: The fitted model instance
        """
        print(f"Fitting K-means clustering model with {self.n_clusters} clusters...")
        
        # Initialize the model
        self.model = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init=10)
        
        # Fit the model
        self.model.fit(embeddings)
        
        return self
    
    def predict(self, embeddings):
        """
        Predict clusters for new data.
        
        Args:
            embeddings (numpy.ndarray): Document embeddings matrix
            
        Returns:
            numpy.ndarray: Cluster labels
        """
        if self.model is None:
            raise ValueError("Model not trained")
        
        # Predict
        return self.model.predict(embeddings)
    
    def get_labels(self):
        """
        Get cluster labels for the training data.
        
        Returns:
            numpy.ndarray: Cluster labels
        """
        if self.model is None:
            raise ValueError("Model not trained")
        
        return self.model.labels_
    
    def evaluate(self, embeddings, labels=None):
        """
        Evaluate clustering quality using various metrics.
        
        Args:
            embeddings (numpy.ndarray): Document embeddings matrix
            labels (numpy.ndarray): Cluster labels (if None, use model's labels)
            
        Returns:
            dict: Dictionary of evaluation metrics
        """
        if labels is None:
            if self.model is None:
                raise ValueError("Model not trained")
            labels = self.get_labels()
        
        # Get number of unique clusters and samples
        n_clusters = len(set(labels))
        n_samples = len(labels)
        
        # Initialize metrics dictionary with cluster distribution
        metrics = {
            "silhouette_score": None,
            "davies_bouldin_score": None,
            "calinski_harabasz_score": None,
            "cluster_distribution": Counter(labels)
        }
        
        # Skip evaluation if all samples are in the same cluster or not enough clusters/samples
        if n_clusters <= 1:
            print("Cannot evaluate: All samples in one cluster")
            return metrics
        
        if n_clusters >= n_samples:
            print("Cannot evaluate: Number of clusters must be less than number of samples")
            return metrics
        
        # Calculate metrics safely
        try:
            metrics["silhouette_score"] = silhouette_score(embeddings, labels)
            print(f"Silhouette Score: {metrics['silhouette_score']:.4f}")
        except Exception as e:
            print(f"Could not calculate silhouette score: {e}")
        
        try:
            metrics["davies_bouldin_score"] = davies_bouldin_score(embeddings, labels)
            print(f"Davies-Bouldin Index: {metrics['davies_bouldin_score']:.4f}")
        except Exception as e:
            print(f"Could not calculate Davies-Bouldin index: {e}")
        
        try:
            metrics["calinski_harabasz_score"] = calinski_harabasz_score(embeddings, labels)
            print(f"Calinski-Harabasz Index: {metrics['calinski_harabasz_score']:.4f}")
        except Exception as e:
            print(f"Could not calculate Calinski-Harabasz index: {e}")
        
        print(f"Cluster Distribution: {metrics['cluster_distribution']}")
        
        return metrics
    
    def save_model(self, model_path=None):
        """
        Save the clustering model.
        
        Args:
            model_path (str): Path to save the model. If None, use default path.
        """
        if self.model is None:
            raise ValueError("Model not trained")
        
        if model_path is None:
            model_path = os.path.join(MODELS_DIR, "kmeans_clustering_model.joblib")
        
        print(f"Saving model to {model_path}")
        
        # Create a dictionary with all necessary components
        model_data = {
            "model": self.model,
            "n_clusters": self.n_clusters,
            "random_state": self.random_state
        }
        
        joblib.dump(model_data, model_path)
    
    def load_model(self, model_path=None):
        """
        Load a pre-trained clustering model.
        
        Args:
            model_path (str): Path to the model file. If None, use default path.
            
        Returns:
            self: The model instance
        """
        if model_path is None:
            model_path = os.path.join(MODELS_DIR, "kmeans_clustering_model.joblib")
        
        print(f"Loading model from {model_path}")
        
        # Load the model data
        model_data = joblib.load(model_path)
        
        # Set attributes
        self.model = model_data["model"]
        self.n_clusters = model_data["n_clusters"]
        self.random_state = model_data["random_state"]
        
        return self

def find_optimal_clusters(embeddings, max_clusters=20):
    """
    Find the optimal number of clusters using silhouette score.
    
    Args:
        embeddings (numpy.ndarray): Document embeddings matrix
        max_clusters (int): Maximum number of clusters to try
        
    Returns:
        tuple: (optimal_n_clusters, scores)
    """
    print("Finding optimal number of clusters...")
    
    # Ensure we don't try more clusters than samples
    n_samples = embeddings.shape[0]
    max_clusters = min(max_clusters, n_samples - 1)
    
    # If we have very few samples, just return 2 clusters
    if max_clusters < 2:
        print("Too few samples to find optimal clusters. Defaulting to 2 clusters.")
        return 2, []
    
    # Try different numbers of clusters
    silhouette_scores = []
    
    for n_clusters in tqdm(range(2, max_clusters + 1)):
        # Initialize and fit the model
        clusterer = MarathiNewsClustering(n_clusters=n_clusters)
        clusterer.fit(embeddings)
        
        # Get labels
        labels = clusterer.get_labels()
        
        # Calculate silhouette score
        try:
            score = silhouette_score(embeddings, labels)
            silhouette_scores.append((n_clusters, score))
            print(f"  {n_clusters} clusters: silhouette score = {score:.4f}")
        except Exception as e:
            print(f"  {n_clusters} clusters: failed to calculate silhouette score - {e}")
    
    # Find the optimal number of clusters
    if silhouette_scores:
        optimal_n_clusters = max(silhouette_scores, key=lambda x: x[1])[0]
        print(f"Optimal number of clusters: {optimal_n_clusters}")
        return optimal_n_clusters, silhouette_scores
    else:
        print("Could not determine optimal number of clusters. Defaulting to 2 clusters.")
        return 2, []  # Default to 2 clusters

def cluster_articles(df, embeddings, n_clusters=None):
    """
    Cluster articles and add cluster labels to the dataframe.
    
    Args:
        df (pandas.DataFrame): Dataframe with articles
        embeddings (numpy.ndarray): Document embeddings matrix
        n_clusters (int): Number of clusters. If None, find optimal number.
        
    Returns:
        tuple: (df_with_clusters, clusterer)
    """
    # Find optimal number of clusters if not specified
    if n_clusters is None:
        n_clusters, _ = find_optimal_clusters(embeddings)
    
    # Initialize and fit the clustering model
    clusterer = MarathiNewsClustering(n_clusters=n_clusters)
    clusterer.fit(embeddings)
    
    # Get cluster labels
    labels = clusterer.get_labels()
    
    # Add cluster labels to the dataframe
    df_with_clusters = df.copy()
    df_with_clusters['cluster'] = labels
    
    # Evaluate clustering
    clusterer.evaluate(embeddings, labels)
    
    # Save the model
    clusterer.save_model()
    
    return df_with_clusters, clusterer

if __name__ == "__main__":
    # Example usage
    from preprocessing import preprocess_text
    from embeddings import MarathiEmbeddings
    
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
    
    print("\nArticles with cluster labels:")
    for i, row in df_with_clusters.iterrows():
        print(f"Cluster {row['cluster']}: {row['text'][:50]}...")
