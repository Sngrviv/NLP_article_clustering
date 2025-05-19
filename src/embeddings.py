"""
Embeddings module for Marathi news articles.
This module implements `TF`-IDF vectorization for text representation.
"""

import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

# Set up paths for saving models
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

class MarathiEmbeddings:
    """Class for generating embeddings for Marathi text."""
    
    def __init__(self, method='tfidf', max_features=5000):
        """
        Initialize the embeddings model.
        
        Args:
            method (str): Embedding method. Only 'tfidf' is supported.
            max_features (int): Maximum number of features for TF-IDF
        """
        if method != 'tfidf':
            print(f"Warning: Only TF-IDF method is supported. Defaulting to TF-IDF.")
            method = 'tfidf'
            
        self.method = method
        self.max_features = max_features
        self.vectorizer = None
    
    def train(self, texts, save_model=True):
        """
        Train the embeddings model on the given texts.
        
        Args:
            texts (list): List of preprocessed texts
            save_model (bool): Whether to save the model after training
            
        Returns:
            self: The trained model instance
        """
        print(f"Training {self.method} embeddings model...")
        
        if self.method == 'tfidf':
            self.vectorizer = TfidfVectorizer(max_features=self.max_features)
            self.vectorizer.fit(texts)
            
            if save_model:
                self.save_model()
        
        return self
    
    def get_embeddings_for_document(self, text):
        """
        Generate embeddings for a single document.
        
        Args:
            text (str): Preprocessed text
            
        Returns:
            numpy.ndarray: Document embedding vector
        """
        if self.vectorizer is None:
            raise ValueError("Model not trained")
        
        # Generate embedding
        if self.method == 'tfidf':
            embedding = self.vectorizer.transform([text]).toarray()[0]
            return embedding
    
    def get_embeddings_for_corpus(self, texts):
        """
        Generate embeddings for a corpus of documents.
        
        Args:
            texts (list): List of preprocessed texts
            
        Returns:
            numpy.ndarray: Document embeddings matrix
        """
        if self.vectorizer is None:
            raise ValueError("Model not trained")
        
        # Generate embeddings
        if self.method == 'tfidf':
            embeddings = self.vectorizer.transform(texts).toarray()
            return embeddings
    
    def save_model(self, model_path=None):
        """
        Save the embeddings model.
        
        Args:
            model_path (str): Path to save the model. If None, use default path.
        """
        if self.vectorizer is None:
            raise ValueError("Model not trained")
        
        if model_path is None:
            model_path = os.path.join(MODELS_DIR, f"{self.method}_model.joblib")
        
        print(f"Saving {self.method} model to {model_path}")
        
        # Create a dictionary with all necessary components
        model_data = {
            "method": self.method,
            "max_features": self.max_features,
            "vectorizer": self.vectorizer
        }
        
        joblib.dump(model_data, model_path)
    
    def load_model(self, model_path=None):
        """
        Load a pre-trained embeddings model.
        
        Args:
            model_path (str): Path to the model file. If None, use default path.
            
        Returns:
            self: The model instance
        """
        if model_path is None:
            model_path = os.path.join(MODELS_DIR, f"{self.method}_model.joblib")
        
        print(f"Loading {self.method} model from {model_path}")
        
        # Load the model data
        model_data = joblib.load(model_path)
        
        # Set attributes
        self.method = model_data["method"]
        self.max_features = model_data["max_features"]
        self.vectorizer = model_data["vectorizer"]
        
        return self

if __name__ == "__main__":
    # Example usage
    from preprocessing import preprocess_text
    
    # Sample data
    sample_texts = [
        "मुंबई: महाराष्ट्रात कोरोनाचा प्रादुर्भाव वाढत असताना राज्य सरकारने नवीन निर्बंध जाहीर केले आहेत.",
        "पुणे: पुण्यात मोठ्या प्रमाणात पाऊस झाल्याने अनेक भागात पूरस्थिती निर्माण झाली आहे.",
        "नागपूर: नागपूर विद्यापीठाने परीक्षांचे वेळापत्रक जाहीर केले आहे.",
        "मुंबई: कोरोना विषाणूच्या नव्या प्रकाराचा पहिला रुग्ण मुंबईत आढळला आहे.",
        "पुणे: पुण्यातील रस्त्यांची दुरुस्ती लवकरच होणार आहे."
    ]
    
    # Preprocess texts
    preprocessed_texts = [' '.join(preprocess_text(text)) for text in sample_texts]
    
    # Train TF-IDF model
    embedder = MarathiEmbeddings(method='tfidf')
    embedder.train(preprocessed_texts)
    
    # Generate embeddings
    embeddings = embedder.get_embeddings_for_corpus(preprocessed_texts)
    
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Sample embedding: {embeddings[0][:10]}")  # Show first 10 dimensions of first document
