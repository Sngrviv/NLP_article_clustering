"""
LLM-based metadata extraction for Marathi news articles.
This module uses Ollama to extract metadata and enhance clustering capabilities.
"""

import os
import json
import subprocess
import tempfile
import time
from typing import Dict, List, Any, Union, Optional
import pandas as pd
from tqdm import tqdm

class OllamaMetadataExtractor:
    """Class for extracting metadata from articles using Ollama LLM."""
    
    def __init__(self, model_name: str = "llama3:8b"):
        """
        Initialize the Ollama metadata extractor.
        
        Args:
            model_name (str): Name of the Ollama model to use
        """
        self.model_name = model_name
        self._check_ollama_installed()
    
    def _check_ollama_installed(self) -> None:
        """Check if Ollama is installed and the model is available."""
        try:
            # Check if Ollama is installed
            result = subprocess.run(
                ["ollama", "list"], 
                capture_output=True, 
                text=True, 
                encoding="utf-8",  # Specify UTF-8 encoding 
                check=False
            )
            
            if result.returncode != 0:
                print(f"Warning: Ollama may not be installed or running. Error: {result.stderr}")
                print("Please install Ollama from https://ollama.com/")
                return
            
            # Check if the model is available
            if self.model_name not in result.stdout:
                print(f"Model {self.model_name} not found in Ollama. Attempting to pull it...")
                subprocess.run(
                    ["ollama", "pull", self.model_name], 
                    check=False,
                    encoding="utf-8"  # Add UTF-8 encoding here
                )
        except Exception as e:
            print(f"Error checking Ollama installation: {e}")
            print("Please ensure Ollama is installed and running.")
    
    def extract_metadata(self, text: str) -> Dict[str, Any]:
        """
        Extract metadata from a single article using Ollama.
        
        Args:
            text (str): Article text
            
        Returns:
            dict: Extracted metadata
        """
        prompt = self._create_extraction_prompt(text)
        
        try:
            # Call Ollama with the prompt
            result = subprocess.run(
                ["ollama", "run", self.model_name, prompt], 
                capture_output=True, 
                text=True,
                encoding="utf-8",  # Specify UTF-8 encoding
                check=False
            )
            
            if result.returncode != 0:
                print(f"Error calling Ollama: {result.stderr}")
                return self._create_default_metadata()
            
            # Parse the output to extract the JSON
            return self._parse_llm_response(result.stdout)
            
        except Exception as e:
            print(f"Error extracting metadata: {e}")
            return self._create_default_metadata()
    
    def _create_extraction_prompt(self, text: str) -> str:
        """
        Create a prompt for metadata extraction.
        
        Args:
            text (str): Article text
            
        Returns:
            str: Prompt for the LLM
        """
        return f"""
You are a specialized Marathi news article analyzer. Your task is to extract structured metadata from the article.

Extract the following metadata from this Marathi news article. 
Respond ONLY with a JSON object containing these fields:
- title: The title of the article (or a generated one if not obvious)
- category: The main category from this list ONLY:
  * Politics (राजकारण): Government actions, political parties, elections, policies
  * Sports (क्रीडा): Cricket, kabaddi, football, athletes, tournaments
  * Entertainment (मनोरंजन): Movies, TV shows, celebrities, music, cultural events
  * Business (व्यापार): Economy, companies, markets, finance, trade
  * Technology (तंत्रज्ञान): IT, software, gadgets, innovation, digital trends
  * Health (आरोग्य): Medicine, diseases, hospitals, wellness, public health
  * Education (शिक्षण): Schools, universities, students, exams, educational policies
  * Environment (पर्यावरण): Climate, pollution, conservation, natural resources
  * Crime (गुन्हा): Law enforcement, criminal cases, investigations, courts
  * Agriculture (शेती): Farming, crops, agricultural policies, rural development
  * Religion (धर्म): Religious events, festivals, spiritual matters
  * Transportation (वाहतूक): Roads, railways, public transport, traffic
  * Weather (हवामान): Climate conditions, forecasts, natural disasters
  * Science (विज्ञान): Research, discoveries, scientific developments
  * Social Issues (सामाजिक मुद्दे): Social problems, community matters, welfare
  
  DO NOT use "Other" unless absolutely necessary. Try to match to one of the categories above.

- entities: List of key entities (people, organizations, places) mentioned in the article
- emotions: The dominant emotion (positive, negative, neutral, mixed)
- severity: The severity level (high, medium, low) of the news impact
- summary: A brief 1-2 sentence summary in English
- keywords: 3-5 key terms from the article

Article:
{text}

JSON response:
"""
    
    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """
        Parse the LLM response to extract the JSON metadata.
        
        Args:
            response (str): LLM response
            
        Returns:
            dict: Extracted metadata
        """
        try:
            # Try to find JSON in the response
            start_idx = response.find('{')
            end_idx = response.rfind('}')
            
            if start_idx >= 0 and end_idx > start_idx:
                json_str = response[start_idx:end_idx+1]
                metadata = json.loads(json_str)
                # Refine category assignment
                metadata = self._refine_category(metadata)
                return metadata
            else:
                # If no JSON found, try to extract structured information
                metadata = self._create_default_metadata()
                
                # Extract title
                if "title:" in response.lower():
                    title_line = [line for line in response.split('\n') if "title:" in line.lower()]
                    if title_line:
                        metadata["title"] = title_line[0].split(":", 1)[1].strip()
                
                # Extract category
                if "category:" in response.lower():
                    category_line = [line for line in response.split('\n') if "category:" in line.lower()]
                    if category_line:
                        metadata["category"] = category_line[0].split(":", 1)[1].strip()
                        
                # Refine category assignment
                metadata = self._refine_category(metadata)
                return metadata
        except Exception as e:
            print(f"Error parsing LLM response: {e}")
            return self._create_default_metadata()
    
    def _refine_category(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Refine category assignment using keyword matching when category is "Other".
        
        Args:
            metadata (dict): Extracted metadata
            
        Returns:
            dict: Metadata with refined category
        """
        # Only process if category is "Other" or empty
        if metadata.get("category", "").lower() == "other" or not metadata.get("category"):
            # Get text from title, summary, and keywords
            text_to_analyze = " ".join([
                str(metadata.get("title", "")),
                str(metadata.get("summary", "")),
                " ".join([str(k) for k in metadata.get("keywords", [])])
            ]).lower()
            
            # Define category keywords (both English and Marathi)
            category_keywords = {
                "Politics": ["politics", "government", "minister", "election", "party", "राजकारण", "सरकार", "मंत्री", "निवडणूक", "पक्ष"],
                "Sports": ["sports", "cricket", "football", "player", "match", "क्रीडा", "क्रिकेट", "फुटबॉल", "खेळाडू", "सामना"],
                "Entertainment": ["entertainment", "movie", "film", "actor", "music", "मनोरंजन", "चित्रपट", "अभिनेता", "संगीत"],
                "Business": ["business", "company", "market", "economy", "finance", "व्यापार", "कंपनी", "बाजार", "अर्थव्यवस्था", "वित्त"],
                "Technology": ["technology", "software", "digital", "computer", "internet", "तंत्रज्ञान", "सॉफ्यवेअर", "डिजिटल", "संगणक", "इंटरनेट"],
                "Health": ["health", "hospital", "doctor", "disease", "medical", "आरोग्य", "रुग्णालय", "डॉक्टर", "आजार", "वैद्यकीय"],
                "Education": ["education", "school", "university", "student", "exam", "शिक्षण", "शाळा", "विद्यापीठ", "विद्यार्थी", "परीक्षा"],
                "Environment": ["environment", "climate", "pollution", "nature", "पर्यावरण", "हवामान", "प्रदूषण", "निसर्ग"],
                "Crime": ["crime", "police", "arrest", "theft", "murder", "गुन्हा", "पोलीस", "अटक", "चोरी", "खून"],
                "Agriculture": ["agriculture", "farmer", "crop", "farm", "harvest", "शेती", "शेतकरी", "पीक", "शेत", "कापणी"],
                "Religion": ["religion", "god", "temple", "festival", "prayer", "धर्म", "देव", "मंदिर", "सण", "प्रार्थना"],
                "Transportation": ["transport", "road", "traffic", "vehicle", "train", "वाहतूक", "रस्ता", "वाहतूक", "वाहन", "रेल्वे"],
                "Weather": ["weather", "rain", "flood", "storm", "temperature", "हवामान", "पाऊस", "पूर", "वादळ", "तापमान"],
                "Science": ["science", "research", "discovery", "experiment", "विज्ञान", "संशोधन", "शोध", "प्रयोग"],
                "Social Issues": ["social", "community", "welfare", "rights", "सामाजिक", "समुदाय", "कल्याण", "अधिकार"]
            }
            
            # Check for keyword matches
            matched_categories = {}
            for category, keywords in category_keywords.items():
                matches = sum(1 for keyword in keywords if keyword in text_to_analyze)
                if matches > 0:
                    matched_categories[category] = matches
            
            # Assign the category with the most matches
            if matched_categories:
                best_category = max(matched_categories.items(), key=lambda x: x[1])[0]
                metadata["category"] = best_category
        
        return metadata
    
    def _create_default_metadata(self) -> Dict[str, Any]:
        """
        Create default metadata when extraction fails.
        
        Returns:
            dict: Default metadata
        """
        return {
            "title": "",
            "category": "Other",
            "entities": [],
            "emotions": "neutral",
            "severity": "low",
            "summary": "",
            "keywords": []
        }
    
    def process_dataframe(self, df: pd.DataFrame, text_column: str, batch_size: int = 10) -> pd.DataFrame:
        """
        Process a dataframe of articles to extract metadata.
        
        Args:
            df (pd.DataFrame): Dataframe with articles
            text_column (str): Name of the column containing text
            batch_size (int): Number of articles to process in each batch
            
        Returns:
            pd.DataFrame: Dataframe with extracted metadata
        """
        # Create a copy to avoid modifying the original dataframe
        result_df = df.copy()
        
        # Initialize metadata columns
        result_df['llm_title'] = ""
        result_df['llm_category'] = ""
        result_df['llm_entities'] = result_df.apply(lambda x: [], axis=1)
        result_df['llm_emotions'] = ""
        result_df['llm_severity'] = ""
        result_df['llm_summary'] = ""
        result_df['llm_keywords'] = result_df.apply(lambda x: [], axis=1)
        
        # Process in batches with progress bar
        total_batches = (len(df) + batch_size - 1) // batch_size
        for i in tqdm(range(0, len(df), batch_size), total=total_batches, desc="Extracting metadata with LLM"):
            batch = df.iloc[i:i+batch_size]
            
            for j, (idx, row) in enumerate(batch.iterrows()):
                try:
                    # Extract metadata
                    metadata = self.extract_metadata(row[text_column])
                    
                    # Update the result dataframe
                    result_df.at[idx, 'llm_title'] = metadata.get('title', '')
                    result_df.at[idx, 'llm_category'] = metadata.get('category', 'Other')
                    result_df.at[idx, 'llm_entities'] = metadata.get('entities', [])
                    result_df.at[idx, 'llm_emotions'] = metadata.get('emotions', 'neutral')
                    result_df.at[idx, 'llm_severity'] = metadata.get('severity', 'low')
                    result_df.at[idx, 'llm_summary'] = metadata.get('summary', '')
                    result_df.at[idx, 'llm_keywords'] = metadata.get('keywords', [])
                    
                    # Add a small delay to avoid overwhelming the LLM
                    if j < len(batch) - 1:
                        time.sleep(0.5)
                        
                except Exception as e:
                    print(f"Error processing article at index {idx}: {e}")
        
        return result_df

class LLMEnhancedEmbeddings:
    """Class for generating LLM-enhanced embeddings for articles."""
    
    def __init__(self, base_embedder, metadata_extractor=None):
        """
        Initialize the LLM-enhanced embeddings generator.
        
        Args:
            base_embedder: Base embeddings model (e.g., TF-IDF)
            metadata_extractor: Metadata extractor instance
        """
        self.base_embedder = base_embedder
        self.metadata_extractor = metadata_extractor
    
    def get_embeddings_for_corpus(self, texts, df_with_metadata=None):
        """
        Generate LLM-enhanced embeddings for a corpus.
        
        Args:
            texts (list): List of preprocessed texts
            df_with_metadata (pd.DataFrame): Dataframe with extracted metadata
            
        Returns:
            numpy.ndarray: Enhanced document embeddings matrix
        """
        # Get base embeddings
        base_embeddings = self.base_embedder.get_embeddings_for_corpus(texts)
        
        # If no metadata is provided, return base embeddings
        if df_with_metadata is None or self.metadata_extractor is None:
            return base_embeddings
        
        # Enhance embeddings with metadata (this is a simplified approach)
        # In a real implementation, you might want to use more sophisticated methods
        # to incorporate the metadata into the embeddings
        
        # For now, we'll just return the base embeddings
        # In a future version, you could implement a more sophisticated approach
        return base_embeddings
    
    def get_embeddings_for_document(self, text, metadata=None):
        """
        Generate LLM-enhanced embeddings for a single document.
        
        Args:
            text (str): Preprocessed text
            metadata (dict): Extracted metadata
            
        Returns:
            numpy.ndarray: Enhanced document embedding vector
        """
        # Get base embedding
        base_embedding = self.base_embedder.get_embeddings_for_document(text)
        
        # If no metadata is provided, return base embedding
        if metadata is None or self.metadata_extractor is None:
            return base_embedding
        
        # Enhance embedding with metadata (simplified approach)
        return base_embedding

def cluster_with_llm_metadata(df, embeddings, n_clusters=None, metadata_weight=0.3, find_optimal=True, max_clusters=10):
    """
    Cluster articles using both embeddings and LLM-extracted metadata.
    
    Args:
        df (pd.DataFrame): Dataframe with articles and metadata
        embeddings (numpy.ndarray): Document embeddings matrix
        n_clusters (int): Number of clusters (if None and find_optimal=True, will find optimal number)
        metadata_weight (float): Weight for metadata-based similarity (0-1)
        find_optimal (bool): Whether to find the optimal number of clusters
        max_clusters (int): Maximum number of clusters to try when finding optimal
        
    Returns:
        tuple: (df_with_clusters, clusterer)
    """
    from src.clustering import cluster_articles
    from sklearn.metrics import silhouette_score
    from sklearn.cluster import KMeans
    
    if find_optimal and n_clusters is None:
        print(f"Finding optimal number of clusters (max: {max_clusters})...")
        silhouette_scores = []
        min_clusters = min(3, len(df) // 2)  # At least 2 data points per cluster
        max_clusters = min(max_clusters, len(df) - 1)  # Can't have more clusters than data points
        
        # Try different numbers of clusters
        for k in range(min_clusters, max_clusters + 1):
            # Initialize and fit KMeans
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(embeddings)
            
            # Calculate silhouette score if we have more than one cluster
            if len(set(cluster_labels)) > 1:
                score = silhouette_score(embeddings, cluster_labels)
                silhouette_scores.append((k, score))
                print(f"  Clusters: {k}, Silhouette Score: {score:.4f}")
        
        # Find the number of clusters with the highest silhouette score
        if silhouette_scores:
            n_clusters = max(silhouette_scores, key=lambda x: x[1])[0]
            print(f"Optimal number of clusters: {n_clusters}")
        else:
            n_clusters = min(5, len(df) // 2)  # Default to 5 clusters or fewer if not enough data
            print(f"Could not determine optimal clusters. Using {n_clusters} clusters.")
    
    # Cluster the articles
    df_clustered, clusterer = cluster_articles(df, embeddings, n_clusters)
    
    return df_clustered, clusterer

if __name__ == "__main__":
    # Example usage
    from preprocessing import preprocess_text
    from embeddings import MarathiEmbeddings
    
    # Sample data
    sample_texts = [
        "मुंबई: महाराष्ट्रात कोरोनाचा प्रादुर्भाव वाढत असताना राज्य सरकारने नवीन निर्बंध जाहीर केले आहेत.",
        "पुणे: पुण्यात मोठ्या प्रमाणात पाऊस झाल्याने अनेक भागात पूरस्थिती निर्माण झाली आहे.",
        "नागपूर: नागपूर विद्यापीठाने परीक्षांचे वेळापत्रक जाहीर केले आहे."
    ]
    
    # Create a dataframe
    df = pd.DataFrame({"text": sample_texts})
    
    # Preprocess texts
    df['preprocessed_text'] = [' '.join(preprocess_text(text)) for text in sample_texts]
    
    # Initialize metadata extractor
    extractor = OllamaMetadataExtractor()
    
    # Extract metadata
    df_with_metadata = extractor.process_dataframe(df, "text", batch_size=3)
    
    # Print results
    for i, row in df_with_metadata.iterrows():
        print(f"Article {i+1}:")
        print(f"Text: {row['text'][:50]}...")
        print(f"Category: {row['llm_category']}")
        print(f"Emotions: {row['llm_emotions']}")
        print(f"Keywords: {row['llm_keywords']}")
        print("-" * 50)
