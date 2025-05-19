# Marathi News Article Clustering

This project provides a user-friendly system for clustering Marathi news articles based on content similarity, with optional LLM-enhanced metadata extraction. It offers both an interactive Streamlit interface and a lightweight command-line pipeline, combining basic NLP techniques with advanced visualization and analysis tools.

## Features

- **File Upload**: Supports CSV, TXT, or PDF files containing Marathi news articles.
- **Text Preprocessing**: Cleans and preprocesses Marathi text for analysis.
- **TF-IDF Vectorization**: Represents articles as TF-IDF vectors for clustering.
- **LLM Metadata Extraction**: Extracts metadata (e.g., categories, emotions) using Ollama (optional).
- **Customizable Clustering**: Adjusts the number of K-means clusters via user input.
- **Interactive Visualizations**: Includes cluster distributions, 2D t-SNE projections, top terms per cluster, and metadata analysis (e.g., category/emotion distributions).
- **Results Download**: Saves clustered data and visualizations for further analysis.
- **Evaluation Metrics**: Provides cluster quality metrics (e.g., silhouette score).
- **Simplified Pipeline**: Offers a minimal-dependency command-line option for basic clustering.

## Prerequisites

1. **Python 3.8+**
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download NLTK stopwords (if needed):
   ```bash
   python -c "import nltk; nltk.download('stopwords')"
   ```
4. **Optional LLC Support**:
   - Install Ollama: [https://ollama.com/](https://ollama.com/)
   - Pull an LLM model (e.g., Llama 3):
     ```bash
     ollama pull llama3:8b
     ```
5. **Optional PDF Support**:
   - Install Tesseract OCR: [https://github.com/tesseract-ocr/tesseract](https://github.com/tesseract-ocr/tesseract)
   - Install Marathi language data:
     ```bash
     # On Linux
     sudo apt-get install tesseract-ocr-mar
     # On Windows
     # Download mar.traineddata from https://github.com/tesseract-ocr/tessdata
     # Place in Tesseract's tessdata directory (e.g., C:\Program Files\Tesseract-OCR\tessdata)
     ```

## Setup

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd marathi-news-clustering
   ```
2. Place Marathi news articles in the `data/raw` directory (CSV, TXT, or PDF format).
3. Use the sample data in `data/raw/sample_marathi_news.csv` for testing.

## Running the Application

### Streamlit Interface
Start the interactive Streamlit app:
```bash
streamlit run src/app_llm.py
```

**Usage Instructions**:
1. **Upload File**: Upload a CSV (with a 'text' column), TXT (each line as an article), or PDF (each page as an article).
2. **Configure Settings**: Specify the text column (for CSV), enable/disable LLM metadata extraction, select an Ollama model, and set the number of clusters.
3. **Process and Cluster**: Click "Process and Cluster Articles" to preprocess, extract metadata (if enabled), and cluster.
4. **View Results**: Explore cluster distributions, 2D visualizations, top terms, metadata analysis, and download results.

### Command-Line Pipeline
Run the basic clustering pipeline with default settings (sample data, 5 clusters):
```bash
python src/main.py
```

**Advanced Usage**:
```bash
python src/main.py --input_file data/raw/your_data.csv --text_column article_text --n_clusters 8 --output_file results/output.csv
```

**Parameters**:
- `--input_file`: Path to CSV/TXT file.
- `--text_column`: Text column name (for CSV).
- `--n_clusters`: Number of clusters.
- `--output_file`: Output file path.

## Project Structure

- `data/`: Raw and processed data.
  - `raw/`: Input files (e.g., `sample_marathi_news.csv`).
- `src/`: Source code.
  - `preprocessing.py`: Text cleaning and preprocessing.
  - `embeddings.py`: TF-IDF vectorization.
  - `clustering.py`: K-means clustering.
  - `visualize.py`: Visualization tools.
  - `app_llm.py`: Streamlit app.
  - `main.py`: Command-line pipeline.
- `models/`: Saved models.
- `results/`: Clustering results and visualizations.

## Visualizations

- **Cluster Distribution**: Article counts per cluster.
- **2D t-SNE Visualization**: Clusters in 2D space.
- **Top Terms per Cluster**: Frequent terms per cluster.
- **Metadata Analysis** (LLM-enabled): Category/emotion distributions.

Visualizations are saved to the `results` directory.

## Sample Data

Test the application with `data/raw/sample_marathi_news.csv`, which contains sample Marathi news articles.

## Troubleshooting

- **Ollama Not Running**: Start Ollama or use the app without LLM features.
- **Model Not Found**: Pull the model with `ollama pull <model_name>`.
- **Large Files**: LLM metadata extraction may be slow for large datasets.
- **PDF OCR Issues**: Ensure Tesseract and Marathi language data are installed correctly.

## Notes

- LLM metadata extraction processes articles sequentially to avoid overloading the model.
- Clustering uses TF-IDF vectors, enhanced by LLM metadata if enabled.
- Larger LLM models improve metadata quality but require more resources.
- The command-line pipeline minimizes dependencies for simplicity and ease of use.