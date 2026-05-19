# Consensus Replication

## Description
This project implements an automated pipeline for tracking, collecting, and analyzing news articles to determine media consensus, bias, and factuality. It leverages NLP techniques to analyze article sentiment, perform hierarchical topic modeling, and use clustering (DBSCAN) and text embeddings to evaluate media consensus across multiple major news sources.

## Project Structure
```
consensus-replication/
├── data/
│   ├── raw/                 # Raw downloaded datasets (articles.csv)
│   ├── processed/           # Transformed datasets (text_processed_results.csv, consensus_summary.csv)
│   └── state/               # API tracking and application state (newsapi_state.txt)
├── src/
│   ├── data_collection.py   # Asynchronously fetches articles from RSS feeds and NewsAPI
│   ├── text_processing.py   # Cleans data, normalizes text, removes stopwords, performs NER and sentiment analysis
│   ├── topic_modeling.py    # Generates hierarchical topic models (BERTopic) and segmentations
│   └── consensus_algorithm.py # Computes DBSCAN clustering and generates multi-document consensus summaries
├── tests/                   # Test suite for unit and integration testing
└── README.md                # Project documentation
```

## Installation
1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd consensus-replication
   ```
2. Create and activate a Python virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```
3. Install required dependencies:
   ```bash
   pip install pandas beautifulsoup4 newspaper3k aiohttp feedparser spacy nltk bertopic scikit-learn
   # Download required NLTK and spaCy packages
   python -m spacy download en_core_web_sm
   ```

## Usage

Run the scripts linearly to build the pipeline:
1. **Data Collection:** `python src/data_collection.py`
2. **Text Processing:** `python src/text_processing.py`
3. **Topic Modeling:** `python src/topic_modeling.py`
4. **Consensus Algorithm:** `python src/consensus_algorithm.py`
