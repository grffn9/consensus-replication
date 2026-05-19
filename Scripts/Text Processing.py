#!/usr/bin/env python3
"""
This script processes a CSV file ("articles.csv") that contains news
articles with the columns: source, title, publication_date, url, and scraped_text.
The processing steps are:
  1. Cleaning & Normalization:
       - Remove HTML tags using BeautifulSoup.
       - Lowercase the text.
       - Remove known boilerplate phrases (e.g., "Getty Images", "Watch:", timestamps like "7 hours ago").
       - Normalize extra whitespace.
  2. Tokenization & Stopword Removal:
       - Use spaCy to tokenize the text.
       - Remove spaCy’s built-in stopwords plus a custom list of political jargon.
  3. Lemmatization:
       - Replace tokens with their lemmas (e.g., “running” → “run”).
  4. Sentence Splitting:
       - Break the article into sentences using spaCy’s sentence boundary detection.
  5. Named Entity Recognition (NER):
       - Extract named entities (e.g., persons, organizations, geopolitical entities) from the cleaned text.
  6. Sentiment Analysis & Stance Detection:
       - Use NLTK’s VADER to compute a sentiment score and determine a basic stance (positive, negative, or neutral).

The script adds the following new columns to the DataFrame:
    - cleaned_text: the cleaned and normalized version of scraped_text.
    - tokens: a list of lemmatized tokens with stopwords and punctuation removed.
    - sentences: a list of sentences from the cleaned text.
    - named_entities: a list of tuples for each recognized entity in the format (entity_text, entity_label).
    - sentiment_score: the VADER compound sentiment score.
    - sentiment_label: a label ("positive", "negative", or "neutral") based on the compound score.

Finally, the processed DataFrame is written to "text_processed_results.csv".
"""

import pandas as pd
import re
from bs4 import BeautifulSoup
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
import nltk

# Download required NLTK resources
nltk.download("vader_lexicon")
nltk.download("punkt")

from nltk.sentiment.vader import SentimentIntensityAnalyzer

# --- Custom Stopwords ---
custom_stopwords = {
    "government", "minister", "mp", "president", "prime", "politician",
    "conservative", "labour", "bbc", "tory", "democrat", "republican"
}
STOP_WORDS |= custom_stopwords  # Add custom words to spaCy's stopword set

# --- Load spaCy English model ---
# Ensure you have installed it with: python -m spacy download en_core_web_sm
nlp = spacy.load("en_core_web_sm")

# Initialize NLTK's VADER sentiment analyzer.
sia = SentimentIntensityAnalyzer()


def clean_text(text):
    """
    Clean and normalize text by:
      - Removing HTML tags using BeautifulSoup.
      - Lowercasing the text.
      - Removing boilerplate patterns (e.g., "Getty Images", "Watch:" and time stamps).
      - Collapsing multiple whitespace characters.
    """
    if not isinstance(text, str):
        return ""

    # Remove HTML tags using BeautifulSoup.
    soup = BeautifulSoup(text, "html.parser")
    text = soup.get_text(separator=" ")

    # Lowercase the text.
    text = text.lower()

    # Define boilerplate regex patterns.
    boilerplate_patterns = [
        r"\bgetty images\b",  # Remove "getty images"
        r"\bwatch:\b",        # Remove "watch:"
        r"\b\d+\s+(hours?|minutes?|secs?|seconds?)\s+ago\b",  # Remove timestamps like "7 hours ago"
        # Additional patterns can be added here.
    ]
    for pattern in boilerplate_patterns:
        text = re.sub(pattern, "", text)

    # Normalize whitespace.
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def process_text(text):
    """
    Process the cleaned text with spaCy:
      - Split into sentences.
      - Tokenize and perform lemmatization.
      - Remove stopwords, punctuation, and whitespace tokens.
      - Extract named entities.
    Returns:
      tokens: List of lemmatized tokens (strings).
      sentences: List of sentences (strings).
      entities: List of tuples (entity_text, entity_label)
    """
    doc = nlp(text)

    # Sentence splitting: get non-empty sentences.
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]

    # Tokenization, lemmatization, and stopword/punctuation removal.
    tokens = [
        token.lemma_
        for token in doc
        if (not token.is_stop) and (not token.is_punct) and (not token.is_space)
    ]

    # Named Entity Recognition (NER): extract (text, label) for each entity.
    entities = [(ent.text, ent.label_) for ent in doc.ents]

    return tokens, sentences, entities


def analyze_sentiment(text):
    """
    Analyze sentiment using VADER.
    Returns:
      sentiment_score: compound score.
      sentiment_label: 'positive', 'negative', or 'neutral' based on thresholds.
    """
    sentiment = sia.polarity_scores(text)
    compound = sentiment.get("compound", 0.0)
    # Define thresholds (these can be tuned further)
    if compound >= 0.05:
        label = "positive"
    elif compound <= -0.05:
        label = "negative"
    else:
        label = "neutral"
    return compound, label


def main():
    input_filename = "C:/Users/griff/OneDrive/Desktop/Misinformation Project/Consensus/Datasets/articles.csv"
    try:
        df = pd.read_csv(input_filename)
    except Exception as e:
        print(f"Error reading {input_filename}: {e}")
        return

    # --- Remove rows missing any required data ---
    required_columns = ["source", "title", "publication_date", "url", "summary", "scraped_text", "bias_rating",
                        "factuality_rating"]
    df.replace("", pd.NA, inplace=True)
    df.dropna(subset=required_columns, inplace=True)

    # Prepare lists for new processed fields.
    cleaned_texts = []
    tokens_list = []
    sentences_list = []
    named_entities_list = []
    sentiment_scores = []
    sentiment_labels = []

    # Process each article.
    for idx, row in df.iterrows():
        raw_text = row.get("scraped_text", "")
        cleaned = clean_text(raw_text)
        tokens, sentences, entities = process_text(cleaned)
        compound, sent_label = analyze_sentiment(cleaned)

        cleaned_texts.append(cleaned)
        tokens_list.append(tokens)
        sentences_list.append(sentences)
        named_entities_list.append(entities)
        sentiment_scores.append(compound)
        sentiment_labels.append(sent_label)

    # Add new columns to the DataFrame.
    df["cleaned_text"] = cleaned_texts
    df["tokens"] = tokens_list
    df["sentences"] = sentences_list
    df["named_entities"] = named_entities_list
    df["sentiment_score"] = sentiment_scores
    df["sentiment_label"] = sentiment_labels

    # Remove any row where the cleaned_text starts with "failed to extract"
    # (Comparison is done on lowercase text.)
    df = df[~df["cleaned_text"].str.startswith("failed to extract")]

    # Save the processed DataFrame.
    output_filename = "C:/Users/griff/OneDrive/Desktop/Misinformation Project/Consensus/Datasets/text_processed_results.csv"
    try:
        df.to_csv(output_filename, index=False)
        print(f"Processed articles saved to {output_filename}")
    except Exception as e:
        print(f"Error writing {output_filename}: {e}")


if __name__ == "__main__":
    main()
