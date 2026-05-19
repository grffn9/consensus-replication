#!/usr/bin/env python3
"""
Consensus Algorithm Script with Source Weighting and Multi-Document Summarization

This script implements a consensus algorithm that:
  1. Loads processed news articles from a CSV file.
  2. Computes text embeddings for each article’s cleaned_text.
  3. Uses DBSCAN clustering (with cosine distance) to group similar articles.
  4. For each cluster (excluding noise), computes aggregate metrics and generates a cluster summary:
       - Number of articles.
       - Weighted cluster size (using source weights derived from factuality ratings).
       - Weighted average and weighted standard deviation of sentiment scores.
       - Top keywords extracted via TF-IDF.
       - Common named entities.
       - A consensus metric that rewards large, consistent clusters:
             consensus_metric = weighted_cluster_size / (1 + weighted_std_sentiment)
       - A **cluster_summary** (base claim) generated via multi-document summarization.
  5. Saves a consensus summary to CSV.

Dependencies:
  - pandas, numpy
  - sentence-transformers
  - scikit-learn
  - transformers
  - nltk (for sentiment, though already computed in preprocessing)
  - json, collections
"""

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
import json
from transformers import pipeline  # For summarization


# --- Functions ---

def load_processed_data(csv_path):
    """Load the processed articles CSV."""
    df = pd.read_csv(csv_path)
    return df


def compute_embeddings(texts, model_name="all-MiniLM-L6-v2"):
    """
    Compute sentence embeddings for a list of texts using SentenceTransformer.
    Returns the embeddings (as a Torch tensor) and the model.
    """
    print("Loading embedding model...")
    model = SentenceTransformer(model_name)
    print("Computing embeddings...")
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_tensor=True)
    return embeddings, model


def cluster_embeddings(embeddings, eps=0.3, min_samples=2):
    """
    Cluster embeddings using DBSCAN with cosine distance.
    Returns cluster labels for each article.
    """
    print("Clustering embeddings using DBSCAN...")
    # Convert tensor to numpy array for DBSCAN.
    clustering_model = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine")
    cluster_labels = clustering_model.fit_predict(embeddings.cpu().numpy())
    return cluster_labels


def get_top_keywords(texts, n=5):
    """
    Given a list of texts, use TF-IDF to extract the top n keywords.
    Filters out empty/whitespace-only documents and, if an empty vocabulary
    error occurs, returns an empty list.
    """
    filtered_texts = [text for text in texts if text.strip() != ""]
    if not filtered_texts:
        return []
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    try:
        tfidf_matrix = vectorizer.fit_transform(filtered_texts)
    except ValueError as e:
        print("Warning: Empty vocabulary in get_top_keywords. Returning empty keyword list.")
        return []
    feature_names = vectorizer.get_feature_names_out()
    scores = tfidf_matrix.sum(axis=0).A1
    top_indices = np.argsort(scores)[::-1][:n]
    top_keywords = [feature_names[i] for i in top_indices]
    return top_keywords


def aggregate_named_entities(named_entities_lists):
    """
    Aggregate named entities across a list of articles.
    Each element is expected to be either a Python list of tuples (entity, label)
    or a JSON string representation.
    Returns a list of (entity, frequency) for the top entities.
    """
    all_entities = []
    for entities in named_entities_lists:
        if isinstance(entities, str):
            try:
                entities = json.loads(entities)
            except Exception:
                continue
        all_entities.extend([ent[0].lower() for ent in entities])
    counter = Counter(all_entities)
    common_entities = counter.most_common(5)
    return common_entities


def summarize_cluster_text(texts, summarizer, max_length=150, min_length=40):
    """
    Given a list of texts, concatenate them (with simple truncation if needed)
    and use the provided summarization pipeline to generate a summary.
    """
    concatenated_text = " ".join(texts)
    # Truncate to a fixed number of characters (e.g., 1000) to avoid exceeding model limits.
    concatenated_text = concatenated_text[:1000]
    summary = summarizer(concatenated_text, max_length=max_length, min_length=min_length, do_sample=False)
    return summary[0]['summary_text']


def extract_base_claim(cluster_df, embeddings, model):
    """
    Compute the centroid of a cluster's embeddings and return the cleaned_text of the article
    whose embedding is closest to the centroid.
    """
    centroid = embeddings.mean(dim=0)
    cos_scores = util.cos_sim(embeddings, centroid.unsqueeze(0)).squeeze(1)
    best_idx = int(np.argmax(cos_scores.cpu().numpy()))
    return cluster_df.iloc[best_idx]['cleaned_text']


def consensus_summary(df, all_embeddings, model, summarizer):
    """
    For each cluster (excluding noise cluster with label -1), compute aggregate metrics
    and generate a representative cluster summary via multi-document summarization.

    Each article is weighted by a source weight computed from its factuality rating:
        weight = 1 / (1 + factuality_rating)

    Returns a summary DataFrame with:
      - Weighted cluster size, weighted average sentiment, weighted sentiment std.
      - Top keywords, common named entities, consensus metric.
      - A 'base claim' (article closest to cluster centroid).
      - A 'cluster_summary' generated via multi-document summarization.
    """
    consensus_data = []

    # Convert factuality_rating to float and compute source weight.
    df['factuality_rating_float'] = df['factuality_rating'].astype(float)
    df['source_weight'] = df['factuality_rating_float'].apply(lambda x: 1 / (1 + x))
    counter = 0

    clusters = df['cluster'].unique()
    for cluster in clusters:
        print(counter)
        counter += 1
        if cluster == -1:
            continue
        cluster_df = df[df['cluster'] == cluster].reset_index(drop=True)
        # Get the embeddings for the cluster.
        cluster_embeddings = all_embeddings[cluster_df.index]

        weighted_cluster_size = cluster_df['source_weight'].sum()
        sentiments = cluster_df['sentiment_score'].values
        weights = cluster_df['source_weight'].values
        
        if len(cluster_df) > 1 and weights.sum() > 0:
            weighted_avg = np.average(sentiments, weights=weights)
            weighted_var = np.average((sentiments - weighted_avg) ** 2, weights=weights)
            weighted_std = np.sqrt(weighted_var)
        else:
            weighted_avg = sentiments[0] if len(sentiments) > 0 else 0.0
            weighted_std = 0.0

        top_keywords = get_top_keywords(cluster_df['cleaned_text'].tolist(), n=5)
        common_entities = aggregate_named_entities(cluster_df['named_entities'].tolist())
        consensus_metric = weighted_cluster_size / (1 + weighted_std)

        # Extract a base claim from the cluster.
        base_claim = extract_base_claim(cluster_df, cluster_embeddings, model)

        # Generate a multi-document summary for the cluster.
        # If the cluster is large, take the top 5 articles (by source_weight) as representatives.
        if len(cluster_df) > 5:
            cluster_texts = cluster_df.sort_values(by='source_weight', ascending=False)['cleaned_text'].head(5).tolist()
        else:
            cluster_texts = cluster_df['cleaned_text'].tolist()
        cluster_summary = summarize_cluster_text(cluster_texts, summarizer, max_length=150, min_length=40)

        consensus_data.append({
            "cluster": cluster,
            "num_articles": len(cluster_df),
            "weighted_cluster_size": weighted_cluster_size,
            "weighted_avg_sentiment": weighted_avg,
            "weighted_std_sentiment": weighted_std,
            "top_keywords": top_keywords,
            "common_entities": common_entities,
            "consensus_metric": consensus_metric,
            "base_claim": base_claim,
            "cluster_summary": cluster_summary
        })
    return pd.DataFrame(consensus_data)


# --- Main Pipeline ---

def main():
    input_csv = "Datasets/text_processed_results.csv"
    df = load_processed_data(input_csv)

    # Work only with articles that have non-empty cleaned_text.
    df = df[df["cleaned_text"].notna() & (df["cleaned_text"].str.strip() != "")]
    texts = df['cleaned_text'].tolist()

    # Compute embeddings.
    embeddings, model = compute_embeddings(texts)

    # Cluster the embeddings.
    cluster_labels = cluster_embeddings(embeddings, eps=0.3, min_samples=2)
    df['cluster'] = cluster_labels
    num_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    print(f"Found {num_clusters} clusters (excluding noise).")

    # Initialize the summarization pipeline.
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

    # Compute consensus metrics and generate cluster summaries.
    consensus_df = consensus_summary(df, embeddings, model, summarizer)
    consensus_df.sort_values(by="consensus_metric", ascending=False, inplace=True)

    # Save the consensus summary.
    output_csv = "Datasets/consensus_summary.csv"
    consensus_df.to_csv(output_csv, index=False)
    print(f"Consensus summary saved to {output_csv}")


if __name__ == "__main__":
    main()
