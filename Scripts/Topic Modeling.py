#!/usr/bin/env python3
"""
Hierarchical Topic Modeling Pipeline:
  1. Full Article Topic Modeling: Unsupervised modeling using BERTopic.
     - Groups full articles into topics.
     - Generates human-readable topic labels.
  2. Per-Article Topic Segmentation & Refinement:
     - Each article is segmented into chunks (default: 5 sentences per chunk).
     - If multiple topics appear in an article’s chunks, a sub-topic model is run on those chunks
       to refine the segmentation.
  3. Hierarchical (Multiple Layers) Topic Modeling:
     - Articles are grouped by their primary (full article) topic.
     - For each group, refined segments are aggregated and a secondary topic model is run
       to capture subthemes within the group.
Results are saved to “topic_modeling_results.csv”.
"""

import os
import json
import pandas as pd
from bertopic import BERTopic
import nltk
from nltk.tokenize import sent_tokenize

# Download the NLTK Punkt tokenizer (if not already present)
nltk.download("punkt")


### 1. Full Article Topic Modeling

def load_articles(csv_file="C:/Users/griff/OneDrive/Desktop/Misinformation Project/Consensus/Datasets/text_processed_results.csv"):
    """
    Load the scraped articles from CSV.
    Only articles with non-empty scraped_text are considered.
    """
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"File {csv_file} not found.")
    df = pd.read_csv(csv_file)
    # Work only with articles that have non-empty full text
    df = df[df["scraped_text"].notna() & (df["scraped_text"].str.strip() != "")]
    return df


def run_full_article_topic_modeling(docs):
    """
    Fit a BERTopic model on the full texts (docs) and return:
      - topic_model: the fitted BERTopic model
      - topics: list of topic numbers for each document
      - probs: topic probability for each document (optional)
    """
    print("Fitting BERTopic on full articles...")
    topic_model = BERTopic(verbose=True)
    topics, probs = topic_model.fit_transform(docs)
    return topic_model, topics, probs


def generate_topic_mapping(topic_model):
    """
    Get topic information from the model and create a mapping from topic number to a human-readable name.
    The topic “name” is derived from the top representative keywords.
    """
    topic_info = topic_model.get_topic_info()  # DataFrame with columns Topic, Count, Name, etc.
    mapping = {}
    for _, row in topic_info.iterrows():
        if row.Topic == -1:
            mapping[row.Topic] = "Outlier"
        else:
            mapping[row.Topic] = row.Name  # row.Name is a string of representative words
    return mapping


### 2. Per-Article Topic Segmentation & Refinement

def segment_article(article_text, chunk_size=5):
    """
    Split an article into chunks. Each chunk consists of chunk_size sentences.
    Returns a list of text chunks.
    """
    sentences = sent_tokenize(article_text)
    chunks = [" ".join(sentences[i:i + chunk_size]) for i in range(0, len(sentences), chunk_size)]
    return chunks


def assign_chunk_topics(chunks, topic_model):
    """
    Given a list of text chunks and a BERTopic model,
    assign a topic to each chunk.
    Returns a list of topic assignments.
    """
    chunk_topics, _ = topic_model.transform(chunks)
    return chunk_topics


def refine_segments(chunks, topic_model, min_chunks_for_refinement=3):
    """
    For an article’s chunks, first assign topics.
    If multiple unique topics are present in the chunks and there are at least min_chunks_for_refinement,
    then re-run a sub-topic model on these chunks to refine the segmentation.
    Returns a list of dictionaries with keys:
        - "chunk": text chunk
        - "original_topic": topic from the full-article model
        - "refined_topic": topic from the sub-model (or same as original if refinement is skipped)
    """
    original_topics = assign_chunk_topics(chunks, topic_model)
    if len(chunks) < min_chunks_for_refinement or len(set(original_topics)) <= 1:
        # Not enough chunks or only one topic is present; skip refinement.
        refined_topics = original_topics
    else:
        print("  Refining segments for an article with multiple topics...")
        try:
            import umap
            import hdbscan
            # Set UMAP parameters based on the number of chunks.
            n_neighbors = max(2, min(15, len(chunks) - 1))
            n_components = max(1, min(5, len(chunks) - 2))
            custom_umap = umap.UMAP(n_neighbors=n_neighbors, n_components=n_components, random_state=42)
            # Build a custom HDBSCAN model for small numbers of chunks.
            custom_hdbscan = hdbscan.HDBSCAN(min_cluster_size=min(5, len(chunks)), min_samples=1)
            sub_topic_model = BERTopic(umap_model=custom_umap, hdbscan_model=custom_hdbscan, verbose=False)
            refined_topics, _ = sub_topic_model.fit_transform(chunks)
        except Exception as e:
            print(f"    Sub-topic refinement failed: {e}")
            refined_topics = original_topics
    # Bundle segmentation results.
    segmented = []
    for chunk, orig_topic, ref_topic in zip(chunks, original_topics, refined_topics):
        segmented.append({
            "chunk": chunk,
            "original_topic": int(orig_topic),
            "refined_topic": int(ref_topic)
        })
    return segmented


### 3. Hierarchical (Multiple Layers) Topic Modeling

def hierarchical_topic_modeling(df, global_topic_model):
    """
    For each full-article topic group, aggregate the refined segments from articles in that group
    and run a secondary BERTopic model to capture subthemes.
    Adds a new column 'hierarchical_subtopic' to the DataFrame.
    """
    hierarchical_results = {}
    # Group articles by full article topic
    topic_groups = df.groupby("topic")
    for topic, group in topic_groups:
        print(f"\nRunning hierarchical modeling for full topic {topic} ({len(group)} articles)...")
        # Aggregate all refined segments from articles in the group.
        aggregated_chunks = []
        article_indices = []
        for idx, row in group.iterrows():
            # Each article's segmented topics is stored as a JSON string.
            segments = json.loads(row["segmented_topics"])
            # Here, we use the refined_topic for each segment.
            for seg in segments:
                aggregated_chunks.append(seg["chunk"])
                article_indices.append(idx)
        if len(aggregated_chunks) < 3:
            print("  Not enough segments to run hierarchical modeling.")
            continue
        try:
            sub_topic_model = BERTopic(verbose=False)
            sub_topics, _ = sub_topic_model.fit_transform(aggregated_chunks)
            # Save sub-topic labels for each article (aggregate by majority vote or list all)
            for idx, sub_topic in zip(article_indices, sub_topics):
                hierarchical_results.setdefault(idx, []).append(int(sub_topic))
        except Exception as e:
            print(f"  Hierarchical modeling failed for topic {topic}: {e}")
            continue

    # For each article, decide on a final hierarchical subtopic.
    # Here we simply take the most frequent subtopic among its segments.
    hierarchical_subtopics = []
    for idx in df.index:
        subtopics = hierarchical_results.get(idx, [])
        if subtopics:
            final_subtopic = max(set(subtopics), key=subtopics.count)
        else:
            final_subtopic = -1  # or a designated value for "not modeled"
        hierarchical_subtopics.append(final_subtopic)
    df["hierarchical_subtopic"] = hierarchical_subtopics
    return df


### 4. Main Pipeline

def main():
    # (A) Load articles
    df = load_articles("C:/Users/griff/OneDrive/Desktop/Misinformation Project/Consensus/Datasets/text_processed_results.csv")
    print(f"Loaded {len(df)} articles for topic modeling.")

    # (B) Run BERTopic on full article texts
    docs = df["scraped_text"].tolist()
    global_topic_model, topics, probs = run_full_article_topic_modeling(docs)
    df["topic"] = topics

    # (C) Generate human-readable topic labels
    topic_mapping = generate_topic_mapping(global_topic_model)
    df["topic_label"] = df["topic"].apply(lambda x: topic_mapping.get(x, "Other"))

    # Optionally, save the full topic model for later use.
    global_topic_model.save("bertopic_full_model")

    # (D) For each article, perform segmentation and per-article refinement.
    segmented_results = []
    print("\nSegmenting and refining topics within each article...")
    for idx, article_text in enumerate(df["scraped_text"]):
        chunks = segment_article(article_text, chunk_size=5)
        refined_segments = refine_segments(chunks, global_topic_model)
        segmented_results.append(json.dumps(refined_segments))
        if (idx + 1) % 100 == 0:
            print(f"  Processed segmentation for {idx + 1} articles...")
    df["segmented_topics"] = segmented_results

    # (E) Hierarchical Topic Modeling:
    # Group articles by full topic and run a secondary model on aggregated refined segments.
    df = hierarchical_topic_modeling(df, global_topic_model)

    # (F) Save the results
    output_file = "C:/Users/griff/OneDrive/Desktop/Misinformation Project/Consensus/Datasets/topic_modeling_results.csv"
    df.sort_values(by="source", inplace=True)  # sort by source to group similar publishers
    df.to_csv(output_file, index=False, encoding="utf-8")
    print(f"\nHierarchical topic modeling complete. Results saved to '{output_file}'.")
    print("\nFull Topic Summary:")
    print(global_topic_model.get_topic_info())

if __name__ == "__main__":
    main()
