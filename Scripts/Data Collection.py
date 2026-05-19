#!/usr/bin/env python3
"""
Combined Async Pipeline:
  1. Load an existing dataset (to avoid duplicate URLs).
  2. Asynchronously fetch new articles from RSS feeds and NewsAPI.
  3. Asynchronously scrape full-text for each new article.
  4. Append new articles to the dataset, sort them by source, and save.

Each article’s source is used to index into bias_ratings and factuality_ratings.
"""

import os
import time
import random
import re
import json
import asyncio
import aiohttp
import feedparser
import pandas as pd
from bs4 import BeautifulSoup
from newspaper import Article, Config
from functools import partial
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ----------------------------
# Configuration
# ----------------------------
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")
NEWSAPI_QUERY = "politics OR world"
NEWSAPI_STATE_FILE = "newsapi_state.txt"
NEWSAPI_PAGES_PER_RUN = 2  # How many pages to fetch per iteration.
NEWSAPI_PAGE_SIZE = 20
NUM_NEWSAPI_ITERATIONS = 20
NEWSAPI_MAX_RESULTS = 100  # Developer accounts are limited to 100 total results.
NEWSAPI_MAX_PAGE = NEWSAPI_MAX_RESULTS // NEWSAPI_PAGE_SIZE  # For page_size=20, max_page = 5.

# ----------------------------
# Source Ratings Dictionaries
# ----------------------------
bias_ratings = {
    "CNN": "-3.6",
    "BBC": "-2.0",
    "Reuters": "-0.5",
    "The Guardian": "-3.6",
    "Al Jazeera": "-3.2",
    "Washington Post": "-3.6",
    "ABC": "-3.3",
    "NBC": "-3.6",
    "CBS News": "-3.3",
    "NPR": "-2.8",
    "PBS NewsHour": "-2.4",
    "C-SPAN": "0.0",
    "Yahoo News": "-3.3",
    "AP News": "-2.1",
    "VOA News (Voice of America)": "-0.8",
    "DW News (Deutsche Welle)": "-2.5",
    "SCMP (South China Morning Post)": "-2.9",
    "The Straits Times": "2.9",
    "The Diplomat": "-0.4",
    "France 24": "-1.75",
    "RFI (Radio France Internationale)": "0.5",
    "Japan Times": "0.3",
    "The Sydney Morning Herald": "-2.3",
    "Times of India": "2.9",
    "The Moscow Times": "-3.7",
    "The Conversation": "-0.9",
    "Business Insider": "-3.3",
    "Foreign Policy": "-5.5",
    "The New Statesman": "-6.7",
    "Bangkok Post": "3.2",
    "The American Conservative": "3.4",
    "RedState": "7.1",
    "Newsmax": "7.8",
    "Breitbart": "8.1",
}

factuality_ratings = {
    "CNN": "3.7",
    "BBC": "0.8",
    "Reuters": "0.0",
    "The Guardian": "5.9",
    "Al Jazeera": "4.7",
    "Washington Post": "2.1",
    "ABC": "1.1",
    "NBC": "1.0",
    "CBS News": "1.0",
    "NPR": "0.9",
    "PBS NewsHour": "1.0",
    "C-SPAN": "0.0",
    "Yahoo News": "1.9",
    "AP News": "0.8",
    "VOA News (Voice of America)": "1.0",
    "DW News (Deutsche Welle)": "0.9",
    "SCMP (South China Morning Post)": "4.8",
    "The Straits Times": "3.4",
    "The Diplomat": "1.1",
    "France 24": "1.0",
    "RFI (Radio France Internationale)": "3.4",
    "Japan Times": "0.8",
    "The Sydney Morning Herald": "1.0",
    "Times of India": "5.1",
    "The Moscow Times": "1.6",
    "The Conversation": "0.0",
    "Business Insider": "2.3",
    "Foreign Policy": "6",
    "The New Statesman": "1.5",
    "Bangkok Post": "5.0",
    "The American Conservative": "2.4",
    "RedState": "4.7",
    "Newsmax": "7.6",
    "Breitbart": "6.4",
}

# ----------------------------
# Full RSS Feeds Dictionary
# ----------------------------
rss_feeds = {
    "CNN": [
        "https://rss.cnn.com/rss/cnn_allpolitics.rss",
        "https://rss.cnn.com/rss/edition_world.rss"
    ],
    "BBC": [
        "https://feeds.bbci.co.uk/news/politics/rss.xml",
        "https://feeds.bbci.co.uk/news/world/rss.xml"
    ],
    "Reuters": [
        "https://www.reutersagency.com/feed/?best-topics=politics",
        "https://www.reutersagency.com/feed/?best-regions=world"
    ],
    "The Guardian": [
        "https://www.theguardian.com/politics/rss",
        "https://www.theguardian.com/world/rss"
    ],
    "Al Jazeera": [
        "https://www.aljazeera.com/xml/rss/all.xml"
    ],
    "Washington Post": [
        "http://feeds.washingtonpost.com/rss/politics",
        "http://feeds.washingtonpost.com/rss/world"
    ],
    "ABC": [
        "https://abcnews.go.com/abcnews/politicsheadlines",
        "https://abcnews.go.com/abcnews/internationalheadlines"
    ],
    "NBC": [
        "https://feeds.nbcnews.com/nbcnews/public/politics",
        "https://feeds.nbcnews.com/nbcnews/public/world"
    ],
    "CBS News": [
        "https://www.cbsnews.com/latest/rss/politics",
        "https://www.cbsnews.com/latest/rss/world"
    ],
    "NPR": [
        "https://feeds.npr.org/1003/rss.xml",
        "https://feeds.npr.org/1004/rss.xml"
    ],
    "PBS NewsHour": [
        "https://www.pbs.org/newshour/feeds/rss/headlines"
    ],
    "C-SPAN": [
        "https://www.c-span.org/rss/news.asp?type=Latest",
        "https://www.c-span.org/rss/news.asp?type=Politics"
    ],
    "Yahoo News": [
        "https://www.yahoo.com/news/rss/politics",
        "https://www.yahoo.com/news/rss/world"
    ],
    "AP News": [
        "https://rss.apnews.com/apf-politics",
        "https://rss.apnews.com/apf-international"
    ],
    "VOA News (Voice of America)": [
        "https://www.voanews.com/usa/rss",
        "https://www.voanews.com/rss"
    ],
    "DW News (Deutsche Welle)": [
        "https://rss.dw.com/rdf/rss-en-top",
        "https://rss.dw.com/rdf/rss-en-world"
    ],
    "SCMP (South China Morning Post)": [
        "https://www.scmp.com/rss/91/feed",
        "https://www.scmp.com/rss/318208/feed"
    ],
    "The Straits Times": [
        "https://www.straitstimes.com/news/singapore/rss.xml",
        "https://www.straitstimes.com/news/asia/rss.xml"
    ],
    "The Diplomat": [
        "https://thediplomat.com/feed/",
        "https://thediplomat.com/feed/world"
    ],
    "France 24": [
        "https://www.france24.com/en/rss",
        "https://www.france24.com/en/france/rss",
        "https://www.france24.com/en/europe/rss",
        "https://www.france24.com/en/africa/rss",
        "https://www.france24.com/en/asia-pacific/rss"
    ],
    "RFI (Radio France Internationale)": [
        "https://www.rfi.fr/en/podcasts/news/rss",
        "https://www.rfi.fr/en/podcasts/world-news/rss"
    ],
    "Japan Times": [
        "https://www.japantimes.co.jp/feed/",
        "https://www.japantimes.co.jp/news_category/politics/feed/",
        "https://www.japantimes.co.jp/news_category/asia-pacific/feed/"
    ],
    "The Sydney Morning Herald": [
        "https://www.smh.com.au/rssheadlines/politics.xml",
        "https://www.smh.com.au/rssheadlines/world.xml"
    ],
    "Times of India": [
        "https://timesofindia.indiatimes.com/rssfeeds/-2128936835.cms",
        "https://timesofindia.indiatimes.com/rssfeeds/296589292.cms"
    ],
    "The Moscow Times": [
        "https://www.themoscowtimes.com/rss/news",
        "https://www.themoscowtimes.com/rss/politics"
    ],
    "The Conversation": [
        "https://theconversation.com/us/topics/politics-11/articles.atom",
        "https://theconversation.com/us/topics/international-11/articles.atom"
    ],
    "Business Insider": [
        "https://www.businessinsider.com/sai.xml",
        "https://www.businessinsider.com/politics/rss"
    ],
    "Foreign Policy": [
        "https://foreignpolicy.com/feed/",
        "https://foreignpolicy.com/category/analysis/feed/"
    ],
    "The New Statesman": [
        "https://www.newstatesman.com/feeds/rss",
        "https://www.newstatesman.com/politics/feed"
    ],
    "Bangkok Post": [
        "https://www.bangkokpost.com/rss/data/news.xml",
        "https://www.bangkokpost.com/rss/data/world.xml"
    ],
    "The American Conservative": [
        "https://www.theamericanconservative.com/feed/"
    ],
    "RedState": [
        "https://redstate.com/feed/"
    ],
    "Newsmax": [
        "https://www.newsmax.com/rss/Politics/1.xml",
        "https://www.newsmax.com/rss/World/1.xml"
    ],
    "Breitbart": [
        "http://feeds.feedburner.com/breitbart"
    ]
}

# ----------------------------
# Helper Function Definitions
# ----------------------------

def clean_summary(text):
    """
    Remove HTML tags, URLs, and extra whitespace from text.
    """
    if not text:
        return "N/A"
    soup = BeautifulSoup(text, "html.parser")
    clean_text = soup.get_text(separator=" ")
    clean_text = re.sub(r'http[s]?://\S+', '', clean_text)
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()
    return clean_text

# ----------------------------
# Asynchronous Functions for Data Fetching
# ----------------------------
async def fetch_url(session, url):
    """
    Fetch the content of a URL using aiohttp.
    """
    try:
        async with session.get(url, timeout=10) as response:
            return await response.text()
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return None


async def fetch_rss_articles(known_urls, session):
    """
    Loop through the RSS feeds asynchronously and return a list of new articles.
    """
    articles = []
    tasks = []
    for source, urls in rss_feeds.items():
        print(f"Processing RSS source: {source}")
        for feed_url in urls:
            tasks.append(asyncio.create_task(fetch_url(session, feed_url)))
    responses = await asyncio.gather(*tasks)
    # Process each feed response.
    idx = 0
    for source, urls in rss_feeds.items():
        for feed_url in urls:
            content = responses[idx]
            idx += 1
            if not content:
                continue
            feed = feedparser.parse(content)
            for entry in feed.entries:
                article_url = entry.get("link", "N/A")
                if article_url in known_urls:
                    continue
                bias = bias_ratings.get(source, "")
                factuality = factuality_ratings.get(source, "")
                article_data = {
                    "source": source,
                    "title": entry.get("title", "N/A"),
                    "publication_date": entry.get("published", "N/A"),
                    "url": article_url,
                    "summary": clean_summary(entry.get("summary", "N/A")),
                    "scraped_text": "",
                    "bias_rating": bias,
                    "factuality_rating": factuality
                }
                articles.append(article_data)
    return articles


async def fetch_newsapi_page(session, api_key, query, page, known_urls, page_size=NEWSAPI_PAGE_SIZE):
    """
    Asynchronously fetch one page from NewsAPI.
    """
    base_url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": page_size,
        "page": page,
        "apiKey": api_key
    }
    try:
        async with session.get(base_url, params=params, timeout=10) as response:
            if response.status != 200:
                text = await response.text()
                print(f"NewsAPI request failed (status {response.status}): {text}")
                return []
            data = await response.json()
    except Exception as e:
        print(f"Error fetching NewsAPI page {page}: {e}")
        return []

    articles = []
    for item in data.get("articles", []):
        article_url = item.get("url", "N/A")
        if article_url in known_urls:
            continue
        source_name = item.get("source", {}).get("name", "NewsAPI")
        bias = bias_ratings.get(source_name, "")
        factuality = factuality_ratings.get(source_name, "")
        article_data = {
            "source": source_name,
            "title": item.get("title", "N/A"),
            "publication_date": item.get("publishedAt", "N/A"),
            "url": article_url,
            "summary": clean_summary(item.get("description", "N/A")),
            "scraped_text": "",
            "bias_rating": bias,
            "factuality_rating": factuality
        }
        articles.append(article_data)
    return articles


async def fetch_newsapi_articles(session, api_key, query, start_page, pages_to_fetch, known_urls):
    """
    Asynchronously fetch articles via NewsAPI for the given pages.
    Returns a tuple: (list of articles, next_page)
    """
    articles = []
    tasks = []
    for page in range(start_page, start_page + pages_to_fetch):
        if page > NEWSAPI_MAX_PAGE:
            break
        print(f"Fetching NewsAPI page {page}")
        tasks.append(asyncio.create_task(fetch_newsapi_page(session, api_key, query, page, known_urls)))
    pages_results = await asyncio.gather(*tasks)
    for result in pages_results:
        articles.extend(result)
    next_page = start_page + len(tasks)
    if next_page > NEWSAPI_MAX_PAGE:
        next_page = 1
    return articles, next_page


def save_newsapi_state(next_page, state_file=NEWSAPI_STATE_FILE):
    with open(state_file, "w") as f:
        f.write(str(next_page))


def load_newsapi_state(state_file=NEWSAPI_STATE_FILE):
    if os.path.exists(state_file):
        try:
            with open(state_file, "r") as f:
                page = int(f.read().strip())
                return page
        except Exception as e:
            print(f"Error reading state file: {e}")
    return 1


async def async_scrape_article(url, executor):
    """
    Asynchronously scrape the full article text using newspaper3k.
    Since newspaper3k is synchronous, run it in an executor.
    """
    loop = asyncio.get_event_loop()
    try:
        result = await asyncio.wait_for(
            loop.run_in_executor(executor, partial(scrape_article, url)),
            timeout=25.0
        )
        return result
    except asyncio.TimeoutError:
        return "Failed to extract: Timeout via asyncio"
    except Exception as e:
        return f"Failed to extract: {e}"


def scrape_article(url):
    """
    Synchronously scrape an article using newspaper3k.
    """
    try:
        config = Config()
        config.request_timeout = 15
        article = Article(url, config=config)
        article.download()
        article.parse()
        return article.text if article.text else "No relevant content found"
    except Exception as e:
        return f"Failed to extract: {e}"


async def main_async():
    # Load or initialize the combined dataset.
    output_csv = "Datasets/articles.csv"
    if os.path.exists(output_csv):
        df_existing = pd.read_csv(output_csv)
        print(f"Loaded existing dataset with {len(df_existing)} entries.")
    else:
        df_existing = pd.DataFrame(
            columns=["source", "title", "publication_date", "url", "summary", "scraped_text", "bias_rating",
                     "factuality_rating"])
        print("No existing dataset found. Starting fresh.")

    known_urls = set(df_existing["url"].dropna().tolist())

    async with aiohttp.ClientSession() as session:
        # 1. Asynchronously fetch articles from RSS feeds.
        rss_articles = await fetch_rss_articles(known_urls, session)
        print(f"Fetched {len(rss_articles)} new articles from RSS feeds.")
        for article in rss_articles:
            known_urls.add(article["url"])

        # 2. Asynchronously fetch articles from NewsAPI.
        start_page = load_newsapi_state()
        all_newsapi_articles = []
        for iteration in range(NUM_NEWSAPI_ITERATIONS):
            print(f"\nNewsAPI iteration {iteration + 1}/{NUM_NEWSAPI_ITERATIONS}, starting at page {start_page}.")
            newsapi_articles, next_page = await fetch_newsapi_articles(session, NEWSAPI_KEY, NEWSAPI_QUERY, start_page, NEWSAPI_PAGES_PER_RUN, known_urls)
            print(f"Fetched {len(newsapi_articles)} new articles from NewsAPI in iteration {iteration + 1}.")
            all_newsapi_articles.extend(newsapi_articles)
            for art in newsapi_articles:
                known_urls.add(art["url"])
            start_page = next_page
            if not newsapi_articles:
                print("No articles returned in this iteration; stopping further API calls.")
                break
            if start_page == 1:
                print("Reached state reset (start_page=1); ending iterations.")
                break
            await asyncio.sleep(1)
        save_newsapi_state(start_page)
        print(f"\nNext NewsAPI fetch will start at page {start_page}.")

        # Combine new articles.
        new_articles = rss_articles + all_newsapi_articles
        print(f"\nTotal new articles to process: {len(new_articles)}")

        # 3. Asynchronously scrape full text for each new article.
        executor = ThreadPoolExecutor(max_workers=10)
        scrape_tasks = []
        for idx, article in enumerate(new_articles):
            url = article["url"]
            if isinstance(url, str) and url.startswith("http"):
                print(f"Scraping article {idx + 1}/{len(new_articles)}: {url}")
                scrape_tasks.append(async_scrape_article(url, executor))
            else:
                scrape_tasks.append(asyncio.sleep(0, result="Invalid URL"))
            await asyncio.sleep(random.uniform(0, 1))  # slight delay between scheduling tasks

        scraped_texts = await asyncio.gather(*scrape_tasks)
        for article, text in zip(new_articles, scraped_texts):
            article["scraped_text"] = text

    # 4. Append new articles to the existing dataset.
    if new_articles:
        df_new = pd.DataFrame(new_articles)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_combined = df_existing

    df_combined.sort_values(by="source", inplace=True)
    df_combined.to_csv(output_csv, index=False, encoding="utf-8")
    print(f"\n✅ Combined data saved to '{output_csv}'. Total articles: {len(df_combined)}")


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
