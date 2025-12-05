"""
Module for clustering articles.
"""

import logging
from typing import List, Dict
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter

from src.models import Article

logger = logging.getLogger(__name__)


class Clusterer:
    """Article clustering using various algorithms."""

    def __init__(self):
        self.embeddings = None
        self.labels = None
        self.model = None

    def create_tfidf_embeddings(self, articles: List[Article],
                                 max_features: int = 100) -> np.ndarray:
        """
        Create TF-IDF embeddings for articles.

        Args:
            articles: List of articles with cleaned_content
            max_features: Maximum number of features

        Returns:
            TF-IDF matrix (n_articles x n_features)
        """
        texts = []
        for article in articles:
            if article.cleaned_content:
                texts.append(article.cleaned_content)

        if not texts:
            raise ValueError("No cleaned content found in articles")

        logger.info(f"Creating TF-IDF embeddings for {len(texts)} articles...")

        vectorizer = TfidfVectorizer(
            max_features=max_features,
            min_df=1,  # More lenient for small datasets
            max_df=0.9
        )

        embeddings = vectorizer.fit_transform(texts).toarray()

        logger.info(f"Embeddings shape: {embeddings.shape}")

        self.embeddings = embeddings
        return embeddings

    def perform_kmeans(self, embeddings: np.ndarray = None,
                       n_clusters: int = 5,
                       random_state: int = 42) -> KMeans:
        """
        Perform K-Means clustering.

        Args:
            embeddings: Document embeddings (if None, uses self.embeddings)
            n_clusters: Number of clusters
            random_state: Random seed

        Returns:
            Fitted KMeans model
        """
        if embeddings is None:
            if self.embeddings is None:
                raise ValueError("No embeddings provided or created")
            embeddings = self.embeddings

        logger.info(f"Performing K-Means clustering (k={n_clusters})...")

        model = KMeans(
            n_clusters=n_clusters,
            random_state=random_state,
            n_init=10
        )

        labels = model.fit_predict(embeddings)

        logger.info(f"Clustering complete: {len(set(labels))} clusters found")

        self.model = model
        self.labels = labels
        return model

    def perform_hierarchical(self, embeddings: np.ndarray = None,
                             n_clusters: int = 5) -> AgglomerativeClustering:
        """
        Perform hierarchical clustering.

        Args:
            embeddings: Document embeddings (if None, uses self.embeddings)
            n_clusters: Number of clusters

        Returns:
            Fitted AgglomerativeClustering model
        """
        if embeddings is None:
            if self.embeddings is None:
                raise ValueError("No embeddings provided or created")
            embeddings = self.embeddings

        logger.info(f"Performing hierarchical clustering (k={n_clusters})...")

        model = AgglomerativeClustering(n_clusters=n_clusters)
        labels = model.fit_predict(embeddings)

        logger.info(f"Clustering complete: {len(set(labels))} clusters found")

        self.model = model
        self.labels = labels
        return model

    def perform_dbscan(self, embeddings: np.ndarray = None,
                       eps: float = 0.5,
                       min_samples: int = 2) -> DBSCAN:
        """
        Perform DBSCAN clustering (density-based).

        Args:
            embeddings: Document embeddings (if None, uses self.embeddings)
            eps: Maximum distance between samples
            min_samples: Minimum samples in a neighborhood

        Returns:
            Fitted DBSCAN model
        """
        if embeddings is None:
            if self.embeddings is None:
                raise ValueError("No embeddings provided or created")
            embeddings = self.embeddings

        logger.info(f"Performing DBSCAN clustering (eps={eps}, min_samples={min_samples})...")

        model = DBSCAN(eps=eps, min_samples=min_samples)
        labels = model.fit_predict(embeddings)

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)

        logger.info(f"Clustering complete: {n_clusters} clusters found, {n_noise} noise points")

        self.model = model
        self.labels = labels
        return model

    def assign_clusters(self, articles: List[Article], labels: np.ndarray = None) -> None:
        """
        Assign cluster IDs to articles (modifies articles in-place).

        Args:
            articles: List of articles
            labels: Cluster labels (if None, uses self.labels)
        """
        if labels is None:
            if self.labels is None:
                raise ValueError("No labels provided or computed")
            labels = self.labels

        # Filter articles with cleaned content
        articles_with_content = [a for a in articles if a.cleaned_content]

        if len(articles_with_content) != len(labels):
            logger.warning(f"Mismatch: {len(articles_with_content)} articles with content, "
                           f"but {len(labels)} labels")
            return

        logger.info(f"Assigning cluster IDs to {len(articles_with_content)} articles...")

        for article, label in zip(articles_with_content, labels):
            article.cluster_id = int(label)

        logger.info("Cluster assignment complete")

    def analyze_clusters(self, articles: List[Article], labels: np.ndarray = None) -> Dict:
        """
        Analyze cluster characteristics.

        Args:
            articles: List of articles with cluster_id assigned
            labels: Cluster labels (if None, uses self.labels)

        Returns:
            Dictionary with cluster statistics
        """
        if labels is None:
            if self.labels is None:
                # Try to extract from articles
                labels = np.array([a.cluster_id for a in articles if a.cluster_id is not None])
            else:
                labels = self.labels

        if labels is None or len(labels) == 0:
            logger.warning("No cluster labels found")
            return {}

        logger.info("Analyzing clusters...")

        cluster_stats = {}
        unique_clusters = set(labels)

        for cluster_id in unique_clusters:
            # Get articles in this cluster
            cluster_articles = [
                a for a, label in zip(articles, labels)
                if label == cluster_id
            ]

            if not cluster_articles:
                continue

            # Collect statistics
            cluster_size = len(cluster_articles)

            # Most common words in this cluster
            all_words = []
            for article in cluster_articles:
                if article.cleaned_content:
                    all_words.extend(article.cleaned_content.split())

            word_counts = Counter(all_words)
            top_words = [word for word, count in word_counts.most_common(10)]

            # Dominant topic (if topics assigned)
            topics = []
            for article in cluster_articles:
                if article.topics:
                    # Get dominant topic (highest probability)
                    dominant = max(article.topics, key=lambda x: x[1])[0]
                    topics.append(dominant)

            if topics:
                topic_counts = Counter(topics)
                dominant_topic = topic_counts.most_common(1)[0][0]
            else:
                dominant_topic = None

            # Average sentiment (if assigned)
            sentiments = []
            for article in cluster_articles:
                if article.sentiment and 'label' in article.sentiment:
                    sentiments.append(article.sentiment['label'])

            if sentiments:
                sentiment_counts = Counter(sentiments)
                dominant_sentiment = sentiment_counts.most_common(1)[0][0]
            else:
                dominant_sentiment = None

            cluster_stats[int(cluster_id)] = {
                'size': cluster_size,
                'top_words': top_words,
                'dominant_topic': dominant_topic,
                'dominant_sentiment': dominant_sentiment,
                'sample_titles': [a.title for a in cluster_articles[:3]]
            }

        logger.info(f"Cluster analysis complete: {len(cluster_stats)} clusters")
        return cluster_stats

    def cluster(self, articles: List[Article], n_clusters: int = 5,
                method: str = 'kmeans') -> Dict:
        """
        Complete clustering pipeline.

        Args:
            articles: List of articles with cleaned_content
            n_clusters: Number of clusters
            method: Clustering method ('kmeans', 'hierarchical', 'dbscan')

        Returns:
            Dictionary with clustering results
        """
        logger.info(f"Starting clustering pipeline (method={method}, k={n_clusters})...")

        # Create embeddings
        self.create_tfidf_embeddings(articles)

        # Perform clustering
        if method == 'kmeans':
            self.perform_kmeans(n_clusters=n_clusters)
        elif method == 'hierarchical':
            self.perform_hierarchical(n_clusters=n_clusters)
        elif method == 'dbscan':
            # For DBSCAN, we don't use n_clusters
            self.perform_dbscan()
        else:
            raise ValueError(f"Unknown clustering method: {method}")

        # Assign clusters to articles
        self.assign_clusters(articles)

        # Analyze clusters
        cluster_stats = self.analyze_clusters(articles)

        results = {
            'method': method,
            'n_clusters': len(cluster_stats),
            'cluster_stats': cluster_stats
        }

        logger.info("Clustering pipeline complete")
        return results
