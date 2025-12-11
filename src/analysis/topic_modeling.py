"""
Module for topic modeling using LDA and BERTopic.
"""

import logging
from typing import List, Dict, Tuple
import numpy as np
from gensim import corpora
from gensim.models import LdaModel, CoherenceModel
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer

from src.models import Article

logger = logging.getLogger(__name__)


class TopicModeler:
    """Topic modeling with LDA and BERTopic."""

    def __init__(self):
        self.lda_model = None
        self.dictionary = None
        self.corpus = None
        self.bertopic_model = None

    # ===== LDA Methods =====

    def create_dictionary(self, articles: List[Article]) -> corpora.Dictionary:
        """
        Create Gensim dictionary from cleaned articles.

        Args:
            articles: List of articles with cleaned_content

        Returns:
            Gensim Dictionary object
        """
        # Extract tokenized texts
        texts = []
        for article in articles:
            if article.cleaned_content:
                tokens = article.cleaned_content.split()
                texts.append(tokens)

        if not texts:
            raise ValueError("No cleaned content found in articles")

        logger.info(f"Creating dictionary from {len(texts)} documents...")

        # Create dictionary
        dictionary = corpora.Dictionary(texts)

        # Filter extremes
        # Remove words that appear in less than 2 documents or more than 80% of documents
        dictionary.filter_extremes(no_below=2, no_above=0.8)

        logger.info(f"Dictionary created: {len(dictionary)} unique tokens")

        self.dictionary = dictionary
        return dictionary

    def create_corpus(self, articles: List[Article], dictionary: corpora.Dictionary = None) -> List:
        """
        Create Gensim corpus from articles.

        Args:
            articles: List of articles with cleaned_content
            dictionary: Gensim dictionary (if None, uses self.dictionary)

        Returns:
            Gensim corpus (bag-of-words representation)
        """
        if dictionary is None:
            if self.dictionary is None:
                raise ValueError("No dictionary provided or created")
            dictionary = self.dictionary

        # Extract tokenized texts
        texts = []
        for article in articles:
            if article.cleaned_content:
                tokens = article.cleaned_content.split()
                texts.append(tokens)

        logger.info(f"Creating corpus from {len(texts)} documents...")

        # Create corpus (bag-of-words)
        corpus = [dictionary.doc2bow(text) for text in texts]

        self.corpus = corpus
        return corpus

    def train_lda(self, articles: List[Article], num_topics: int = 8,
                  passes: int = 15, random_state: int = 42) -> LdaModel:
        """
        Train LDA topic model.

        Args:
            articles: List of articles with cleaned_content
            num_topics: Number of topics to extract
            passes: Number of passes through the corpus
            random_state: Random seed for reproducibility

        Returns:
            Trained LDA model
        """
        logger.info(f"Training LDA model with {num_topics} topics...")

        # Create dictionary and corpus if not already done
        if self.dictionary is None:
            self.create_dictionary(articles)

        if self.corpus is None:
            self.create_corpus(articles)

        # Train LDA model
        self.lda_model = LdaModel(
            corpus=self.corpus,
            id2word=self.dictionary,
            num_topics=num_topics,
            passes=passes,
            alpha='auto',  # Learn asymmetric prior
            eta='auto',  # Learn asymmetric prior
            random_state=random_state,
            per_word_topics=True
        )

        logger.info("LDA training complete")

        # Log topics
        for idx, topic in self.lda_model.print_topics(num_words=5):
            logger.info(f"Topic {idx}: {topic}")

        return self.lda_model

    def get_topic_keywords(self, model: LdaModel = None, num_words: int = 10) -> Dict[int, List[str]]:
        """
        Extract top keywords for each topic.

        Args:
            model: LDA model (if None, uses self.lda_model)
            num_words: Number of keywords per topic

        Returns:
            Dictionary mapping topic_id to list of keywords
        """
        if model is None:
            if self.lda_model is None:
                raise ValueError("No LDA model provided or trained")
            model = self.lda_model

        topics = {}
        for topic_id in range(model.num_topics):
            # Get top words for this topic
            topic_words = model.show_topic(topic_id, topn=num_words)
            # Extract just the words (ignore probabilities)
            keywords = [word for word, prob in topic_words]
            topics[topic_id] = keywords

        return topics

    def assign_topics_to_articles(self, articles: List[Article],
                                   model: LdaModel = None,
                                   corpus: List = None) -> None:
        """
        Assign topic distributions to articles (modifies articles in-place).

        Args:
            articles: List of articles
            model: LDA model (if None, uses self.lda_model)
            corpus: Corpus (if None, uses self.corpus)
        """
        if model is None:
            if self.lda_model is None:
                raise ValueError("No LDA model provided or trained")
            model = self.lda_model

        if corpus is None:
            if self.corpus is None:
                raise ValueError("No corpus provided or created")
            corpus = self.corpus

        logger.info(f"Assigning topics to {len(articles)} articles...")

        # Filter articles with cleaned content
        articles_with_content = [a for a in articles if a.cleaned_content]

        if len(articles_with_content) != len(corpus):
            logger.warning(f"Mismatch: {len(articles_with_content)} articles with content, "
                           f"but {len(corpus)} documents in corpus")

        # Assign topics
        for idx, (article, doc_bow) in enumerate(zip(articles_with_content, corpus)):
            # Get topic distribution for this document
            topic_dist = model.get_document_topics(doc_bow)

            # Sort by probability (descending)
            topic_dist = sorted(topic_dist, key=lambda x: x[1], reverse=True)

            # Assign to article
            article.topics = topic_dist

        logger.info("Topic assignment complete")

    def compute_coherence_score(self, articles: List[Article],
                                 model: LdaModel = None) -> float:
        """
        Compute coherence score for LDA model.

        Args:
            articles: List of articles
            model: LDA model (if None, uses self.lda_model)

        Returns:
            Coherence score (c_v metric)
        """
        if model is None:
            if self.lda_model is None:
                raise ValueError("No LDA model provided or trained")
            model = self.lda_model

        # Extract tokenized texts
        texts = []
        for article in articles:
            if article.cleaned_content:
                tokens = article.cleaned_content.split()
                texts.append(tokens)

        # Compute coherence
        coherence_model = CoherenceModel(
            model=model,
            texts=texts,
            dictionary=self.dictionary,
            coherence='c_v'
        )

        coherence_score = coherence_model.get_coherence()
        logger.info(f"Coherence score (c_v): {coherence_score:.4f}")

        return coherence_score

    # ===== BERTopic Methods =====

    def train_bertopic(self, articles: List[Article], language: str = 'multilingual',
                       num_topics: int = 5) -> BERTopic:
        """
        Train BERTopic model with a specified number of topics.
        
        Uses KMeans clustering instead of HDBSCAN for predictable topic count,
        similar to how LDA works.

        Args:
            articles: List of articles with cleaned_content
            language: Language for the model ('multilingual', 'it', 'sl')
            num_topics: Number of topics to extract (like LDA)

        Returns:
            Trained BERTopic model
        """
        logger.info(f"Training BERTopic model (language: {language}, num_topics: {num_topics})...")

        # Extract original texts (not lemmatized, BERTopic works better with original text)
        texts = []
        for article in articles:
            # Use original content, not cleaned (BERTopic handles preprocessing)
            if article.content:
                texts.append(article.content)

        if not texts:
            raise ValueError("No content found in articles")

        n_docs = len(texts)
        logger.info(f"Number of documents: {n_docs}")
        
        # Adjust num_topics if greater than number of documents
        if num_topics >= n_docs:
            num_topics = max(2, n_docs - 1)
            logger.warning(f"Reduced num_topics to {num_topics} (must be < n_docs)")

        # Choose embedding model based on language
        if language == 'sl':
            embedding_model = "EMBEDDIA/sloberta"
            logger.info("Using SloBERTa for Slovenian text")
        elif language == 'it':
            embedding_model = "dbmdz/bert-base-italian-cased"
            logger.info("Using Italian BERT for Italian text")
        else:
            embedding_model = "paraphrase-multilingual-MiniLM-L12-v2"
            logger.info("Using multilingual MiniLM for multilingual text")

        logger.info(f"Using embedding model: {embedding_model}")

        # Use KMeans for predictable number of topics (like LDA)
        from sklearn.cluster import KMeans
        from sklearn.feature_extraction.text import CountVectorizer
        
        cluster_model = KMeans(n_clusters=num_topics, random_state=42, n_init=10)
        logger.info(f"Using KMeans clustering with {num_topics} clusters")
        
        # Load stopwords from spaCy
        import spacy
        try:
            if language == 'sl':
                nlp = spacy.load("sl_core_news_lg", disable=['parser', 'ner'])
                logger.info("Loaded Slovenian stopwords from spaCy")
            else:
                nlp = spacy.load("it_core_news_lg", disable=['parser', 'ner'])
                logger.info("Loaded Italian stopwords from spaCy")
            
            stopwords = list(nlp.Defaults.stop_words)
            logger.info(f"Using {len(stopwords)} stopwords from spaCy")
        except Exception as e:
            logger.warning(f"Could not load spaCy stopwords: {e}. Using minimal list.")
            # Fallback to minimal stopwords
            if language == 'sl':
                stopwords = ['in', 'je', 'da', 'na', 'se', 'za', 'so', 'ki', 'bi', 'pa', 'ali', 'po']
            else:
                stopwords = ['di', 'a', 'da', 'in', 'con', 'il', 'la', 'che', 'e', 'è', 'un', 'per']
        
        # Create vectorizer that filters stopwords and short words
        vectorizer_model = CountVectorizer(
            stop_words=stopwords,
            min_df=1,
            ngram_range=(1, 2),  # Include bigrams for better topics
            token_pattern=r'\b[a-zA-ZàèéìòùÀÈÉÌÒÙčšžČŠŽ]{3,}\b'  # Only words with 3+ chars
        )

        # Create BERTopic model with KMeans (no UMAP needed for small datasets)
        # Setting umap_model=None lets BERTopic skip dimensionality reduction
        self.bertopic_model = BERTopic(
            embedding_model=embedding_model,
            hdbscan_model=cluster_model,  # BERTopic accepts any sklearn clusterer here
            umap_model=None,  # Skip UMAP, use embeddings directly
            vectorizer_model=vectorizer_model,  # Filter stopwords
            language='multilingual',
            calculate_probabilities=False,  # KMeans doesn't support soft clustering
            verbose=True
        )

        # Train model
        topics, probs = self.bertopic_model.fit_transform(texts)

        n_topics_found = len(set(topics))
        if -1 in set(topics):
            n_topics_found -= 1  # Don't count outlier topic
            
        logger.info("BERTopic training complete")
        logger.info(f"Number of topics found: {n_topics_found}")

        return self.bertopic_model

    def get_bertopic_info(self, model: BERTopic = None) -> dict:
        """
        Extract topic information from BERTopic model.

        Args:
            model: BERTopic model (if None, uses self.bertopic_model)

        Returns:
            Dictionary with topic information
        """
        if model is None:
            if self.bertopic_model is None:
                raise ValueError("No BERTopic model provided or trained")
            model = self.bertopic_model

        topic_info = model.get_topic_info()

        # Convert to dictionary format
        topics = {}
        for idx, row in topic_info.iterrows():
            topic_id = row['Topic']
            if topic_id == -1:  # Skip outlier topic
                continue

            # Get top words for this topic
            topic_words = model.get_topic(topic_id)
            if topic_words:
                keywords = [word for word, score in topic_words[:10]]
                topics[topic_id] = {
                    'keywords': keywords,
                    'count': row['Count'],
                    'name': row['Name'] if 'Name' in row else f"Topic_{topic_id}"
                }

        return topics

    # ===== Analysis Pipeline =====

    def analyze_lda(self, articles: List[Article], num_topics: int = 8) -> dict:
        """
        Complete LDA analysis pipeline.

        Args:
            articles: List of articles with cleaned_content
            num_topics: Number of topics

        Returns:
            Dictionary with analysis results
        """
        logger.info("Starting LDA analysis pipeline...")

        # Train model
        self.train_lda(articles, num_topics=num_topics)

        # Get topic keywords
        topics = self.get_topic_keywords(num_words=10)

        # Assign topics to articles
        self.assign_topics_to_articles(articles)

        # Compute coherence
        coherence = self.compute_coherence_score(articles)

        results = {
            'topics': topics,
            'num_topics': num_topics,
            'coherence_score': coherence,
            'n_articles': len([a for a in articles if a.cleaned_content])
        }

        logger.info("LDA analysis complete")
        return results

    def analyze_bertopic(self, articles: List[Article], language: str = 'multilingual',
                          num_topics: int = None) -> dict:
        """
        Complete BERTopic analysis pipeline.
        
        Works like analyze_lda but uses BERT embeddings for better semantic understanding.

        Args:
            articles: List of articles
            language: Language for the model
            num_topics: Number of topics (if None, calculated automatically)

        Returns:
            Dictionary with analysis results (same format as analyze_lda)
        """
        logger.info("Starting BERTopic analysis pipeline...")

        # Determine num_topics based on number of articles (same logic as LDA)
        n_articles = len([a for a in articles if a.content])
        if num_topics is None:
            num_topics = min(5, max(2, n_articles // 3))
        
        logger.info(f"Using {num_topics} topics for {n_articles} articles")

        # Train model
        self.train_bertopic(articles, language=language, num_topics=num_topics)

        # Get topic info and convert to LDA-like format
        bertopic_info = self.get_bertopic_info()
        
        # Convert to same format as LDA: {topic_id: [keyword1, keyword2, ...]}
        topics = {}
        for topic_id, info in bertopic_info.items():
            topics[topic_id] = info['keywords']

        # Get per-article topic assignments
        article_topics = []
        if self.bertopic_model is not None:
            # Get texts for transform
            texts = [a.content for a in articles if a.content]
            try:
                assigned_topics, probs = self.bertopic_model.transform(texts)
                for i, (topic_id, article) in enumerate(zip(assigned_topics, articles)):
                    topic_name = f"Topic {topic_id}"
                    # Get topic name from topic info if available
                    if str(topic_id) in bertopic_info:
                        topic_name = bertopic_info[str(topic_id)].get('name', topic_name)
                    
                    article_topics.append({
                        'article_idx': i,
                        'topic': int(topic_id),
                        'name': topic_name
                    })
            except Exception as e:
                logger.warning(f"Could not get article topic assignments: {e}")

        results = {
            'topics': topics,
            'num_topics': len(topics),
            'n_articles': n_articles,
            'method': 'bertopic',
            'article_topics': article_topics  # Per-article assignments
        }

        logger.info("BERTopic analysis complete")
        return results

