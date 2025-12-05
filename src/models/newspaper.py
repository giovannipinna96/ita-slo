"""
Data model for newspaper collection.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict
import numpy as np

from .article import Article


@dataclass
class Newspaper:
    """Represents a complete newspaper with all articles."""

    # Core attributes
    name: str  # 'Il Piccolo' or 'Edinost'
    language: str  # 'it' or 'sl'
    date: str  # 'YYYY-MM-DD' format
    articles: List[Article] = field(default_factory=list)

    # Analysis results (populated during analysis pipeline)
    topic_distribution: Optional[np.ndarray] = None  # Distribution of topics across all articles
    overall_sentiment: Optional[Dict] = None  # Aggregated sentiment statistics
    vocabulary_size: Optional[int] = None  # Number of unique words

    def __post_init__(self):
        """Validate newspaper data."""
        if not self.name or not self.name.strip():
            raise ValueError("Newspaper name cannot be empty")
        if self.language not in ['it', 'sl']:
            raise ValueError(f"Invalid language: {self.language}")

    def __len__(self) -> int:
        """Return the number of articles in the newspaper."""
        return len(self.articles)

    def add_article(self, article: Article) -> None:
        """Add an article to the newspaper."""
        if article.language != self.language:
            raise ValueError(
                f"Article language ({article.language}) doesn't match newspaper language ({self.language})"
            )
        self.articles.append(article)

    def get_articles_by_section(self, section: str) -> List[Article]:
        """Get all articles from a specific section."""
        return [a for a in self.articles if a.section == section]

    def get_articles_by_topic(self, topic_id: int) -> List[Article]:
        """Get all articles where the given topic is dominant."""
        result = []
        for article in self.articles:
            if article.topics:
                # Get dominant topic (highest probability)
                dominant_topic = max(article.topics, key=lambda x: x[1])[0]
                if dominant_topic == topic_id:
                    result.append(article)
        return result

    def get_total_words(self) -> int:
        """Get total word count across all articles."""
        total = 0
        for article in self.articles:
            content = article.cleaned_content if article.cleaned_content else article.content
            total += len(content.split())
        return total

    def to_dict(self) -> Dict:
        """Convert newspaper to dictionary for JSON serialization."""
        return {
            'name': self.name,
            'language': self.language,
            'date': self.date,
            'articles': [a.to_dict() for a in self.articles],
            'topic_distribution': self.topic_distribution.tolist() if self.topic_distribution is not None else None,
            'overall_sentiment': self.overall_sentiment,
            'vocabulary_size': self.vocabulary_size
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'Newspaper':
        """Create Newspaper from dictionary."""
        articles = [Article.from_dict(a) for a in data.get('articles', [])]
        topic_dist = data.get('topic_distribution')
        if topic_dist is not None:
            topic_dist = np.array(topic_dist)

        return cls(
            name=data['name'],
            language=data['language'],
            date=data['date'],
            articles=articles,
            topic_distribution=topic_dist,
            overall_sentiment=data.get('overall_sentiment'),
            vocabulary_size=data.get('vocabulary_size')
        )
