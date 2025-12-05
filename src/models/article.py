"""
Data model for newspaper articles.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict


@dataclass
class Article:
    """Represents a single newspaper article."""

    # Core attributes
    title: str
    content: str
    source: str  # 'il_piccolo' or 'edinost'
    language: str  # 'it' or 'sl'

    # Metadata
    section: Optional[str] = None
    page: Optional[int] = None

    # Preprocessed content
    cleaned_content: Optional[str] = None

    # Analysis results (populated during analysis pipeline)
    topics: Optional[List[Tuple[int, float]]] = None  # [(topic_id, probability), ...]
    sentiment: Optional[Dict] = None  # {'label': str, 'score': float}
    tfidf_keywords: Optional[List[Tuple[str, float]]] = None  # [(word, score), ...]
    cluster_id: Optional[int] = None

    def __post_init__(self):
        """Validate article data."""
        if not self.title or not self.title.strip():
            raise ValueError("Article title cannot be empty")
        if not self.content or not self.content.strip():
            raise ValueError("Article content cannot be empty")
        if self.source not in ['il_piccolo', 'edinost']:
            raise ValueError(f"Invalid source: {self.source}")
        if self.language not in ['it', 'sl']:
            raise ValueError(f"Invalid language: {self.language}")

    def __len__(self) -> int:
        """Return the length of the article content."""
        return len(self.content)

    def to_dict(self) -> Dict:
        """Convert article to dictionary for JSON serialization."""
        # Helper to convert numpy types to Python types
        def convert_value(val):
            if val is None:
                return None
            if isinstance(val, (list, tuple)):
                return [convert_value(v) for v in val]
            if isinstance(val, dict):
                return {k: convert_value(v) for k, v in val.items()}
            # Convert numpy types to Python types
            if hasattr(val, 'item'):  # numpy scalar
                return val.item()
            return val

        return {
            'title': self.title,
            'content': self.content,
            'source': self.source,
            'language': self.language,
            'section': self.section,
            'page': self.page,
            'cleaned_content': self.cleaned_content,
            'topics': convert_value(self.topics),
            'sentiment': convert_value(self.sentiment),
            'tfidf_keywords': convert_value(self.tfidf_keywords),
            'cluster_id': int(self.cluster_id) if self.cluster_id is not None else None
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'Article':
        """Create Article from dictionary."""
        return cls(**data)
