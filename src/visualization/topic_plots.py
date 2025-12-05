"""
Visualization module for topic modeling results.
"""

import logging
from pathlib import Path
from typing import List, Dict
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import numpy as np

from src.models import Article

logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


def plot_topic_distribution(articles: List[Article], topics: Dict[int, List[str]],
                             newspaper_name: str, output_path: str):
    """
    Plot distribution of articles across topics.

    Args:
        articles: List of articles with topic assignments
        topics: Dictionary of topic keywords
        newspaper_name: Name of the newspaper
        output_path: Path to save the plot
    """
    logger.info(f"Creating topic distribution plot for {newspaper_name}...")

    # Count articles per topic
    topic_counts = {}
    for article in articles:
        if article.topics:
            # Get dominant topic
            dominant_topic = max(article.topics, key=lambda x: x[1])[0]
            topic_counts[dominant_topic] = topic_counts.get(dominant_topic, 0) + 1

    if not topic_counts:
        logger.warning("No topic assignments found")
        return

    # Sort by topic ID
    sorted_topics = sorted(topic_counts.items())
    topic_ids = [f"Topic {t}" for t, _ in sorted_topics]
    counts = [c for _, c in sorted_topics]

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(topic_ids, counts, color='steelblue', alpha=0.7)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom')

    ax.set_xlabel('Topic', fontsize=12)
    ax.set_ylabel('Number of Articles', fontsize=12)
    ax.set_title(f'Topic Distribution - {newspaper_name}', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved topic distribution plot to {output_path}")


def plot_topic_wordclouds(topics: Dict[int, List[str]], newspaper_name: str,
                           output_dir: str):
    """
    Create wordclouds for each topic.

    Args:
        topics: Dictionary mapping topic_id to keyword list
        newspaper_name: Name of the newspaper
        output_dir: Directory to save wordclouds
    """
    logger.info(f"Creating topic wordclouds for {newspaper_name}...")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for topic_id, keywords in topics.items():
        # Create text from keywords (with frequency)
        # Repeat words based on their position (earlier = more important)
        text_parts = []
        for i, word in enumerate(keywords[:15]):  # Top 15 words
            frequency = len(keywords) - i  # Higher frequency for earlier words
            text_parts.extend([word] * frequency)

        text = ' '.join(text_parts)

        # Create wordcloud
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white',
            colormap='viridis',
            max_words=30
        ).generate(text)

        # Plot
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis('off')
        ax.set_title(f'{newspaper_name} - Topic {topic_id}\nTop Keywords: {", ".join(keywords[:5])}',
                     fontsize=12, fontweight='bold')

        # Save
        output_path = output_dir / f'{newspaper_name.lower().replace(" ", "_")}_topic_{topic_id}_wordcloud.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

    logger.info(f"Saved {len(topics)} wordclouds to {output_dir}")


def plot_topic_keywords_table(topics: Dict[int, List[str]], newspaper_name: str,
                                output_path: str, top_n: int = 10):
    """
    Create a table visualization of top keywords per topic.

    Args:
        topics: Dictionary mapping topic_id to keyword list
        newspaper_name: Name of the newspaper
        output_path: Path to save the plot
        top_n: Number of top keywords to show
    """
    logger.info(f"Creating topic keywords table for {newspaper_name}...")

    n_topics = len(topics)
    fig, ax = plt.subplots(figsize=(12, max(6, n_topics * 0.8)))

    # Prepare data for table
    table_data = []
    for topic_id in sorted(topics.keys()):
        keywords = topics[topic_id][:top_n]
        table_data.append([f"Topic {topic_id}", ", ".join(keywords)])

    # Create table
    table = ax.table(cellText=table_data,
                     colLabels=['Topic', f'Top {top_n} Keywords'],
                     cellLoc='left',
                     loc='center',
                     colWidths=[0.15, 0.85])

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)

    # Style header
    for i in range(2):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Alternate row colors
    for i in range(1, len(table_data) + 1):
        if i % 2 == 0:
            for j in range(2):
                table[(i, j)].set_facecolor('#f0f0f0')

    ax.axis('off')
    ax.set_title(f'Topic Keywords - {newspaper_name}', fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved topic keywords table to {output_path}")


def plot_topic_alignment_heatmap(alignment: Dict[int, tuple], topics_a: Dict[int, List[str]],
                                   topics_b: Dict[int, List[str]], newspaper_a: str,
                                   newspaper_b: str, output_path: str):
    """
    Create heatmap showing topic alignment between two newspapers.

    Args:
        alignment: Topic alignment mapping from comparative analysis
        topics_a: Topics from newspaper A
        topics_b: Topics from newspaper B
        newspaper_a: Name of newspaper A
        newspaper_b: Name of newspaper B
        output_path: Path to save the plot
    """
    logger.info("Creating topic alignment heatmap...")

    # Create similarity matrix
    n_topics_a = len(topics_a)
    n_topics_b = len(topics_b)

    similarity_matrix = np.zeros((n_topics_a, n_topics_b))

    for topic_a_id, (topic_b_id, similarity) in alignment.items():
        similarity_matrix[topic_a_id, topic_b_id] = similarity

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))

    sns.heatmap(similarity_matrix, annot=True, fmt='.2f', cmap='YlOrRd',
                xticklabels=[f"T{i}" for i in range(n_topics_b)],
                yticklabels=[f"T{i}" for i in range(n_topics_a)],
                cbar_kws={'label': 'Similarity Score'},
                ax=ax)

    ax.set_xlabel(f'{newspaper_b} Topics', fontsize=12)
    ax.set_ylabel(f'{newspaper_a} Topics', fontsize=12)
    ax.set_title(f'Topic Alignment: {newspaper_a} vs {newspaper_b}',
                 fontsize=14, fontweight='bold')

    plt.tight_layout()

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved topic alignment heatmap to {output_path}")


def create_all_topic_plots(articles: List[Article], topics: Dict[int, List[str]],
                            newspaper_name: str, output_dir: str):
    """
    Create all topic-related plots for a newspaper.

    Args:
        articles: List of articles with topic assignments
        topics: Dictionary of topic keywords
        newspaper_name: Name of the newspaper
        output_dir: Directory to save plots
    """
    logger.info(f"Creating all topic plots for {newspaper_name}...")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Topic distribution
    plot_topic_distribution(
        articles, topics, newspaper_name,
        output_dir / f'{newspaper_name.lower().replace(" ", "_")}_topic_distribution.png'
    )

    # Wordclouds
    wordcloud_dir = output_dir / 'topic_wordclouds'
    plot_topic_wordclouds(topics, newspaper_name, wordcloud_dir)

    # Keywords table
    plot_topic_keywords_table(
        topics, newspaper_name,
        output_dir / f'{newspaper_name.lower().replace(" ", "_")}_topic_keywords.png'
    )

    logger.info(f"Completed all topic plots for {newspaper_name}")
