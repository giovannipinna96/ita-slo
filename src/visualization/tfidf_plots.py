"""
Visualization module for TF-IDF analysis results.
"""

import logging
from pathlib import Path
from typing import List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")


def plot_top_tfidf_words(keywords: List[Tuple[str, float]], newspaper_name: str,
                          output_path: str, top_n: int = 20):
    """
    Plot top TF-IDF keywords as horizontal bar chart.

    Args:
        keywords: List of (word, score) tuples
        newspaper_name: Name of the newspaper
        output_path: Path to save the plot
        top_n: Number of top words to show
    """
    logger.info(f"Creating TF-IDF bar plot for {newspaper_name}...")

    # Get top N keywords
    top_keywords = keywords[:top_n]
    words = [w for w, _ in top_keywords]
    scores = [s for _, s in top_keywords]

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 8))

    # Horizontal bar chart
    bars = ax.barh(range(len(words)), scores, color='steelblue', alpha=0.7)

    # Add value labels
    for i, (bar, score) in enumerate(zip(bars, scores)):
        ax.text(score, i, f' {score:.3f}', va='center', fontsize=9)

    ax.set_yticks(range(len(words)))
    ax.set_yticklabels(words)
    ax.set_xlabel('TF-IDF Score', fontsize=12)
    ax.set_title(f'Top {top_n} TF-IDF Keywords - {newspaper_name}',
                 fontsize=14, fontweight='bold')
    ax.invert_yaxis()  # Highest score at top

    plt.tight_layout()

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved TF-IDF plot to {output_path}")


def plot_tfidf_comparison(keywords_a: List[Tuple[str, float]],
                           keywords_b: List[Tuple[str, float]],
                           newspaper_a: str, newspaper_b: str,
                           output_path: str, top_n: int = 15):
    """
    Create side-by-side comparison of TF-IDF keywords.

    Args:
        keywords_a: Keywords from newspaper A
        keywords_b: Keywords from newspaper B
        newspaper_a: Name of newspaper A
        newspaper_b: Name of newspaper B
        output_path: Path to save the plot
        top_n: Number of top words to show
    """
    logger.info("Creating TF-IDF comparison plot...")

    # Get top N keywords
    top_a = keywords_a[:top_n]
    top_b = keywords_b[:top_n]

    words_a = [w for w, _ in top_a]
    scores_a = [s for _, s in top_a]

    words_b = [w for w, _ in top_b]
    scores_b = [s for _, s in top_b]

    # Create side-by-side plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Newspaper A
    ax1.barh(range(len(words_a)), scores_a, color='steelblue', alpha=0.7)
    ax1.set_yticks(range(len(words_a)))
    ax1.set_yticklabels(words_a)
    ax1.set_xlabel('TF-IDF Score', fontsize=11)
    ax1.set_title(newspaper_a, fontsize=13, fontweight='bold')
    ax1.invert_yaxis()

    # Add scores
    for i, score in enumerate(scores_a):
        ax1.text(score, i, f' {score:.3f}', va='center', fontsize=8)

    # Newspaper B
    ax2.barh(range(len(words_b)), scores_b, color='coral', alpha=0.7)
    ax2.set_yticks(range(len(words_b)))
    ax2.set_yticklabels(words_b)
    ax2.set_xlabel('TF-IDF Score', fontsize=11)
    ax2.set_title(newspaper_b, fontsize=13, fontweight='bold')
    ax2.invert_yaxis()

    # Add scores
    for i, score in enumerate(scores_b):
        ax2.text(score, i, f' {score:.3f}', va='center', fontsize=8)

    fig.suptitle(f'TF-IDF Comparison: {newspaper_a} vs {newspaper_b}',
                 fontsize=15, fontweight='bold')

    plt.tight_layout()

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved TF-IDF comparison plot to {output_path}")


def create_all_tfidf_plots(keywords_a: List[Tuple[str, float]],
                            keywords_b: List[Tuple[str, float]],
                            newspaper_a: str, newspaper_b: str,
                            output_dir: str):
    """
    Create all TF-IDF plots.

    Args:
        keywords_a: Keywords from newspaper A
        keywords_b: Keywords from newspaper B
        newspaper_a: Name of newspaper A
        newspaper_b: Name of newspaper B
        output_dir: Directory to save plots
    """
    logger.info("Creating all TF-IDF plots...")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Individual plots
    plot_top_tfidf_words(
        keywords_a, newspaper_a,
        output_dir / f'{newspaper_a.lower().replace(" ", "_")}_tfidf.png'
    )

    plot_top_tfidf_words(
        keywords_b, newspaper_b,
        output_dir / f'{newspaper_b.lower().replace(" ", "_")}_tfidf.png'
    )

    # Comparison plot
    plot_tfidf_comparison(
        keywords_a, keywords_b, newspaper_a, newspaper_b,
        output_dir / 'tfidf_comparison.png'
    )

    logger.info("Completed all TF-IDF plots")
