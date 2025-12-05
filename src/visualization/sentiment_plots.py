"""
Visualization module for sentiment analysis results.

Includes:
- Overall newspaper sentiment pie charts
- Per-article sentiment pie charts (sentence-level voting)
- Comparison plots between newspapers
"""

import logging
import math
from pathlib import Path
from typing import Dict, List
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from src.models import Article

logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")


def plot_sentiment_distribution(sentiment_data: Dict, newspaper_name: str,
                                  output_path: str):
    """
    Plot sentiment distribution as pie chart.

    Args:
        sentiment_data: Sentiment analysis results with percentages
        newspaper_name: Name of the newspaper
        output_path: Path to save the plot
    """
    logger.info(f"Creating sentiment distribution pie chart for {newspaper_name}...")

    percentages = sentiment_data.get('percentages', {})
    counts = sentiment_data.get('counts', {})

    labels = ['Positive', 'Neutral', 'Negative']
    sizes = [
        percentages.get('positive', 0),
        percentages.get('neutral', 0),
        percentages.get('negative', 0)
    ]
    actual_counts = [
        counts.get('positive', 0),
        counts.get('neutral', 0),
        counts.get('negative', 0)
    ]

    colors = ['#4CAF50', '#FFC107', '#F44336']
    explode = (0.05, 0, 0.05)  # Explode positive and negative slightly

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 7))

    wedges, texts, autotexts = ax.pie(
        sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', startangle=90, textprops={'fontsize': 11}
    )

    # Make percentage text bold
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')

    # Add counts in legend
    legend_labels = [f'{label}: {count} articles' for label, count in zip(labels, actual_counts)]
    ax.legend(legend_labels, loc='upper left', bbox_to_anchor=(1, 1))

    ax.set_title(f'Sentiment Distribution - {newspaper_name}',
                 fontsize=14, fontweight='bold')

    plt.tight_layout()

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved sentiment distribution plot to {output_path}")


def plot_article_sentiment_pie(article: Article, output_path: str):
    """
    Plot sentiment distribution pie chart for a single article.
    
    The chart shows the distribution of sentiment across sentences in the article.
    The article title is used as the chart title.
    
    Args:
        article: Article with sentence-level sentiment data
        output_path: Path to save the plot
    """
    if not article.sentiment or 'vote_distribution' not in article.sentiment:
        logger.warning(f"Article '{article.title}' has no sentence-level sentiment data")
        return
    
    vote_dist = article.sentiment.get('vote_distribution', {})
    sentence_count = article.sentiment.get('sentence_count', 0)
    
    labels = ['Positive', 'Neutral', 'Negative']
    sizes = [
        vote_dist.get('positive', 0),
        vote_dist.get('neutral', 0),
        vote_dist.get('negative', 0)
    ]
    
    # Skip if no sentences
    if sum(sizes) == 0:
        logger.warning(f"Article '{article.title}' has no sentiment votes")
        return
    
    colors = ['#4CAF50', '#FFC107', '#F44336']
    
    # Create plot
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Filter out zero values for cleaner pie
    non_zero_labels = []
    non_zero_sizes = []
    non_zero_colors = []
    for label, size, color in zip(labels, sizes, colors):
        if size > 0:
            non_zero_labels.append(label)
            non_zero_sizes.append(size)
            non_zero_colors.append(color)
    
    if non_zero_sizes:
        wedges, texts, autotexts = ax.pie(
            non_zero_sizes, labels=non_zero_labels, colors=non_zero_colors,
            autopct=lambda pct: f'{pct:.1f}%\n({int(round(pct/100*sum(non_zero_sizes)))})',
            startangle=90, textprops={'fontsize': 10}
        )
        
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
    
    # Truncate title if too long
    title = article.title if article.title else "Untitled Article"
    if len(title) > 60:
        title = title[:57] + "..."
    
    ax.set_title(title, fontsize=12, fontweight='bold', wrap=True)
    
    # Add subtitle with overall sentiment
    overall_label = article.sentiment.get('label', 'unknown').capitalize()
    overall_score = article.sentiment.get('score', 0)
    subtitle = f"Overall: {overall_label} | Sentences: {sentence_count}"
    ax.text(0.5, -0.1, subtitle, transform=ax.transAxes, ha='center', 
            fontsize=10, style='italic')
    
    plt.tight_layout()
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()


def plot_all_articles_sentiment(articles: List[Article], newspaper_name: str, 
                                 output_dir: str, max_cols: int = 4):
    """
    Create a grid of sentiment pie charts for all articles in a newspaper.
    
    Args:
        articles: List of articles with sentiment data
        newspaper_name: Name of the newspaper
        output_dir: Directory to save plots
        max_cols: Maximum columns in the grid
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Filter articles with sentiment data
    # Accept both sentence-level (vote_distribution) and article-level (label only)
    valid_articles = []
    for a in articles:
        if a.sentiment:
            if 'vote_distribution' in a.sentiment and sum(a.sentiment.get('vote_distribution', {}).values()) > 0:
                valid_articles.append(a)
            elif 'label' in a.sentiment:
                # Convert article-level to fake vote_distribution for visualization
                label = a.sentiment['label'].lower()
                a.sentiment['vote_distribution'] = {
                    'positive': 1 if 'pos' in label else 0,
                    'neutral': 1 if 'neut' in label or label == 'neutral' else 0,
                    'negative': 1 if 'neg' in label else 0
                }
                a.sentiment['sentence_count'] = 1
                valid_articles.append(a)
    
    if not valid_articles:
        logger.warning(f"No articles with sentiment data for {newspaper_name}")
        return
    
    logger.info(f"Creating sentiment pie charts for {len(valid_articles)} articles from {newspaper_name}...")
    
    # Create individual plots
    articles_dir = output_dir / f"{newspaper_name.lower().replace(' ', '_')}_articles"
    articles_dir.mkdir(parents=True, exist_ok=True)
    
    for i, article in enumerate(valid_articles):
        safe_title = "".join(c for c in article.title[:30] if c.isalnum() or c in (' ', '-', '_')).strip()
        safe_title = safe_title.replace(' ', '_') if safe_title else f"article_{i}"
        output_path = articles_dir / f"{i+1:02d}_{safe_title}.png"
        plot_article_sentiment_pie(article, output_path)
    
    # Create grid overview
    n_articles = len(valid_articles)
    n_cols = min(max_cols, n_articles)
    n_rows = math.ceil(n_articles / n_cols)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
    
    # Flatten axes for easy iteration
    if n_articles == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if n_rows > 1 or n_cols > 1 else [axes]
    
    colors = ['#4CAF50', '#FFC107', '#F44336']
    
    for idx, article in enumerate(valid_articles):
        ax = axes[idx]
        
        vote_dist = article.sentiment.get('vote_distribution', {})
        sizes = [
            vote_dist.get('positive', 0),
            vote_dist.get('neutral', 0),
            vote_dist.get('negative', 0)
        ]
        
        # Filter zeros
        non_zero = [(s, c) for s, c in zip(sizes, colors) if s > 0]
        if non_zero:
            non_zero_sizes, non_zero_colors = zip(*non_zero)
            ax.pie(non_zero_sizes, colors=non_zero_colors, 
                   autopct='%1.0f%%', startangle=90, textprops={'fontsize': 8})
        
        # Title (truncated)
        title = article.title[:35] + "..." if len(article.title) > 35 else article.title
        ax.set_title(title, fontsize=9, fontweight='bold')
    
    # Hide unused axes
    for idx in range(n_articles, len(axes)):
        axes[idx].set_visible(False)
    
    fig.suptitle(f'Sentiment per Article - {newspaper_name}', fontsize=14, fontweight='bold')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#4CAF50', label='Positive'),
        Patch(facecolor='#FFC107', label='Neutral'),
        Patch(facecolor='#F44336', label='Negative')
    ]
    fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.99, 0.99))
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    output_path = output_dir / f"{newspaper_name.lower().replace(' ', '_')}_articles_sentiment_grid.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved article sentiment grid to {output_path}")
    logger.info(f"Saved {len(valid_articles)} individual article charts to {articles_dir}/")


def plot_sentiment_comparison(sentiment_a: Dict, sentiment_b: Dict,
                                newspaper_a: str, newspaper_b: str,
                                output_path: str):
    """
    Create side-by-side bar plot comparing sentiment distributions.

    Args:
        sentiment_a: Sentiment data for newspaper A
        sentiment_b: Sentiment data for newspaper B
        newspaper_a: Name of newspaper A
        newspaper_b: Name of newspaper B
        output_path: Path to save the plot
    """
    logger.info("Creating sentiment comparison plot...")

    percentages_a = sentiment_a.get('percentages', {})
    percentages_b = sentiment_b.get('percentages', {})

    labels = ['Positive', 'Neutral', 'Negative']
    values_a = [
        percentages_a.get('positive', 0),
        percentages_a.get('neutral', 0),
        percentages_a.get('negative', 0)
    ]
    values_b = [
        percentages_b.get('positive', 0),
        percentages_b.get('neutral', 0),
        percentages_b.get('negative', 0)
    ]

    x = np.arange(len(labels))
    width = 0.35

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))

    bars1 = ax.bar(x - width/2, values_a, width, label=newspaper_a,
                    color='steelblue', alpha=0.8)
    bars2 = ax.bar(x + width/2, values_b, width, label=newspaper_b,
                    color='coral', alpha=0.8)

    # Add value labels on bars
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}%',
                        ha='center', va='bottom', fontsize=9)

    add_labels(bars1)
    add_labels(bars2)

    ax.set_xlabel('Sentiment', fontsize=12)
    ax.set_ylabel('Percentage (%)', fontsize=12)
    ax.set_title(f'Sentiment Comparison: {newspaper_a} vs {newspaper_b}',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.set_ylim(0, max(max(values_a), max(values_b)) * 1.15)

    plt.tight_layout()

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved sentiment comparison plot to {output_path}")


def plot_sentiment_bar_comparison(sentiment_a: Dict, sentiment_b: Dict,
                                    newspaper_a: str, newspaper_b: str,
                                    output_path: str):
    """
    Create stacked bar chart showing absolute numbers.

    Args:
        sentiment_a: Sentiment data for newspaper A
        sentiment_b: Sentiment data for newspaper B
        newspaper_a: Name of newspaper A
        newspaper_b: Name of newspaper B
        output_path: Path to save the plot
    """
    logger.info("Creating sentiment stacked bar comparison...")

    counts_a = sentiment_a.get('counts', {})
    counts_b = sentiment_b.get('counts', {})

    newspapers = [newspaper_a, newspaper_b]
    positive = [counts_a.get('positive', 0), counts_b.get('positive', 0)]
    neutral = [counts_a.get('neutral', 0), counts_b.get('neutral', 0)]
    negative = [counts_a.get('negative', 0), counts_b.get('negative', 0)]

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(newspapers))
    width = 0.5

    p1 = ax.bar(x, positive, width, label='Positive', color='#4CAF50')
    p2 = ax.bar(x, neutral, width, bottom=positive, label='Neutral', color='#FFC107')
    p3 = ax.bar(x, negative, width, bottom=np.array(positive) + np.array(neutral),
                label='Negative', color='#F44336')

    ax.set_ylabel('Number of Articles', fontsize=12)
    ax.set_title('Sentiment Distribution by Newspaper (Absolute Numbers)',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(newspapers)
    ax.legend(loc='upper right')

    # Add total count on top
    totals = [sum(x) for x in zip(positive, neutral, negative)]
    for i, total in enumerate(totals):
        ax.text(i, total, f'Total: {total}', ha='center', va='bottom',
                fontweight='bold')

    plt.tight_layout()

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved sentiment stacked bar plot to {output_path}")


def create_all_sentiment_plots(sentiment_a: Dict, sentiment_b: Dict,
                                newspaper_a: str, newspaper_b: str,
                                output_dir: str,
                                articles_a: List[Article] = None,
                                articles_b: List[Article] = None):
    """
    Create all sentiment plots.

    Args:
        sentiment_a: Sentiment data for newspaper A
        sentiment_b: Sentiment data for newspaper B
        newspaper_a: Name of newspaper A
        newspaper_b: Name of newspaper B
        output_dir: Directory to save plots
        articles_a: Optional list of articles from newspaper A (for per-article plots)
        articles_b: Optional list of articles from newspaper B (for per-article plots)
    """
    logger.info("Creating all sentiment plots...")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Individual pie charts (overall sentiment)
    plot_sentiment_distribution(
        sentiment_a, newspaper_a,
        output_dir / f'{newspaper_a.lower().replace(" ", "_")}_sentiment_pie.png'
    )

    plot_sentiment_distribution(
        sentiment_b, newspaper_b,
        output_dir / f'{newspaper_b.lower().replace(" ", "_")}_sentiment_pie.png'
    )

    # Comparison plots
    plot_sentiment_comparison(
        sentiment_a, sentiment_b, newspaper_a, newspaper_b,
        output_dir / 'sentiment_comparison_percentage.png'
    )

    plot_sentiment_bar_comparison(
        sentiment_a, sentiment_b, newspaper_a, newspaper_b,
        output_dir / 'sentiment_comparison_stacked.png'
    )

    # Per-article sentiment charts (if articles provided)
    if articles_a:
        logger.info(f"Creating per-article sentiment charts for {newspaper_a}...")
        plot_all_articles_sentiment(articles_a, newspaper_a, output_dir)
    
    if articles_b:
        logger.info(f"Creating per-article sentiment charts for {newspaper_b}...")
        plot_all_articles_sentiment(articles_b, newspaper_b, output_dir)

    logger.info("Completed all sentiment plots")
