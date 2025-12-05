"""
Visualization module for comparative analysis results.
"""

import logging
from pathlib import Path
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import numpy as np
from matplotlib_venn import venn2

logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")


def plot_vocabulary_venn(vocab_comparison: Dict, newspaper_a: str,
                          newspaper_b: str, output_path: str):
    """
    Create Venn diagram showing vocabulary overlap.

    Args:
        vocab_comparison: Vocabulary comparison data
        newspaper_a: Name of newspaper A
        newspaper_b: Name of newspaper B
        output_path: Path to save the plot
    """
    logger.info("Creating vocabulary Venn diagram...")

    size_a = vocab_comparison.get('vocab_size_a', 0)
    size_b = vocab_comparison.get('vocab_size_b', 0)
    common = vocab_comparison.get('common_words_count', 0)

    unique_a = size_a - common
    unique_b = size_b - common

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 7))

    venn = venn2(subsets=(unique_a, unique_b, common),
                 set_labels=(newspaper_a, newspaper_b),
                 ax=ax)

    # Customize colors
    venn.get_patch_by_id('10').set_color('steelblue')
    venn.get_patch_by_id('10').set_alpha(0.5)
    venn.get_patch_by_id('01').set_color('coral')
    venn.get_patch_by_id('01').set_alpha(0.5)
    venn.get_patch_by_id('11').set_color('purple')
    venn.get_patch_by_id('11').set_alpha(0.5)

    overlap_pct = vocab_comparison.get('overlap_percentage', 0)
    ax.set_title(f'Vocabulary Overlap: {newspaper_a} vs {newspaper_b}\n'
                 f'({overlap_pct:.1f}% overlap)',
                 fontsize=14, fontweight='bold')

    plt.tight_layout()

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved vocabulary Venn diagram to {output_path}")


def plot_distinctive_terms_wordclouds(distinctive_terms: Dict, newspaper_a: str,
                                       newspaper_b: str, output_path: str):
    """
    Create side-by-side wordclouds of distinctive terms.

    Args:
        distinctive_terms: Dictionary with distinctive terms for each newspaper
        newspaper_a: Name of newspaper A
        newspaper_b: Name of newspaper B
        output_path: Path to save the plot
    """
    logger.info("Creating distinctive terms wordclouds...")

    terms_a = distinctive_terms.get('distinctive_to_a', [])
    terms_b = distinctive_terms.get('distinctive_to_b', [])

    # Create frequency dictionaries (no duplicates, unique words with scores)
    # Use a dict to ensure each word appears only once
    freq_a = {}
    for word, score in terms_a:
        if word not in freq_a:  # Only keep first occurrence
            freq_a[word] = max(score * 100, 1)  # Scale score for visibility
    
    freq_b = {}
    for word, score in terms_b:
        if word not in freq_b:  # Only keep first occurrence
            freq_b[word] = max(score * 100, 1)  # Scale score for visibility

    # Create wordclouds using frequency dictionaries
    if freq_a:
        wordcloud_a = WordCloud(width=600, height=400, background_color='white',
                                 colormap='Blues', max_words=40).generate_from_frequencies(freq_a)
    else:
        wordcloud_a = WordCloud(width=600, height=400, background_color='white').generate("no data")
    
    if freq_b:
        wordcloud_b = WordCloud(width=600, height=400, background_color='white',
                                 colormap='Reds', max_words=40).generate_from_frequencies(freq_b)
    else:
        wordcloud_b = WordCloud(width=600, height=400, background_color='white').generate("no data")

    # Create side-by-side plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    ax1.imshow(wordcloud_a, interpolation='bilinear')
    ax1.axis('off')
    ax1.set_title(f'{newspaper_a}\nDistinctive Terms ({len(freq_a)} unique)', fontsize=13, fontweight='bold')

    ax2.imshow(wordcloud_b, interpolation='bilinear')
    ax2.axis('off')
    ax2.set_title(f'{newspaper_b}\nDistinctive Terms ({len(freq_b)} unique)', fontsize=13, fontweight='bold')

    fig.suptitle('Distinctive Vocabulary Comparison', fontsize=15, fontweight='bold')

    plt.tight_layout()

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved distinctive terms wordclouds to {output_path}")


def plot_common_topics_comparison(common_topics: List[Tuple[int, int, float]],
                                    topics_a: Dict, topics_b: Dict,
                                    newspaper_a: str, newspaper_b: str,
                                    output_path: str):
    """
    Visualize common topics with their keywords.

    Args:
        common_topics: List of (topic_a_id, topic_b_id, similarity) tuples
        topics_a: Topics from newspaper A
        topics_b: Topics from newspaper B
        newspaper_a: Name of newspaper A
        newspaper_b: Name of newspaper B
        output_path: Path to save the plot
    """
    logger.info("Creating common topics comparison...")

    if not common_topics:
        logger.warning("No common topics found")
        return

    n_topics = min(len(common_topics), 5)  # Show max 5 common topics
    common_topics = sorted(common_topics, key=lambda x: x[2], reverse=True)[:n_topics]

    fig, axes = plt.subplots(n_topics, 2, figsize=(14, n_topics * 2))

    if n_topics == 1:
        axes = [axes]

    for idx, (topic_a_id, topic_b_id, similarity) in enumerate(common_topics):
        keywords_a = topics_a.get(topic_a_id, [])[:10]
        keywords_b = topics_b.get(topic_b_id, [])[:10]

        # Left: Newspaper A
        axes[idx][0].barh(range(len(keywords_a)), range(len(keywords_a), 0, -1),
                          color='steelblue', alpha=0.7)
        axes[idx][0].set_yticks(range(len(keywords_a)))
        axes[idx][0].set_yticklabels(keywords_a, fontsize=9)
        axes[idx][0].set_title(f'{newspaper_a} - Topic {topic_a_id}',
                                fontsize=10, fontweight='bold')
        axes[idx][0].set_xlabel('Relevance', fontsize=9)
        axes[idx][0].invert_yaxis()

        # Right: Newspaper B
        axes[idx][1].barh(range(len(keywords_b)), range(len(keywords_b), 0, -1),
                          color='coral', alpha=0.7)
        axes[idx][1].set_yticks(range(len(keywords_b)))
        axes[idx][1].set_yticklabels(keywords_b, fontsize=9)
        axes[idx][1].set_title(f'{newspaper_b} - Topic {topic_b_id}',
                                fontsize=10, fontweight='bold')
        axes[idx][1].set_xlabel('Relevance', fontsize=9)
        axes[idx][1].invert_yaxis()

        # Add similarity score
        fig.text(0.5, 1 - (idx + 0.5) / n_topics, f'Similarity: {similarity:.2f}',
                 ha='center', fontsize=10, fontweight='bold',
                 bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

    fig.suptitle(f'Common Topics: {newspaper_a} vs {newspaper_b}',
                 fontsize=14, fontweight='bold', y=0.995)

    plt.tight_layout()

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved common topics comparison to {output_path}")


def plot_summary_dashboard(summary_data: Dict, newspaper_a: str,
                            newspaper_b: str, output_path: str):
    """
    Create a summary dashboard with key metrics.

    Args:
        summary_data: Dictionary with all analysis results
        newspaper_a: Name of newspaper A
        newspaper_b: Name of newspaper B
        output_path: Path to save the plot
    """
    logger.info("Creating summary dashboard...")

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # Title
    fig.suptitle(f'Analysis Summary: {newspaper_a} vs {newspaper_b}',
                 fontsize=16, fontweight='bold', y=0.98)

    # 1. Article counts
    ax1 = fig.add_subplot(gs[0, 0])
    preprocessing = summary_data.get('preprocessing', {})
    articles_a = preprocessing.get('il_piccolo_articles', 0)
    articles_b = preprocessing.get('edinost_articles', 0)

    ax1.bar([newspaper_a, newspaper_b], [articles_a, articles_b],
            color=['steelblue', 'coral'], alpha=0.7)
    ax1.set_ylabel('Count')
    ax1.set_title('Articles Analyzed', fontweight='bold')
    for i, v in enumerate([articles_a, articles_b]):
        ax1.text(i, v, str(v), ha='center', va='bottom', fontweight='bold')

    # 2. Vocabulary sizes
    ax2 = fig.add_subplot(gs[0, 1])
    stats_a = preprocessing.get('il_piccolo_stats', {})
    stats_b = preprocessing.get('edinost_stats', {})
    vocab_a = stats_a.get('vocabulary_size', 0)
    vocab_b = stats_b.get('vocabulary_size', 0)

    ax2.bar([newspaper_a, newspaper_b], [vocab_a, vocab_b],
            color=['steelblue', 'coral'], alpha=0.7)
    ax2.set_ylabel('Unique Words')
    ax2.set_title('Vocabulary Size', fontweight='bold')
    for i, v in enumerate([vocab_a, vocab_b]):
        ax2.text(i, v, str(v), ha='center', va='bottom', fontweight='bold')

    # 3. Topics count
    ax3 = fig.add_subplot(gs[0, 2])
    topic_data = summary_data.get('topic_modeling', {})
    topics_a = topic_data.get('il_piccolo', {}).get('num_topics', 0)
    topics_b = topic_data.get('edinost', {}).get('num_topics', 0)

    ax3.bar([newspaper_a, newspaper_b], [topics_a, topics_b],
            color=['steelblue', 'coral'], alpha=0.7)
    ax3.set_ylabel('Count')
    ax3.set_title('Topics Identified (LDA)', fontweight='bold')
    for i, v in enumerate([topics_a, topics_b]):
        ax3.text(i, v, str(v), ha='center', va='bottom', fontweight='bold')

    # 4. Sentiment comparison
    ax4 = fig.add_subplot(gs[1, :])
    sentiment_data = summary_data.get('sentiment', {})
    sent_a = sentiment_data.get('il_piccolo', {}).get('percentages', {})
    sent_b = sentiment_data.get('edinost', {}).get('percentages', {})

    labels = ['Positive', 'Neutral', 'Negative']
    x = np.arange(len(labels))
    width = 0.35

    values_a = [sent_a.get('positive', 0), sent_a.get('neutral', 0), sent_a.get('negative', 0)]
    values_b = [sent_b.get('positive', 0), sent_b.get('neutral', 0), sent_b.get('negative', 0)]

    ax4.bar(x - width/2, values_a, width, label=newspaper_a, color='steelblue', alpha=0.7)
    ax4.bar(x + width/2, values_b, width, label=newspaper_b, color='coral', alpha=0.7)

    ax4.set_ylabel('Percentage (%)')
    ax4.set_title('Sentiment Distribution', fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(labels)
    ax4.legend()

    # 5. Key metrics table
    ax5 = fig.add_subplot(gs[2, :])
    ax5.axis('off')

    comparative = summary_data.get('comparative', {})
    metrics = [
        ['Metric', newspaper_a, newspaper_b],
        ['Articles', str(articles_a), str(articles_b)],
        ['Vocabulary', str(vocab_a), str(vocab_b)],
        ['Topics (LDA)', str(topics_a), str(topics_b)],
        ['Common Topics', str(comparative.get('n_common_topics', 0)), '-'],
        ['Avg Article Length', f"{stats_a.get('avg_article_length', 0):.0f}",
         f"{stats_b.get('avg_article_length', 0):.0f}"],
    ]

    table = ax5.table(cellText=metrics, cellLoc='center', loc='center',
                      colWidths=[0.3, 0.35, 0.35])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Style header
    for i in range(3):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Alternate row colors
    for i in range(1, len(metrics)):
        if i % 2 == 0:
            for j in range(3):
                table[(i, j)].set_facecolor('#f0f0f0')

    ax5.set_title('Summary Metrics', fontweight='bold', pad=20, fontsize=12)

    plt.tight_layout()

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved summary dashboard to {output_path}")


def create_all_comparison_plots(comparative_results: Dict, topics_a: Dict,
                                 topics_b: Dict, summary_data: Dict,
                                 newspaper_a: str, newspaper_b: str,
                                 output_dir: str):
    """
    Create all comparison plots.

    Args:
        comparative_results: Results from comparative analysis
        topics_a: Topics from newspaper A
        topics_b: Topics from newspaper B
        summary_data: All analysis summary data
        newspaper_a: Name of newspaper A
        newspaper_b: Name of newspaper B
        output_dir: Directory to save plots
    """
    logger.info("Creating all comparison plots...")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Vocabulary Venn diagram
    plot_vocabulary_venn(
        comparative_results.get('vocabulary_comparison', {}),
        newspaper_a, newspaper_b,
        output_dir / 'vocabulary_venn.png'
    )

    # Distinctive terms wordclouds
    plot_distinctive_terms_wordclouds(
        comparative_results.get('distinctive_terms', {}),
        newspaper_a, newspaper_b,
        output_dir / 'distinctive_terms_wordclouds.png'
    )

    # Common topics (if any)
    common_topics = comparative_results.get('common_topics', [])
    if common_topics:
        plot_common_topics_comparison(
            common_topics, topics_a, topics_b,
            newspaper_a, newspaper_b,
            output_dir / 'common_topics_comparison.png'
        )

    # Summary dashboard
    plot_summary_dashboard(
        summary_data, newspaper_a, newspaper_b,
        output_dir / 'summary_dashboard.png'
    )

    logger.info("Completed all comparison plots")
