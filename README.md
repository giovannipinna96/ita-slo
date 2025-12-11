# 📰 Analisi NLP di Giornali Storici

Pipeline di analisi linguistica per confrontare giornali storici italiani e sloveni di Trieste (1902).

## 🎯 Funzionalità

- **Topic Modeling**: LDA e BERTopic per identificare i temi principali
- **Sentiment Analysis**: Analisi del tono con modelli specifici per lingua
- **TF-IDF**: Estrazione di parole chiave distintive
- **Clustering**: Raggruppamento semantico degli articoli
- **Analisi Comparativa**: Confronto tra giornali o sezioni

## 📋 Requisiti

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) package manager

## 🚀 Installazione

```bash
# Clona il repository
cd ita_slo

# Installa le dipendenze con uv
uv sync
```

## 📖 Utilizzo

### 1. Analisi Principale (Italiano vs Sloveno)

Confronta Il Piccolo (IT) con Edinost (SL):

```bash
uv run python main.py
```

**Output:**
- `outputs/results/` - Risultati JSON delle analisi
- `outputs/visualizations/` - Grafici e wordcloud
- `outputs/parsed/` - Articoli estratti

---

### 2. Confronto Stessa Lingua

#### Confronta le pagine di Il Piccolo

```bash
uv run python compare_same_language.py --mode pages --language it
```

#### Confronta due file custom

```bash
uv run python compare_same_language.py \
    --mode custom \
    --source1 path/to/primo_file.txt \
    --source2 path/to/secondo_file.txt \
    --name1 "Giornale A" \
    --name2 "Giornale B" \
    --language it
```

**Parametri:**

| Parametro | Descrizione | Default |
|-----------|-------------|---------|
| `--mode` | `pages` (pagine Il Piccolo) o `custom` (file custom) | `pages` |
| `--language` | Lingua: `it` (italiano) o `sl` (sloveno) | `it` |
| `--source1` | Percorso al primo file di testo | - |
| `--source2` | Percorso al secondo file di testo | - |
| `--name1` | Nome descrittivo per il primo file | `Source 1` |
| `--name2` | Nome descrittivo per il secondo file | `Source 2` |

**Esempio con file sloveni:**

```bash
uv run python compare_same_language.py \
    --mode custom \
    --source1 data/giornale_sloveno_1.txt \
    --source2 data/giornale_sloveno_2.txt \
    --name1 "Edinost Mattina" \
    --name2 "Edinost Sera" \
    --language sl
```

---

### 3. Confronto Rapido tra Due File (`compare_two.py`)

Script semplificato per confrontare due file di testo con analisi sentiment, topic modeling e estrazione delle frasi più significative.

#### Sintassi

```bash
uv run python compare_two.py <file1> <file2> [opzioni]
```

#### Esempio

```bash
uv run python compare_two.py \
    data/il_piccolo_19020909_tot.txt \
    data/edinost_19020909_traduzione_italiana.txt \
    --name1 "Il Piccolo" \
    --name2 "Edinost" \
    --language it \
    --output outputs/my_comparison
```

#### Parametri

| Parametro | Descrizione | Default |
|-----------|-------------|---------|
| `file1` | Percorso al primo file (posizionale, obbligatorio) | - |
| `file2` | Percorso al secondo file (posizionale, obbligatorio) | - |
| `--name1` | Nome visualizzato per il primo file | nome del file |
| `--name2` | Nome visualizzato per il secondo file | nome del file |
| `--language` `-l` | Lingua: `it` (italiano) o `sl` (sloveno) | `it` |
| `--output` `-o` | Cartella di output | `outputs/comparison` |

#### Output Generati

```
outputs/my_comparison/
├── il_piccolo_results.json              # Risultati completi analisi
├── edinost_results.json                 # Risultati completi analisi
├── il_piccolo_extreme_sentences.json   # 🆕 Top 10 frasi positive/negative
├── edinost_extreme_sentences.json      # 🆕 Top 10 frasi positive/negative
├── sentiment_comparison.png             # Confronto sentiment (grafico)
├── il_piccolo_articles_sentiment_grid.png  # Griglia sentiment per articolo
├── edinost_articles_sentiment_grid.png     # Griglia sentiment per articolo
├── il_piccolo_articles/                 # Grafici singoli per ogni articolo
│   ├── 01_titolo_articolo.png
│   └── ...
└── edinost_articles/                    # Grafici singoli per ogni articolo
    ├── 01_titolo_articolo.png
    └── ...
```

---

### 4. Analisi Singolo File (`analyze_single.py`) 🆕

Analizza un singolo file di testo con output dettagliato per ogni articolo.

#### Sintassi

```bash
python analyze_single.py <file> [opzioni]
```

#### Esempio

```bash
# Italiano con BERTopic
python analyze_single.py data/piccolo_19020217.txt --language it --use-bertopic

# Sloveno con BERTopic
python analyze_single.py data/edinost_19020217.txt --language sl --use-bertopic
```

#### Parametri

| Parametro | Descrizione | Default |
|-----------|-------------|---------|
| `file` | Percorso al file da analizzare (obbligatorio) | - |
| `--language` `-l` | Lingua: `it` o `sl` | `it` |
| `--use-bertopic` | Attiva BERTopic (richiede GPU) | `False` |
| `--output` `-o` | Cartella di output | `outputs/single` |
| `--num-topics` | Numero di topic per LDA/BERTopic | auto |

#### Struttura Output

```
outputs/single/{filename}/
├── global_results.json           # Risultati globali (sentiment, TF-IDF, topics)
├── global_extreme_sentences.json # Frasi più positive/negative
├── {name}_articles_sentiment_grid.png  # Griglia sentiment globale
└── articles/                     # 📁 Cartella articoli
    ├── 01/                       # Primo articolo
    │   ├── original_text.txt     # Testo originale con titolo
    │   ├── analysis.json         # Analisi completa (vedi sotto)
    │   ├── sentiment_pie.png     # 🥧 Pie chart sentiment
    │   ├── topic_distribution_lda.png      # 📊 Topics LDA
    │   ├── topic_distribution_bertopic.png # 📊 Topics BERTopic
    │   └── wordcloud.png         # ☁️ Word cloud
    ├── 02/
    └── ...
```

#### Contenuto `analysis.json` per Articolo

```json
{
  "article_number": 1,
  "title": "Titolo dell'articolo",
  "content_length": 1560,
  "language": "it",
  "source": "piccolo_19020217",
  "sentiment": {
    "label": "negative",
    "score": 0.92,
    "distribution": {"positive": 0.1, "neutral": 0.1, "negative": 0.8},
    "sentence_details": [...]
  },
  "topics_lda": [
    {"topic_id": 0, "probability": 0.85},
    {"topic_id": 2, "probability": 0.15}
  ],
  "topics_bertopic": {
    "topic_id": 2,
    "topic_name": "2_vienna_governo_ministro",
    "keywords": ["vienna", "governo", "ministro", "austria", "politica"]
  },
  "tfidf_keywords": [["parola1", 0.73], ["parola2", 0.36]]
}
```

#### Formati File Supportati

Il parser riconosce automaticamente questi formati:

| Pattern | Descrizione |
|---------|-------------|
| `piccolo_*.txt` / `edinost_*.txt` | Nuovo formato con separatori `===` |
| `il_piccolo_*.txt` / `edinost_19020909*.txt` | Formato originale con marker `PAGINA` |

#### Formato `*_extreme_sentences.json`

Contiene le 10 frasi più positive e le 10 più negative con i relativi punteggi:

```json
{
    "source": "Il Piccolo",
    "total_sentences": 312,
    "positive_count": 45,
    "negative_count": 128,
    "neutral_count": 139,
    "top_positive": [
        {
            "rank": 1,
            "sentence": "La festa procede animatissima e il concorso è enorme.",
            "confidence": 0.9823,
            "article": "Piedigrotta"
        }
    ],
    "top_negative": [
        {
            "rank": 1,
            "sentence": "Il suo stato è grave e i medici non danno speranza.",
            "confidence": 0.9654,
            "article": "Cronaca Locale"
        }
    ]
}
```

## 🤖 Modelli AI Utilizzati

### Sentiment Analysis

| Lingua | Modello | Descrizione |
|--------|---------|-------------|
| 🇮🇹 Italiano | `MilaNLProc/feel-it-italian-sentiment` | FEEL-IT (classificazione binaria: positivo/negativo) |
| 🇸🇮 Sloveno | `classla/xlm-r-parlasent` | Addestrato su testi parlamentari sloveni |

### Topic Modeling (BERTopic)

| Lingua | Embedding Model | Stopwords |
|--------|-----------------|-----------|
| 🇮🇹 Italiano | `dbmdz/bert-base-italian-cased` | spaCy `it_core_news_lg` (~300 parole) |
| 🇸🇮 Sloveno | `EMBEDDIA/sloberta` | spaCy `sl_core_news_lg` (~200 parole) |

> **Nota**: BERTopic usa KMeans per garantire un numero fisso di topic. Le stopwords vengono caricate automaticamente da spaCy.

---

## 📁 Struttura Output

```
outputs/
├── parsed/                     # Articoli estratti (JSON)
├── models/                     # Modelli addestrati
├── results/                    # Risultati analisi
│   ├── analysis_summary.json
│   ├── *_sentiment.json
│   ├── *_tfidf.json
│   ├── *_lda_topics.json
│   └── page_comparisons/       # Confronti tra pagine
└── visualizations/             # Grafici
    ├── *_sentiment_pie.png
    ├── *_topic_distribution.png
    └── topic_wordclouds/
```

---

## 📊 Formato File Input

I file di testo devono contenere articoli separati. Il parser riconosce automaticamente:
- Titoli in maiuscolo
- Separatori tra articoli
- Struttura tipica dei giornali d'epoca

---

## 🛠️ Sviluppo

### Struttura del progetto

```
src/
├── preprocessing/      # Caricamento e pulizia testi
├── analysis/           # Moduli di analisi NLP
├── visualization/      # Generazione grafici
├── models.py           # Dataclass per articoli
└── utils/              # Utilità varie
```

### Eseguire i test

```bash
uv run python test_parsing.py
```

---

## 📚 Riferimenti

- Grootendorst, M. (2022). "BERTopic: Neural topic modeling"
- EMBEDDIA Project: SloBERTa per lo sloveno
- CLASSLA: Risorse NLP per le lingue slave

---

## 📝 Licenza

Progetto di ricerca per analisi storica di giornali triestini.
