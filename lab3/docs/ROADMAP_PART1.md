# ROADMAP, Lab 3, Part 1: Arabic Text Classification (RNN / BiRNN / GRU / LSTM)

> **Instructions for the AI assistant (Copilot):** Follow this roadmap step by step, in order. Work in a Kaggle notebook (`part1_classification.ipynb`) with Internet and GPU enabled. After each step, show a small sample of the output (head of dataframe, shapes, a plot) before moving on. Use PyTorch only (no Keras/TensorFlow). Fix all random seeds (`random`, `numpy`, `torch`) to 42.

---

## 0. Global decisions (do not change)

| Item | Decision |
|---|---|
| Topic | **Football (كرة القدم)** |
| Language | Modern Standard Arabic (some Moroccan/Egyptian news style) |
| Task | Predict the **relevance score (0-10)** of a text to football |
| Main formulation | **Classification** into 3 classes (Low / Medium / High) obtained by discretizing the score |
| Optional extra | Same pipeline as regression (MAE, RMSE, R2) for comparison |
| Libraries | `requests`, `beautifulsoup4`, `lxml`, `pandas`, `nltk`, `qalsadi` (or `farasapy`), `torch`, `scikit-learn`, `sentence-transformers`, `optuna` (optional), `matplotlib`, `seaborn` |

Repo structure to produce:

```
├── README.md
├── part1_classification.ipynb
├── part2_gpt2.ipynb
├── data/
│   ├── raw_articles.csv
│   └── dataset.csv
└── requirements.txt
```

---

## 1. Data collection (scraping)

### 1.1 Rules
- Use `requests` + `BeautifulSoup` (`lxml` parser). **No Scrapy, no Selenium** (too heavy on Kaggle).
- Set a browser-like `User-Agent` header, `timeout=15`, and `time.sleep(random.uniform(1, 2.5))` between requests.
- Check `https://<site>/robots.txt` before scraping each site and respect it. Skip any site that disallows the paths used.
- Wrap every request in `try/except`, retry up to 2 times, log failures, never crash the whole run.
- Save progress to CSV every 100 articles so a Kaggle crash does not lose everything.
- Deduplicate by URL, then by normalized text.

### 1.2 Websites to scrape

Scrape a **mix of football and non-football sections**. This is essential: if every article is about football, every score will be high and the classifier will have nothing to learn. The score range 0-10 must be well spread.

> **Important:** Website HTML changes often. For each site, Copilot must first open one category page and one article page, inspect the HTML, and find the correct selectors by itself. The selectors below are only hints and must be verified. If a site is JavaScript-rendered or blocks requests, **drop it and use a fallback**, do not switch to Selenium.

| # | Site | Football sections (expected high score) | Non-football sections (expected low/medium score) | Hints |
|---|---|---|---|---|
| 1 | **Hespress** (`hespress.com`) | Sport section (`/sport`) | Politics, economy, society, culture sections | Article text is usually in `<p>` tags inside the main article container; paginate with `/page/N` |
| 2 | **Al Jazeera Arabic** (`aljazeera.net`) | Sport section (`/sport`) | Economy, culture and art, technology, politics | Use the section/archive pages to collect links; article text in `<p>` inside `<article>` / main content div |
| 3 | **BBC Arabic** (`bbc.com/arabic`) | Sport section | Business, technology, science, world | Clean HTML, text in `<p>` blocks inside `<main>` |
| 4 | **Yallakora** (`yallakora.com`) | News / football news pages | (mostly football, use as high-score source) | Check whether the news list is server-rendered |
| 5 | **Elbotola** (`elbotola.com`) | Moroccan football news | none | Moroccan football, good for domain diversity |

**Fallbacks if a site fails:** Goal Arabic (`goal.com/ar`), Sky News Arabia (sport + other sections), Kooora (`kooora.com`, often JS heavy, use last).

### 1.3 Target volume (balanced across sources)

| Group | Target articles |
|---|---|
| Football articles (5 sources) | ~1,400 |
| Non-football articles (Hespress, Al Jazeera, BBC Arabic other sections) | ~800 |
| **Total** | **>= 2,000** (minimum acceptable: 1,000) |

Also deliberately include **borderline articles** (e.g. sports business, athlete news, general sport, Olympics, other sports like tennis or basketball, sports politics) to get medium scores.

### 1.4 What to extract per article
Raw file `data/raw_articles.csv`:

| Column | Description |
|---|---|
| `url` | Article URL |
| `source` | Site name (hespress, aljazeera, bbc, yallakora, elbotola) |
| `section` | Section name the link was found in |
| `title` | Article title |
| `text` | Full body text (paragraphs joined with a space) |
| `date_published` | ISO format `YYYY-MM-DD` if available, else empty |
| `scraped_at` | ISO timestamp |

Filtering rules before saving:
- Drop texts shorter than **50 words** or longer than **2,000 words** (truncate later, not now).
- Drop empty/duplicate texts.
- Remove boilerplate lines inside articles ("اقرأ أيضا", "شارك المقال", "المصدر", "حقوق النشر", etc.).

---

## 2. Building the score (0-10 relevance to football)

The dataset must end up as the format required by the consigne:

| Text | Score |
|---|---|
| نص عربي 1 | 6.0 |
| نص عربي 2 | 7.5 |

**Do NOT derive the score from the `section` column** (that would be label leakage and would make the task trivial). The score is computed from the **text content only**.

### 2.1 Scoring formula
Final score = `10 * clip(0.5 * kw_score + 0.5 * emb_score, 0, 1)`, rounded to 1 decimal.

**(a) Keyword score `kw_score`** (0-1)
1. Build a list of ~60-100 Arabic football keywords (examples: كرة القدم، مباراة، هدف، الدوري، لاعب، مدرب، فريق، منتخب، بطولة، كأس العالم، دوري أبطال أوروبا، الاتحاد الدولي، ركلة جزاء، حارس المرمى، الميركاتو، الانتقالات، تسجيل، الفوز، التعادل، الهزيمة, حكم, ملعب، جماهير، المهاجم، المدافع، الأهلي، الوداد، الرجاء، ريال مدريد، برشلونة ...).
2. Normalize the keywords the same way as the text (see preprocessing, Arabic normalization).
3. `density = (number of keyword hits / number of tokens) * 100`.
4. `kw_score = min(1, density / D)` where `D` is the 95th percentile of the density over the whole corpus (compute it from the data, do not hard-code).

**(b) Embedding score `emb_score`** (0-1)
1. Model: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`.
2. Build a topic vector = mean embedding of ~10 football description sentences written in Arabic.
3. Cosine similarity between each text (first ~256 tokens) and the topic vector.
4. Min-max rescale to 0-1 using the 5th and 95th percentiles (clip outside).

### 2.2 Validation of the score (mandatory, goes in the README)
- Plot the **histogram of scores**. It should be spread out, not concentrated. If heavily skewed, adjust the share of non-football sources or the percentiles.
- Manually label **50 random articles** (0-10 by hand, or low/medium/high) and report the **Spearman correlation** between the manual labels and the computed scores.
- Show 3 examples each of a low (<3.5), medium (3.5-7), and high (>7) scored article.

### 2.3 Discretization into classes
| Class | Score range | Label id |
|---|---|---|
| Low relevance | 0 - 3.5 | 0 |
| Medium relevance | 3.5 - 7 | 1 |
| High relevance | 7 - 10 | 2 |

- Check class balance. If one class has less than 15% of the data, move the thresholds to the **33rd and 66th percentiles** (quantile-based) and document the new thresholds.
- Save final file `data/dataset.csv` with columns: `text`, `score`, `label` (keep `url` and `source` in `raw_articles.csv` only).

---

## 3. Preprocessing pipeline (NLP)

Implement as **one reusable function** `preprocess(text) -> tokens`, applied in this order. For every step, print a **before/after example** on 2 articles.

1. **Cleaning:** remove HTML tags, URLs, emails, emojis, Latin characters, digits (Arabic-Indic and Western), punctuation (Arabic + Latin), repeated whitespace.
2. **Arabic normalization:**
   - remove diacritics (tashkeel) and tatweel (ـ);
   - normalize `أ إ آ` to `ا`, `ى` to `ي`, `ة` to `ه`;
   - collapse repeated letters (3+ same letter to 1).
3. **Tokenization:** `nltk.word_tokenize` (or `str.split` after cleaning). Keep the choice documented.
4. **Stop word removal:** NLTK Arabic stop words list (`nltk.corpus.stopwords.words('arabic')`), normalized the same way as the text, plus a small custom list of news boilerplate words.
5. **Stemming:** NLTK `ISRIStemmer`.
6. **Lemmatization:** `qalsadi` (fallback: `farasapy`). Run it on a **sample first** to check speed, since it can be slow on 2,000+ long texts (use `multiprocessing` or limit text length to the first 300 tokens).
7. **Comparison:** show a table of 10 words: original, stem, lemma. Then choose **one** representation for the models (recommended: lemma, fallback stem if lemmatization is too slow) and justify the choice.
8. **Discretization:** done in section 2.3 (mention it again in the README preprocessing list).

### Vectorization for PyTorch
- **Split first:** stratified 70% train / 15% validation / 15% test (`random_state=42`, stratify on `label`). Save the split indices.
- Build vocabulary from the **train set only** (min frequency 2) with special tokens `<PAD>=0` and `<UNK>=1`.
- Choose `max_len` from the 90th percentile of train token lengths (typically 200-400), truncate/pad to it.
- Build `torch.utils.data.Dataset` + `DataLoader` (batch size 32 by default, `shuffle=True` for train only).
- Optional: initialize embeddings with pretrained Arabic vectors (AraVec / fastText `cc.ar.300`) as an extra experiment, not mandatory.

---

## 4. Models

Write **one generic class** so the 4 models differ only by the recurrent layer:

```
Embedding(vocab, emb_dim, padding_idx=0)
  -> RNN | BiRNN | GRU | LSTM
  -> Dropout
  -> Linear(hidden * (2 if bidirectional else 1), n_classes)
```

| Model | Layer |
|---|---|
| RNN | `nn.RNN(bidirectional=False)` |
| Bidirectional RNN | `nn.RNN(bidirectional=True)`, concatenate the last forward and backward hidden states |
| GRU | `nn.GRU` |
| LSTM | `nn.LSTM` |

Details:
- Use `pack_padded_sequence` (with true lengths) or a mask so padding does not affect the result.
- For the final representation use the last hidden state (for BiRNN concatenate both directions).
- Loss: `CrossEntropyLoss` (add class weights if the classes are imbalanced).
- Optimizer: Adam. **Gradient clipping** `clip_grad_norm_(..., 1.0)` (essential for the plain RNN).
- **Early stopping** on validation loss (patience 5), max 30 epochs, save the best checkpoint to `/kaggle/working/best_<model>.pt`.
- Log per epoch: train loss, val loss, train acc, val acc, time.
- Run on GPU, move data with `.to(device)`.

### Hyperparameter tuning
Use **Optuna** (20 trials per model) or a small random search on the **validation set only**. Search space:

| Hyperparameter | Values |
|---|---|
| Embedding dim | 100, 200, 300 |
| Hidden size | 64, 128, 256 |
| Layers | 1, 2 |
| Dropout | 0.2, 0.3, 0.5 |
| Learning rate | 1e-4 to 1e-2 (log scale) |
| Batch size | 16, 32, 64 |
| max_len | 200, 300, 400 |

Output a table of the best hyperparameters for each of the 4 models. Then **retrain each model once** with its best config and evaluate on the test set.

---

## 5. Evaluation

For each of the 4 models, on the **test set**:
- Accuracy, macro Precision, macro Recall, macro F1 (`sklearn.metrics`), plus `classification_report`.
- Confusion matrix (heatmap).
- Training and validation loss/accuracy curves.
- Training time and number of parameters.

Final **comparison table** (this goes in the README):

| Model | Accuracy | Precision | Recall | F1 | Params | Train time |
|---|---|---|---|---|---|---|
| RNN | | | | | | |
| BiRNN | | | | | | |
| GRU | | | | | | |
| LSTM | | | | | | |

Also:
- **Regression variant (optional bonus):** same architecture with a single output + `MSELoss`, report MAE, RMSE, R2 on the raw score.
- Show 5 misclassified test examples with a short comment.
- Statistical sanity check: run the best config with 3 different seeds and report mean +/- std.

### About BLEU
BLEU is a text-generation metric and does not fit a classifier. In the README, **explain this explicitly** and state that BLEU is computed in **Part 2**, where text is generated. (Optional only if the teacher insists: a clearly labeled side experiment, not part of the main comparison.)

---

## 6. Deliverables for Part 1
- [ ] `part1_classification.ipynb` runs top to bottom on Kaggle without errors.
- [ ] `data/raw_articles.csv` and `data/dataset.csv` (or a public Kaggle Dataset link if too large for GitHub).
- [ ] README section: topic, websites + number of articles per site, scoring method + validation (histogram, Spearman), preprocessing steps with examples, hyperparameter table, comparison table, curves, confusion matrices, analysis (which model is best and why: vanishing gradients in the RNN, gating in GRU/LSTM, context from both directions in BiRNN).
- [ ] Short personal synthesis paragraph.
