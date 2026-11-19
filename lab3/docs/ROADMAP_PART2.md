# ROADMAP, Lab 3, Part 2: Text Generation with GPT-2 (Fine-tuning)

> **Instructions for the AI assistant (Copilot):** Follow this roadmap step by step, in order. Work in a Kaggle notebook (`part2_gpt2.ipynb`) with **Internet ON** and **GPU (T4) ON**. Use PyTorch and Hugging Face `transformers`. After each step, print a short result (shapes, loss, sample output) before moving on. Fix seeds to 42.

---

## 0. Global decisions (do not change)

| Item | Decision |
|---|---|
| Topic | **Football (كرة القدم)**, same as Part 1 |
| Language | Arabic |
| Model | `aubmindlab/aragpt2-base` (GPT-2 pretrained on Arabic). The original English `gpt2` handles Arabic very badly. |
| Fallback | If `aragpt2-base` cannot be loaded, use `gpt2` on an **English** football corpus instead, and document it |
| Library | `transformers` (the `pytorch-transformers` package in the consigne is deprecated; `transformers` is its successor with the same API), `datasets`, `torch`, `arabert`, `nltk`, `sacrebleu`, `evaluate` |
| Install | `pip install transformers datasets arabert sacrebleu evaluate` |

---

## 1. Data for fine-tuning

### 1.1 Source
Reuse the Part 1 corpus (`data/raw_articles.csv`). Add it as a Kaggle Dataset input or re-upload it.

### 1.2 Selection
- Keep only **football-related articles**: those with `score >= 6` (or label High/Medium-high). This keeps the generated text on topic.
- Use the **raw cleaned text**, NOT the stemmed/lemmatized/stop-word-removed text from Part 1 (GPT-2 needs natural, complete sentences).
- Light cleaning only: remove URLs, HTML leftovers, boilerplate lines, extra whitespace. Keep punctuation and diacritic-free normal Arabic.
- Target size: **1,000-1,500 articles**. Truncate each article to its first ~250 words.

### 1.3 Split
90% train / 10% held-out (validation + references for BLEU). Save the split. The held-out set must **never** be used for training.

### 1.4 Preprocessing for AraGPT2
- Apply `arabert.preprocess.ArabertPreprocessor(model_name="aubmindlab/aragpt2-base")` to every text before tokenizing (required by this model family).
- Add the end-of-text token (`tokenizer.eos_token`) at the end of each article so the model learns where to stop.

---

## 2. Setup

1. Load tokenizer and model:
   - `AutoTokenizer.from_pretrained("aubmindlab/aragpt2-base")`
   - `AutoModelForCausalLM.from_pretrained("aubmindlab/aragpt2-base")`
2. Set `tokenizer.pad_token = tokenizer.eos_token` if no pad token exists.
3. Move the model to `cuda`.
4. **Save a copy of the base model behavior first**: generate outputs for the test prompts (section 5) **before** fine-tuning, to compare later.

---

## 3. Fine-tuning

### 3.1 Tokenization and dataset
- Tokenize all train texts, concatenate them and split into blocks of **`block_size = 256`** tokens (or truncate each article to 256 tokens if concatenation is too complex).
- Build a `torch.utils.data.Dataset` returning `input_ids`, `attention_mask`, and `labels` (labels = `input_ids`, the model shifts them internally).
- Use `DataCollatorForLanguageModeling(tokenizer, mlm=False)` if using the `Trainer`.

### 3.2 Training configuration

| Parameter | Value |
|---|---|
| Epochs | 3 to 5 |
| Batch size | 4 per device, gradient accumulation 4 (effective 16) |
| Learning rate | 5e-5 |
| Optimizer | AdamW, weight decay 0.01 |
| Scheduler | Linear warmup (10% of steps) then linear decay |
| Precision | `fp16=True` |
| Gradient clipping | 1.0 |
| Evaluation | After every epoch on the held-out set |

Choose **one** approach and keep it simple:
- **A (recommended for learning):** a plain PyTorch loop (AdamW + `get_linear_schedule_with_warmup`), like the tutorial in the consigne.
- **B (faster to write):** Hugging Face `Trainer` with `TrainingArguments`.

If the GPU runs out of memory: reduce `block_size` to 128, batch size to 2, or enable `gradient_checkpointing`.

### 3.3 Tracking
- Log train loss and validation loss every epoch (and every N steps for the training curve).
- Compute **perplexity** = `exp(validation loss)` before fine-tuning and after fine-tuning.
- Plot the loss curves.
- Save the fine-tuned model and tokenizer to `/kaggle/working/aragpt2-football/` with `save_pretrained` (also download it or save it as a Kaggle output/dataset).

---

## 4. Generation experiments

### 4.1 Prompt sentences (the consigne asks: generate a paragraph from a given sentence)
Use at least these **8 Arabic prompts** (written in natural news style, adapt them if needed):

1. `فاز المنتخب المغربي على`
2. `أعلن النادي رسميا عن التعاقد مع`
3. `في مباراة قوية انتهت بالتعادل بين`
4. `صرح المدرب بعد المباراة أن`
5. `تأهل الفريق إلى نهائي دوري أبطال`
6. `تصدر الفريق جدول الترتيب بعد`
7. `سجل اللاعب هدفا رائعا في الدقيقة`
8. `استعدادا لكأس العالم يخوض المنتخب`

### 4.2 Decoding strategies
Compare on the **same prompts**:

| Strategy | Settings |
|---|---|
| Greedy | `do_sample=False` |
| Beam search | `num_beams=5`, `no_repeat_ngram_size=3`, `early_stopping=True` |
| Top-k sampling | `do_sample=True`, `top_k=50`, `temperature=0.8` |
| Top-p (nucleus) | `do_sample=True`, `top_p=0.92`, `temperature=0.8` |

For all: `max_length=150` to 200 tokens, `repetition_penalty=1.2`, `no_repeat_ngram_size=3`, `pad_token_id=tokenizer.eos_token_id`.

### 4.3 Before vs after
For each prompt, show: **base model output** vs **fine-tuned model output** (using the best decoding strategy). Put them in a table in the notebook and the README.

---

## 5. Evaluation

### 5.1 Perplexity
- Report perplexity on the held-out set: base model vs fine-tuned model (the fine-tuned one should be clearly lower).

### 5.2 BLEU (this is where BLEU belongs)
1. Take **50 held-out articles**. For each one, use its **first sentence (or first ~10 words)** as the prompt, and its **next ~50 words** as the reference.
2. Generate a continuation of about 50 words with the base model and with the fine-tuned model.
3. Compute BLEU with `sacrebleu` (corpus BLEU) and/or `nltk.translate.bleu_score` (sentence BLEU with `SmoothingFunction().method1`), BLEU-1 to BLEU-4.
4. Report in a table: **base vs fine-tuned**, for each decoding strategy if time allows.
5. Explain in the README that BLEU scores are naturally low for open-ended generation (many valid continuations), and that the interesting part is the **relative improvement** after fine-tuning.

Optional extra metrics: ROUGE-L, distinct-n (diversity), average length.

### 5.3 Qualitative evaluation
- Show 5 to 8 generated paragraphs with a short comment each (fluency, relevance to football, factual hallucinations, repetition).
- Note honestly the limits: made-up scores, wrong names/facts, repetitions.

---

## 6. Optional improvements (only if time remains)
- Conditional generation using the class from Part 1: prepend a control token (e.g. `[HIGH]`) to the training texts.
- Compare `aragpt2-base` with another Arabic model, or compare 3 vs 5 epochs.
- Build a tiny function `generate_paragraph(sentence)` that takes a sentence and returns a paragraph, as a demo for the end of the notebook.

---

## 7. Deliverables for Part 2
- [ ] `part2_gpt2.ipynb` runs top to bottom on Kaggle without errors.
- [ ] The fine-tuned model saved (Kaggle output) or a link to it; do not push model weights (500 MB) to GitHub.
- [ ] README section: model choice and why (Arabic GPT-2 vs English GPT-2), dataset used for fine-tuning, hyperparameters, loss curves, perplexity before/after, BLEU table, before/after generation examples, decoding comparison, limitations.
- [ ] Short personal synthesis: what you learned about PyTorch, RNN vs GRU vs LSTM (Part 1), Arabic NLP difficulties, and fine-tuning a pretrained transformer.

---

## 8. Final GitHub checklist
- [ ] Repo contains both notebooks, `data/`, `requirements.txt`, `README.md`.
- [ ] README has links to the Kaggle notebooks and the Kaggle dataset.
- [ ] Notebooks are saved **with outputs visible**.
- [ ] Final synthesis paragraph included.
