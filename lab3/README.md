# Lab 3 - Arabic Text Deep Learning

This lab contains two PyTorch notebooks:

- `part1_classification.ipynb`: Arabic football relevance classification with RNN, BiRNN, GRU, and LSTM.
- `part2_gpt2.ipynb`: fine-tuning AraGPT2 for football text generation.

The detailed Kaggle upload and testing procedure is in [KAGGLE_IMPORT.md](KAGGLE_IMPORT.md).

## Project structure

```text
lab3/
├── README.md
├── KAGGLE_IMPORT.md
├── requirements.txt
├── part1_classification.ipynb
├── part2_gpt2.ipynb
├── data/
│   ├── raw_articles.csv
│   └── dataset.csv
└── docs/
```

## Topic

Football (كرة القدم), with Arabic text and relevance scores from 0 to 10.

## Part 1 - Classification

The notebook:

1. scrapes football and non-football Arabic articles;
2. saves `data/raw_articles.csv`;
3. computes content-based relevance scores without using the source section as a label;
4. preprocesses Arabic text and discretizes scores into three classes;
5. trains RNN, BiRNN, GRU, and LSTM classifiers;
6. reports test metrics and training curves.

BLEU is not appropriate for the main classifier. It belongs in the text-generation evaluation in Part 2.

## Part 2 - Text generation

The notebook loads `aubmindlab/aragpt2-base`, fine-tunes it on football articles, compares base and fine-tuned generation, and computes perplexity and BLEU.

If the model cannot be downloaded, use the notebook's documented fallback or an English GPT-2 model with an English football corpus.

## Personal synthesis

<!-- Add 6-10 lines after running both notebooks: explain what you learned about
Arabic cleaning, PyTorch recurrence, RNN/GRU/LSTM differences, GPT-2 fine-tuning,
BLEU/perplexity, and Kaggle/GPU limitations. -->