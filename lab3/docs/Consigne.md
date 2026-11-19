# Lab 3

## Objective

The main purpose behind this lab is to get familiar with PyTorch, to build deep neural network architecture for Natural Language Processing by using Sequence Models.

---

## Part 1 – Classification Task

1. **Data Collection**
   - Using scraping libraries (Scrapy / BeautifulSoup), collect text data from several Arabic websites concerning one topic.
   - Prepare your dataset as follows:

     | Text | Score |
     |------|-------|
     | Text 1 (Arabic) | 6 |
     | Text 2 (Arabic) | 7.5 |

   > The score represents the relevance of each text (range: 0–10).

2. **Preprocessing**
   - Establish a preprocessing NLP pipeline (tokenization, stemming, lemmatization, stop words, discretization, etc.) of the collected dataset.

3. **Model Training**
   - Train models using **RNN**, **Bidirectional RNN**, **GRU**, and **LSTM** architectures.
   - Tune hyper-parameters to get the best performance.

4. **Evaluation**
   - Evaluate the four models using standard metrics and others like **BLEU score**.

---

## Part 2 – Transformer (Text Generation)

Install `pytorch-transformers`, then load the **GPT-2** pre-trained model.

1. **Fine-tuning**
   - Fine-tune the pre-trained model (GPT-2) on a customized dataset (you can generate your own).

2. **Generation**
   - Generate a new paragraph according to a given sentence.   

> You can follow this tutorial : https://gist.github.com/mf1024/3df214d2f17f3dcc56450ddf0d5a4cd7

## Notes :
• At the end each student must give a brief synthesis about what he has learn during
the proposed lab.

• Push the work in the Github repository and write a brief report in Github readme
file.

Tools:
Kaggle, github.

> i will test the nootebooks u gonna create here on my kaggle enviremet not on my local envirenment  