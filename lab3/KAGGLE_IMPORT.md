# Import Lab 3 into Kaggle

This guide explains how to test the notebooks in this directory on Kaggle.
The notebooks are designed to run in Kaggle rather than on a local machine.

## 1. Push the repository to GitHub

From the repository root:

```bash
git add lab3/
git commit -m "Add Lab 3 notebooks and Kaggle guide"
git push origin main
```

Do not commit trained model weights. They are large and should remain a Kaggle
output or Kaggle Dataset.

## 2. Create a Kaggle account and enable Internet

1. Open <https://www.kaggle.com/>.
2. Sign in.
3. Open **Code > New Notebook**.
4. In the right sidebar, open **Session options**.
5. Set **Internet** to **On**.
6. Set **Accelerator** to **GPU T4** for training.

The first notebook needs Internet while scraping. The second also needs Internet
to download the Hugging Face model.

## 3. Upload the Lab 3 repository

For a notebook with no Kaggle Dataset input yet:

1. Create a GitHub repository containing `lab3/`.
2. In Kaggle, open **Code > New Notebook**.
3. In a first code cell, run:

```python
!git clone https://github.com/OWNER/REPOSITORY.git
%cd REPOSITORY/lab3
```

Replace `OWNER` and `REPOSITORY` with the real GitHub names.

Then upload `part1_classification.ipynb`:

1. In Kaggle Notebook, select **File > Import Notebook**.
2. Upload `lab3/part1_classification.ipynb`.
3. Enable Internet and GPU.
4. Run every cell.

Alternatively, clone GitHub in the first Kaggle cell and open the notebook file
from the repository if Kaggle's Import Notebook interface is unavailable.

## 4. Upload the generated dataset

After the first notebook finishes:

1. Open **Save Version** or inspect `/kaggle/working`.
2. Download `data/raw_articles.csv` and `data/dataset.csv`, or save
   `/kaggle/working` as a Kaggle Dataset.

To create a Kaggle Dataset:

1. Open **Data > New Dataset**.
2. Upload the local `lab3/data` folder.
3. Name it `lab3-arabic-football-data`.
4. Open **Part 2**.
5. Select **Add Input > Kaggle Dataset** and add the uploaded dataset.

The notebook checks these paths automatically:

- `/kaggle/input/lab3-arabic-football-data/raw_articles.csv`
- `/kaggle/input/lab3-arabic-football-data/dataset.csv`
- `/kaggle/working/data/raw_articles.csv`
- `/kaggle/working/data/dataset.csv`

## 5. Install dependencies

Run the setup cell in each notebook. To install all dependencies manually:

```python
%pip install -r /kaggle/working/REPOSITORY/lab3/requirements.txt
```

Change `REPOSITORY` to the cloned repository folder name.

## 6. Test Part 1

In `part1_classification.ipynb`:

1. Set Internet **On**.
2. Set GPU **T4**.
3. Run all cells.
4. Verify that `/kaggle/working/data/raw_articles.csv` exists.
5. Verify that `/kaggle/working/data/dataset.csv` exists.
6. Confirm that at least 1,000 articles are collected.
7. Download or save the notebook outputs.

The notebook saves the best model checkpoints as:

```text
/kaggle/working/best_rnn.pt
/kaggle/working/best_birnn.pt
/kaggle/working/best_gru.pt
/kaggle/working/best_lstm.pt
```

## 7. Test Part 2

Add the Lab 3 dataset as Kaggle input, then:

1. Open `part2_gpt2.ipynb`.
2. Set Internet **On**.
3. Set GPU **T4**.
4. Run all cells.
5. The notebook saves the fine-tuned model in
   `/kaggle/working/aragpt2-football/`.

Do not select **Save > Commit** until all cells finish. Kaggle may stop the
session if the timeout is reached.

## 8. Save the final work

For each notebook:

1. Run **Save Version**.
2. Keep **Save & Run All** selected.
3. Set **Notebook visibility** as requested.
4. Copy the Kaggle notebook URL.
5. Add the URLs to `lab3/README.md`.

At minimum, the GitHub repository should contain:

- `lab3/part1_classification.ipynb`
- `lab3/part2_gpt2.ipynb`
- `lab3/README.md`
- `lab3/KAGGLE_IMPORT.md`
- `lab3/requirements.txt`
- `lab3/data/raw_articles.csv`
- `lab3/data/dataset.csv`

Notebook results, logs, confusion matrices, and the fine-tuned model outputs
should remain visible in Kaggle.
