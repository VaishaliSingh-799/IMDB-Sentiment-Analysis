# IMDB-Sentiment-Analysis
A binary sentiment classifier that predicts whether a movie review is **positive** or **negative**. The project walks through the full ML cycle: data preparation, text cleaning, model comparison, evaluation, error analysis, and saving a model for inference.

 Overview

Task: binary text classification (positive / negative)
Dataset:[IMDB Dataset of 50K Movie Reviews]("C:\Users\dell\Downloads\IMDB Dataset.csv.zip"): 49,582 reviews after removing 418 duplicates (24,884 positive, 24,698 negative)
Frameworks: TensorFlow / Keras, NLTK, scikit-learn, pandas
Models compared:average-pooling baseline vs. Bidirectional LSTM

Approach

1. Load and dedupe: drop missing values and duplicate reviews so the same review cannot appear in both train and test.
2. Clean text: strip HTML tags, lowercase, keep letters and apostrophes, remove stopwords **except negations** (so "not good" stays negative), and lemmatize.
3. Split: stratified 60% train / 20% validation / 20% test (29,749 / 9,916 / 9,917 reviews), with a fixed random seed.
4. Vectorize: a Keras `TextVectorization` layer (vocabulary of 20,000 tokens, sequences of 200 tokens), fitted on the training set only and built into the model.
5. Train two models:
   - Pooling baseline: Embedding → GlobalAveragePooling → Dense
   - BiLSTM: Embedding → Bidirectional LSTM → Dense
   - Both use dropout, L2 regularization, early stopping, and learning-rate reduction.
6. Select and evaluate: the model is chosen on the **validation** set, then evaluated **once** on the held-out test set.
7. Error analysis: inspect the most confident mistakes.
8. Save and reload: export to `model.keras` and run inference on raw text.

## Results

Both models were compared on the validation set, and the better one was evaluated once on the held-out test set.

| Model | Validation accuracy | Validation loss |
|---|---|---|
| **Pooling baseline (selected)** | **89.1%** | 0.298 |
| Bidirectional LSTM | 88.5% | 0.290 |

The simple pooling baseline matched the BiLSTM on accuracy while training much faster (about 10 s per epoch vs. about 125 s on the same GPU). The BiLSTM overfit quickly, reaching about 96% training accuracy by epoch 3 while validation accuracy stayed near 88%, so early stopping ended its training after 3 epochs.

**Final test performance (pooling baseline, 9,917 reviews):**

| Class | Precision | Recall | F1-score |
|---|---|---|---|
| Negative | 0.89 | 0.90 | 0.89 |
| Positive | 0.90 | 0.89 | 0.89 |
| **Accuracy** | | | **0.89** |

The model is balanced across classes: 4,433 negative and 4,441 positive reviews were classified correctly, with 507 and 536 errors respectively.

![Confusion matrix]("C:\Users\dell\Downloads\IMDB_sentiment_analysis.confusion matrix.png")
![Training curves]("C:\Users\dell\Downloads\IMDB_sentiment_analysis.graph.png")

### Error analysis

The most confident mistakes show where the model struggles:

- **Reviews of "so bad it's good" films** use negative words in an affectionate way (e.g., "chintzy", "schlocky", "ludicrous") and are predicted negative despite a positive label.
- **Plot-heavy reviews** spend most of the text summarizing the story and only briefly state an opinion, which can mislead a bag-of-words style model.
- **Short or mild reviews** get scores close to 0.5 (e.g., 0.504 and 0.498 for two example sentences), meaning the model is unsure and these predictions should not be trusted.

## Project structure

```
sentiment-analysis-imdb/
├── IMDB_sentiment_analysis.ipynb   # full pipeline: data -> model -> evaluation -> inference
├── images/                         # confusion matrix and training curves used in this README
├── requirements.txt
├── .gitignore
└── README.md
```

The dataset and the trained model file are not committed (see `.gitignore`).

## Getting started

### Option 1: Google Colab (easiest)

1. Upload `IMDB_sentiment_analysis.ipynb` to [Google Colab](https://colab.research.google.com).
2. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) and upload `IMDB Dataset.csv.zip` to the Colab file panel.
3. Optional: enable a GPU under **Runtime → Change runtime type**.
4. Choose **Runtime → Run all**.

### Option 2: Run locally

```bash
git clone https://github.com/<Vaishali-799>/sentiment-analysis-imdb.git
cd sentiment-analysis-imdb

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install jupyter

# Place "IMDB Dataset.csv.zip" in the project folder, then:
jupyter notebook IMDB_sentiment_analysis.ipynb
```

## Using the trained model

After running the notebook, the model is saved as `model.keras`. The notebook includes a `predict_sentiment()` helper:

```python
predict_sentiment("Not good at all, a complete waste of time.")
# [{'text': 'Not good at all, a complete waste of time.', 'sentiment': 'negative', 'score': 0.134}]
```

The vectorizer is part of the saved model, so only the `clean_text()` function from the notebook is needed before calling `model.predict`.

## Limitations

- Trained only on movie reviews, so performance may drop on other domains (products, restaurants, social media).
- Sarcasm, ironic praise of bad movies, mixed opinions, and long reviews that exceed 200 tokens are harder cases.
- The model is not confident on short or mildly worded text, where scores land near 0.5.
- Binary labels only; there is no neutral class.

## Future work

- Fine-tune a pretrained transformer (e.g., DistilBERT) and compare it with these models.
- Serve the model through a FastAPI endpoint and containerize it with Docker.
- Add a simple Streamlit demo app.
- Extend to aspect-based sentiment (per feature of a product or movie).

## License

This project is released under the MIT License.

