# Text Classification — IMDB Sentiment Analysis

Sentiment classification on the IMDB movie reviews dataset, comparing classical machine learning against deep learning approaches for binary text classification (positive/negative).

## Overview

This project implements and compares two distinct approaches to sentiment classification:

1. **Classical ML pipeline**: TF-IDF vectorization with Logistic Regression, Naive Bayes, and Linear SVM
2. **Deep learning pipeline**: Pre-trained GloVe word embeddings with a PyTorch feedforward neural network

## Dataset

- **Source**: [IMDB Movie Reviews](https://huggingface.co/datasets/stanfordnlp/imdb) (via Hugging Face Datasets)
- 50,000 movie reviews total — 25,000 train / 25,000 test
- Balanced binary labels: positive (1) / negative (0)

## Pipeline

1. Data loading and exploratory data analysis (label distribution, review length distribution)
2. Text preprocessing (HTML tag removal, lowercasing, punctuation removal)
3. Feature extraction:
   - TF-IDF vectorization (unigrams + bigrams, 10,000 features)
   - Averaged GloVe embeddings (100-dimensional, `glove-wiki-gigaword-100`)
4. Model training and evaluation

## Results

| Model | Feature Representation | Accuracy |
|---|---|---|
| Logistic Regression | TF-IDF | 89.38% |
| Linear SVM | TF-IDF | 88.41% |
| Naive Bayes | TF-IDF | 86.17% |
| Feedforward Neural Network | GloVe (averaged) | 79.22% |

TF-IDF with Logistic Regression outperformed the GloVe + neural network approach, primarily due to information loss from averaging word vectors, which discards word order and dilutes the contribution of strong sentiment-bearing words.

## Tech Stack

- Python 3.12
- scikit-learn — TF-IDF vectorization, classical ML models, evaluation metrics
- PyTorch — neural network implementation
- Gensim — pre-trained GloVe embedding loader
- Hugging Face `datasets` — data loading
- pandas, numpy, matplotlib, seaborn — data handling and visualization

## Project Structure

\`\`\`
text_classification.ipynb   # Full pipeline: EDA, preprocessing, feature extraction, training, evaluation
requirements.txt            # Python dependencies
\`\`\`

## Setup

\`\`\`bash
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
\`\`\`

## Future Work

- LSTM/GRU-based sequence modeling to preserve word order
- Fine-tuned transformer models (BERT/DistilBERT)
