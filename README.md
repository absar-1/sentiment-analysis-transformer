# SentimentScope: Sentiment Analysis with a Transformer Built From Scratch

## Overview

This project implements a **transformer-based neural network from scratch** using PyTorch to classify IMDB movie reviews as **positive** or **negative**. Rather than fine-tuning an existing pretrained model, this project builds the full transformer architecture — token embeddings, positional embeddings, multi-head self-attention, and a classification head — from the ground up, to demonstrate a deep understanding of how transformers actually work internally.

The project was completed as part of Udacity's AI Programming curriculum, framed around a scenario where the model helps a fictional entertainment company (Cinescope) better understand user sentiment to improve its recommendation system.

## What This Project Does

Given a movie review as raw text, the model predicts whether the sentiment expressed is **positive (1)** or **negative (0)**. This is a standard binary text classification task, commonly used in real-world applications like customer feedback analysis, social media monitoring, and content recommendation systems.

Example:

| Review | Predicted Sentiment |
|---|---|
| "The movie was a rollercoaster of emotions, and I loved every moment of it!" | Positive |
| "The plot was predictable, and the acting was subpar. A waste of time." | Negative |

## Project Pipeline

The notebook is organized into the following stages:

1. **Load, Explore, and Prepare the Dataset** — reading raw review text files, visualizing label distribution and review length, and splitting the training data into training/validation subsets
2. **Tokenization** — using Hugging Face's `bert-base-uncased` tokenizer to convert raw text into subword tokens and numerical IDs
3. **Custom PyTorch Dataset and DataLoader** — building an `IMDBDataset` class to handle tokenization and batching efficiently
4. **Transformer Architecture (DemoGPT)** — a custom transformer model including:
   - Token and positional embeddings
   - Multi-head self-attention blocks
   - Mean pooling across token representations
   - A linear classification head producing 2-class logits
5. **Training Loop** — training the model using cross-entropy loss and the AdamW optimizer, with validation accuracy tracked after each epoch
6. **Evaluation** — measuring final accuracy on a held-out test set

## Model Architecture

The custom transformer (`DemoGPT`) consists of:

- **Token Embedding Layer** — maps each token ID to a dense vector representation
- **Positional Embedding Layer** — encodes the position of each token in the sequence
- **Stacked Transformer Blocks** — each containing multi-head self-attention and dropout for regularization
- **Layer Normalization** — stabilizes training across layers
- **Mean Pooling** — condenses token-level representations into a single vector per review
- **Classification Head** — a linear layer mapping the pooled representation to 2 output classes (positive/negative)

## Dataset

This project uses the **Large Movie Review Dataset (IMDB)**, created by Maas et al. (2011), containing 50,000 highly polar movie reviews split evenly between training and testing sets.

- **Source:** [https://ai.stanford.edu/~amaas/data/sentiment/](https://ai.stanford.edu/~amaas/data/sentiment/)
- **Training set:** 25,000 labeled reviews (12,500 positive, 12,500 negative)
- **Test set:** 25,000 labeled reviews (12,500 positive, 12,500 negative)
- **Unsupervised set:** additional unlabeled reviews (not used in this project)

### Downloading the Dataset

The dataset is not included in this repository due to its size. To run this project yourself:

```bash
wget https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
tar -xzvf aclImdb_v1.tar.gz
```

Make sure the extracted `aclImdb/` folder sits in the same directory as the notebook, with the following structure:

aclImdb/
├── train/
│ ├── pos/ # Positive reviews for training
│ ├── neg/ # Negative reviews for training
│ ├── unsup/ # Unsupervised data (not used)
├── test/
│ ├── pos/ # Positive reviews for testing
│ ├── neg/ # Negative reviews for testing


## Getting Started

### Prerequisites

- Python 3.8+
- Jupyter Notebook
- A GPU is recommended (though not required) for reasonable training times

### Installation

Clone this repository:

```bash
git clone https://github.com/absar-1/sentiment-analysis-transformergit
cd your-repo-name
```

Install the required dependencies:

```bash
pip install torch transformers pandas matplotlib numpy
```

Download the dataset following the instructions above.

### Running the Project

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Open `SentimentScope_starter.ipynb` and run the cells sequentially from top to bottom.

## Results

The model was trained on 22,500 reviews, validated on 2,500 reviews, and evaluated on 25,000 held-out test reviews. The project's target was to achieve above **75% test accuracy**, which the trained model met.

## Key Concepts Demonstrated

- Tokenization and subword text encoding
- Self-attention and multi-head attention mechanisms
- Positional embeddings in transformers
- Adapting a generative (GPT-style) transformer architecture for a classification task
- Mean pooling for sequence-to-vector representation
- Training loops involving forward passes, loss computation, backpropagation, and gradient descent
- Model evaluation using accuracy metrics on held-out validation and test sets

## Project Structure

.
├── SentimentScope_starter.ipynb # Main notebook containing the full project
├── README.md # This file


## Acknowledgments

- Dataset: Maas, A. L., Daly, R. E., Pham, P. T., Huang, D., Ng, A. Y., & Potts, C. (2011). *Learning Word Vectors for Sentiment Analysis*. The 49th Annual Meeting of the Association for Computational Linguistics (ACL 2011).
- Tokenizer: [`bert-base-uncased`](https://huggingface.co/bert-base-uncased) via Hugging Face Transformers
- Project scaffolding provided by Udacity as part of the AI Programming with Python / Deep Learning curriculum
