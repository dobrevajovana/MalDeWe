
# MalDeWe: Malware Website Detector using Natural Language Processing

MalDeWe is a Natural Language Processing (NLP) based malware detection model that leverages the RoBERTa transformer model to detect malicious websites by analyzing their JavaScript content. This approach is highly effective in identifying malware with a **Roc AUC score of 0.95** when trained on a balanced dataset.

This repository contains the code for training, evaluating, and using the MalDeWe model.

## Features

- **Malware Detection**: Detects whether a website is malicious based on its JavaScript code.
- **RoBERTa Model**: Utilizes a pre-trained RoBERTa model fine-tuned on JavaScript code.
- **Balanced Dataset**: Handles the issue of dataset imbalance by training on a balanced dataset of malicious and benign websites.
- **High Performance**: Achieves high accuracy in detecting malicious sites with precision and recall metrics.
  
## Model Architecture

The MalDeWe model architecture is based on the RoBERTa transformer model, fine-tuned to classify websites as malicious or benign. The model processes JavaScript content as input and generates sentence embeddings, which are then used for binary classification.

![Model Architecture](model_architecture.png)

## Dataset

The dataset used for training and evaluation is publicly available on [Mendeley](https://data.mendeley.com/datasets/gdx3pkwp47/2). It includes:

- **URL**: The URL of the website
- **JavaScript Content**: The raw JavaScript code from the website
- **Labels**: Binary labels indicating whether the site is malicious (1) or benign (0)

For training, the dataset was balanced to ensure an equal number of benign and malicious websites, improving the model's ability to correctly identify malware.

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch
- Transformers (Hugging Face)
- Scikit-learn

## Results

MalDeWe achieves the following performance metrics when evaluated on the test dataset:

- **F1 Score**: 0.96
- **Accuracy**: 95%
- **Roc Auc**: 0.95
- **Recall**: 0.95

## Find the paper
https://ieeexplore.ieee.org/document/9799038

## Acknowledgments

This research was supported by the Faculty of Computer Science & Engineering at Ss. Cyril and Methodius University, Skopje.
