# Kaggle Movie Reviews Sentiment Classification

## Overview

This project focuses on classifying movie reviews into sentiment categories (0-4) based on their textual content. The goal is to improve classification accuracy by leveraging advanced text preprocessing, feature engineering, and machine learning models. The dataset used is from Kaggle, and the project addresses class imbalance using oversampling techniques.

## Table of Contents

1. [Introduction](#introduction)
2. [Dataset Description](#dataset-description)
3. [Implementation](#implementation)
   - [Data Loading](#data-loading)
   - [Data Preprocessing](#data-preprocessing)
   - [Feature Engineering](#feature-engineering)
   - [Addressing Class Imbalance](#addressing-class-imbalance)
   - [Model Training and Evaluation](#model-training-and-evaluation)
4. [Observations](#observations)

## Introduction

The project aims to classify movie reviews into five sentiment categories:

- **0: Negative**
- **1: Strong Negative**
- **2: Neutral**
- **3: Positive**
- **4: Strong Positive**

We achieve this by preprocessing the text, engineering advanced features, and applying multiple machine learning models. The approach also includes handling class imbalances using oversampling techniques.

## Dataset Description

### Training Data

- **Size:** 156,060 entries labeled with sentiment values (0-4).
- **Columns:**
  - `Phrase`: The review text.
  - `PhraseId`: Unique identifier for each phrase.
  - `SentenceId`: Identifier for sentences containing the phrase.
  - `Sentiment`: Target labels.

### Test Data

- **Size:** 66,292 entries.
- **No sentiment labels provided.**

### Sentiment Lexicon

- A subjectivity lexicon containing words annotated with positive or negative sentiment used to extract features.

## Implementation

### Data Loading

- **Purpose:** Load training and test datasets for preprocessing and analysis.
- **Approach:** Initially, datasets were loaded without specifying the separator (`sep='\t'`), causing parsing issues. This was corrected by specifying the separator.

### Data Preprocessing

1. **Lemmatization, Tokenization, and Negation Handling:**
   - **Lowercasing:** Standardizes tokens and avoids duplication.
   - **Tokenization:** Breaks text into smaller units (words).
   - **Stopword Removal:** Removes common words that do not add value.
   - **Lemmatization:** Converts words to their base forms.
   - **Negation Handling:** Captures sentiment more effectively by appending negation terms to the next word.

2. **Remove Duplicates:**
   - **Purpose:** Reduce redundancy and ensure unique data points in training.

3. **Preprocess and Store Text:**
   - **Purpose:** Preprocess both datasets consistently for feature extraction.

### Feature Engineering

1. **TF-IDF Features:**
   - **Purpose:** Assigns weights to words based on their importance.
   - **Unigrams and Bigrams:** Captures both single-word and two-word patterns.

2. **Sentiment Lexicon Features:**
   - **Purpose:** Capture sentiment-specific information using positive and negative word counts.

3. **POS Tagging Features:**
   - **Purpose:** Include syntactic features (e.g., nouns, verbs, adjectives) for better classification.

4. **Combine Features:**
   - **Purpose:** Unify different features into a comprehensive representation for model training.

### Addressing Class Imbalance

- **Purpose:** Balance minority classes using oversampling.
- **Approach:** Initially, class imbalance led to poor performance on underrepresented classes. This was addressed using SMOTE (Synthetic Minority Over-sampling Technique).

### Model Training and Evaluation

- **Results:**
  - **Naive Bayes (TF-IDF Only):**
    - F1 Scores: [0.442, 0.449, 0.446]
    - Mean F1 Score: 0.446
  - **Logistic Regression (Combined Features):**
    - F1 Scores: [0.552, 0.555, 0.558]
    - Mean F1 Score: 0.555
  - **Random Forest (Tuned):**
    - Best Parameters: `n_estimators=200`, `max_depth=None`

## Observations

- **Combined features** outperformed single-feature models.
- **SMOTE** improved performance on minority classes.
- **Logistic Regression** demonstrated the best results, achieving an F1 score of 0.555.

## Conclusion

This project successfully classifies movie reviews into sentiment categories by leveraging advanced text preprocessing, feature engineering, and machine learning models. The use of combined features and oversampling techniques significantly improved the model's performance, especially on minority classes.
