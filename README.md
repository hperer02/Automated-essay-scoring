# Automated Essay Scoring 2.0

This repository contains my solution for the Kaggle competition Automated Essay Scoring 2.0. The goal of this project is to develop an automated system capable of scoring essays based on their content and quality using machine learning techniques.

## Table of Contents
- [Data Loading & Preprocessing](#data-loading--preprocessing)
- [Feature Engineering](#feature-engineering)
  - [Paragraph Features](#paragraph-features)
  - [Sentence Features](#sentence-features)
  - [Word Features](#word-features)
  - [Vectorizer](#vectorizer)
  - [DeBERTa Predictions to LGBM as Features](#deberta-predictions-to-lgbm-as-features)
- [Feature Selection](#feature-selection)
- [Model Building & Training](#model-building--training)
- [Inference](#inference)

## Data Loading & Preprocessing

In this section, I loaded the dataset provided by Kaggle and performed initial preprocessing steps to prepare the data for feature engineering and model training. 

### Steps:
1. **Loading Data**: Imported the dataset using Pandas.
2. **Handling Missing Values**: Checked for and handled any missing values to ensure data integrity.
3. **Data Cleaning**: Removed any irrelevant information and standardized the format of the essays.

## Feature Engineering

Feature engineering is crucial for improving the performance of the model. Various features were extracted from the essays to capture different aspects of the text.

### Paragraph Features

Extracted features related to the structure of the paragraphs within the essays, such as the number of paragraphs, average length of paragraphs, and coherence between paragraphs.

### Sentence Features

Extracted features related to sentence structure, such as the number of sentences, average sentence length, grammatical correctness, and complexity of sentences.

### Word Features

Extracted features related to the words used in the essays, such as vocabulary richness, usage of unique words, frequency of stop words, and sentiment analysis.

### Vectorizer

Converted the text data into numerical format using different vectorization techniques to capture the importance and frequency of words.

#### TF-IDF Vectorizer

Used TF-IDF (Term Frequency-Inverse Document Frequency) Vectorizer to assign weights to words based on their frequency and importance across the corpus. This technique helps in identifying significant words that contribute to the meaning of the essays.

#### Count Vectorizer

Used Count Vectorizer to convert the text data into a matrix of token counts. This technique helps in capturing the frequency of words in the essays, providing a simple yet effective way to quantify textual data.

### DeBERTa Predictions to LGBM as Features

Utilized the DeBERTa (Decoding-enhanced BERT with disentangled attention) model to generate predictions for the essays. These predictions were then used as additional features in the LightGBM (LGBM) model to improve its performance.

## Feature Selection

Performed feature selection to identify the most significant features that contribute to the essay scores. This step helps in reducing the dimensionality of the data and improving the efficiency of the model.

## Model Building & Training

In this section, I built and trained the machine learning models using the selected features.

### Steps:
1. **Model Selection**: Chose LightGBM (LGBM) as the primary model due to its efficiency and performance.
2. **Training**: Trained the LGBM model on the training dataset with the selected features.
3. **Validation**: Validated the model using cross-validation techniques to ensure its robustness and generalizability.

## Inference

During the inference phase, the trained model was used to score new, unseen essays. The process involved transforming the new essays using the same feature engineering pipeline and then predicting the scores using the trained model.

| Method  | Leader Board Score (QWK) | Validation Score (QWK) |
| ----------- | ----------- |----------- |
| DeBERTa only |       ||
|    |         ||

### Steps:
1. **Data Transformation**: Transformed the new essays using the same preprocessing and feature engineering steps as the training data.
2. **Prediction**: Used the trained LGBM model to predict the scores for the new essays.
3. **Output**: Generated the final scores and saved the results in the required format for submission.

## Conclusion

This project showcases a comprehensive approach to automated essay scoring using advanced machine learning techniques. By leveraging powerful models like DeBERTa and LGBM, and performing thorough feature engineering and selection, the solution aims to achieve high accuracy and robustness in scoring essays.

## Acknowledgements

Special thanks to Learning Agency Lab for providing the dataset and hosting the competition. Also, gratitude to the open-source community for providing the tools and libraries that made this project possible.

**Competition link** - https://www.kaggle.com/competitions/learning-agency-lab-automated-essay-scoring-2
