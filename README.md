# Automated Essay Scoring 2.0

This repository contains my solution for the Kaggle competition **Automated Essay Scoring 2.0**. The goal is to develop an **automated system** that evaluates essays based on their **content and quality** using advanced machine learning techniques.

Multiple approaches were considered, including:
- **Fine-tuning DeBERTa**, a transformer-based language model.
- **Ensembling multiple DeBERTa models** trained across different folds.
- **Combining LightGBM & XGBoost** with feature engineering, model optimization, and hyperparameter tuning.

The **best Quadratic Weighted Kappa (QWK) score** was achieved using **LightGBM + XGBoost**, with more weight assigned to LightGBM’s predictions. The details of each approach and their results are available in the **Inference** section.

---

## Table of Contents
- [Data Loading & Preprocessing](#data-loading--preprocessing)
- [Feature Engineering](#feature-engineering)
  - [Paragraph-Level Features](#paragraph-level-features)
  - [Sentence-Level Features](#sentence-level-features)
  - [Word-Level Features](#word-level-features)
  - [Spelling & Grammar Features](#spelling--grammar-features)
  - [TF-IDF & Count Vectorizer](#tf-idf--count-vectorizer)
  - [DeBERTa Predictions as Features](#deberta-predictions-as-features)
- [Feature Selection](#feature-selection)
- [Model Building & Training](#model-building--training)
- [Inference](#inference)
- [Results & Performance](#results--performance)
- [Conclusion](#conclusion)
- [Acknowledgements](#acknowledgements)

---

## Data Loading & Preprocessing

This phase involves preparing the dataset for further analysis and model training.

### Steps:
1. **Loading Data**: Essays are loaded using `pandas` and stored in a structured format.
2. **Text Cleaning**: A `dataPreprocessing` function is applied to:
   - Convert text to lowercase.
   - Remove **HTML tags**, **URLs**, **mentions (@user)**, and **numeric values**.
   - Replace consecutive spaces, commas, and periods with single instances.
   - Trim whitespace for a structured output.
3. **Handling Missing Values**: Any missing data is handled to maintain data integrity.

---

## Feature Engineering

Feature engineering plays a crucial role in improving model performance. Multiple **text-based features** were extracted at different levels.

### Paragraph-Level Features
- Number of **paragraphs** per essay.
- **Average paragraph length**.
- **Coherence score** between paragraphs.

### Sentence-Level Features
- Number of **sentences** per essay.
- **Average sentence length**.
- **Sentence complexity**, calculated using grammatical structure.

### Word-Level Features
- **Vocabulary richness**.
- **Word frequency distribution**.
- **Stop word usage analysis**.
- **Sentiment polarity** of the essay.

### Spelling & Grammar Features
- **Spelling errors** detected using **NLTK’s WordNet Lemmatizer** and an English vocabulary set.
- **Grammar mistakes** identified using **Python’s LanguageTool**.
- Count of **adjectives**, **adverbs**, and **grammatical errors** using **POS tagging**.

### TF-IDF & Count Vectorizer
- **TF-IDF Vectorizer**: Assigns weights to words based on frequency & importance.
- **Count Vectorizer**: Captures **word frequency** in essays.

### DeBERTa Predictions as Features
- **DeBERTa Transformer Model** generates predictions for essay scores.
- These predictions are **fed into LightGBM** as additional features.

---

## Feature Selection

To enhance model efficiency, only the **most important features** are selected:
- A **10-fold Stratified CV** trains a **LightGBM regressor** with a **custom QWK objective**.
- **Feature importance scores** are accumulated across folds.
- The **top 13,000 most important features** are retained.

---

## Model Building & Training

Two ensemble models are used: **LightGBM** and **XGBoost**.

### Cross-Validation Strategy:
- **Stratified K-Fold (n_splits=20)** ensures **class balance** across training & validation sets.

### Training Process:
1. **LightGBM Regressor**:
   - Initialized with optimized **hyperparameters** (learning rate, depth, regularization).
   - Trained using **quadratic weighted kappa (QWK) loss**.
   
2. **XGBoost Regressor**:
   - Uses early stopping & QWK-based loss function.
   - Pre-tuned **learning rate, depth, and colsample parameters**.

3. **Model Ensembling**:
   - Final prediction = **76% LightGBM + 24% XGBoost**.
   - Predictions are **adjusted** using a **constant `a`** and **clipped between 1 and 6**.

4. **Performance Metrics**:
   - Evaluated using **F1 Score** and **Cohen's Kappa**.
   - Memory optimized using **garbage collection**.

---

## Inference

### Steps:
1. **Data Transformation**:
   - New essays undergo the same **preprocessing & feature engineering** pipeline.
2. **Prediction**:
   - Trained **LightGBM + XGBoost model** predicts essay scores.
3. **Post-Processing**:
   - Scores **rounded & clipped** to valid range.
4. **Output**:
   - Final predictions are saved for submission.

---

## Results & Performance

| Method  | Description| Leader Board Score (QWK) | Validation Score (QWK) |
| -----------| ----------- | ----------- |----------- |
|<sub>1</sub>|<sub>  DeBERTa only </sub> | <sub>   0.7507 </sub>   | <sub> 0.77816 </sub>|
|<sub>2</sub>|<sub>  DeBERTa only (5 fold CV)</sub> | <sub>   0.7900 </sub>   | <sub> 0.8201 </sub>|
|<sub>3</sub>| <sub> LightGBM + XGBoost + Feature Engineering (Spelling errors, Word count etc.)  </sub> |   <sub> 0.81434 </sub>    | <sub>0.82712 </sub>|
|<sub>4</sub>| <sub> LightGBM + XGBoost + Feature Engineering (DeBERTa predictions, Spelling errors, Word count etc.)  + Vectorization (TF-IDF) </sub>   | <sub>  0.8169  </sub>    |<sub>0.8315</sub>|
|<sub>5</sub>| <sub> LightGBM + XGBoost + Feature Engineering (DeBERTa predictions, Spelling errors, Word count etc.) + Vectorization (TF-IDF)+ Standardscaler </sub> |  <sub> 0.8175 </sub> |<sub>0.8318</sub>|
|<sub>6</sub>|<sub>  LightGBM + XGBoost + Feature Engineering (DeBERTa predictions, Spelling errors, Word count etc.) + Vectorization (TF-IDF, Count)+ Standardscaler </sub>| <sub>  0.8178 </sub>  |<sub> 0.8320 </sub>|
|<sub>7</sub>|<sub>  LightGBM + XGBoost + Feature Engineering (DeBERTa predictions, Spelling errors, Word count, Grammar, Adjectives, Pronouns etc.) + Vectorization (TF-IDF, Count)+ Standardscaler </sub>| <sub>  0.8182 </sub>    |<sub>0.83269</sub>|
|<sub>8</sub>|<sub>  LightGBM(LR 0.1) + XGBoost(LR 0.05<kbd>↓</kbd>) + Feature Engineering (DeBERTa predictions, Spelling errors, Word count, Grammar, Adjectives, Pronouns etc.) + Vectorization (TF-IDF, Count)+ Standardscaler </sub>| <sub>  0.8199 </sub>    |<sub>0.8324</sub>|
|<sub>9</sub>|<sub>  LightGBM(ngram change) + XGBoost(ngram change) + Feature Engineering (DeBERTa predictions, Spelling errors, Word count, Grammar, Adjectives, Pronouns etc.) + Vectorization (TF-IDF, Count)+ Standardscaler </sub>| <sub>  0.8019 </sub>    |<sub>0.8124</sub>|
|<sub>10</sub>|<sub>  LightGBM + XGBoost + Feature Engineering (DeBERTa predictions, Spelling errors, Word count, Grammar, Adjectives, Pronouns etc.) + Vectorization (TF-IDF, Count) + Standardscaler + CV 10<kbd>↓</kbd></sub>|   <sub>0.8165   </sub>  |<sub>0.8122</sub>|
|<sub>11</sub>| <sub> LightGBM(LR 0.1, Max Depth 10)  + XGBoost(LR 0.05, Max Depth 10) + Feature Engineering (DeBERTa predictions, Spelling errors, Word count, Grammar, Adjectives, Pronouns etc.) + Vectorization (TF-IDF, Count)+ Standardscaler + CV 20 <kbd>↑</kbd></sub>|   <sub>0.8224  </sub>   |<sub>0.8275 </sub>|
|<sub>12</sub>| <sub> LightGBM(LR 0.1, Max Depth 8)  + XGBoost(LR 0.05, Max Depth 8) + Feature Engineering (DeBERTa predictions, Spelling errors, Word count, Grammar, Adjectives, Pronouns etc.) + Vectorization (TF-IDF, Count)+ Standardscaler + CV 20 </sub>| <sub>  0.8243 </sub>    |<sub>0.8299</sub>|
---

## Conclusion

This project presents a **comprehensive approach** to automated essay scoring by combining:
- **State-of-the-art transformers (DeBERTa)**
- **Tree-based models (LightGBM & XGBoost)**
- **Advanced feature engineering**
- **Custom optimization strategies for QWK metric**

By leveraging multiple models, ensembling techniques, and rigorous evaluation, this approach achieves **high accuracy & robustness** in essay scoring.

---

## Acknowledgements

Special thanks to **Learning Agency Lab** for providing the dataset and hosting the competition. Additional gratitude to the **open-source community** for developing tools that enabled this work.

🔗 **Competition Link**: [Kaggle: Automated Essay Scoring 2.0](https://www.kaggle.com/competitions/learning-agency-lab-automated-essay-scoring-2)


