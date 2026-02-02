# Text2Meta — Textual Meta-Feature Extraction

**Text2Meta** is a modular framework for extracting **meta-features from textual datasets**, designed for **Meta-Learning**, **AutoML**, and **dataset characterization** tasks.

It supports multiple groups of meta-features, including:

- General (Simple)
- Statistical
- Information-Theoretic
- Model-Based
- Landmarking
- Complexity

This README documents **all currently available meta-features**.

---

## 🚀 Quick Usage

```python
from text2meta import Text2Meta

t2m = Text2Meta(data=df, text_column="text", target="label")

# Compute specific metrics
features = t2m.compute(metrics=["number_of_documents", "vocabulary_size"])

# Compute entire groups
features = t2m.compute(groups=["general", "statistical"])
```

## Architecture Overview

Text2Meta follows a modular and extensible architecture, where each group of meta-features is implemented as an independent module.

| Group  | Module |
| ------------- |:-------------:|
| General      | SimpleFeatures    |
|  Statistical     |  StatisticalFeatures    |
|  Information-Theoretic     |   InfoFeatures  |
| Model-Based  |  ModelBasedFeatures   |
|  Landmarking |  LandmarkingFeatures   |
| Complexity  |  ComplexityFeatures   |

- Users may compute:
    - Individual meta-features via metrics
    - Entire groups via groups

- Each module exposes methods that compute one meta-feature.

- **New meta-features can be added by defining a new method in any module**

---

## 📦 General Meta-Features (SimpleFeatures)

These features describe **basic structural, lexical, and class-distribution** properties of the dataset.

| Metric  | Group | Description |
| ------------- |:-------------:|-------------|
|  `number_of_documents`     | General |  Total number of non-null text documents.    |
|  `number_of_classes`     | General |  Number of unique target labels.    |
|  `number_of_words`     | General | Total number of word tokens across all documents.     |
|  `avg_char_per_word`     | General | Average number of characters per word.     |
|  `min_char_per_word`     | General |  Length (in characters) of the shortest word appearing in the entire corpus.    |
|  `max_char_per_word`     | General |  Length (in characters) of the longest word appearing in the entire corpus.    |
|  `min_words_per_doc`     | General | Minimum document length (in words).     |
|  `max_words_per_doc`     | General |  Maximum document length (in words).    |
|  `vocabulary_size`     | General |  Number of unique word types (case-insensitive).    |
|  `type_token_relation`     | General | Type-token ratio (lexical diversity).     |
|  `avg_unique_words_per_doc`     | General |  Average number of unique words per document.    |
|  `english_stopword_ratio`     | General |  Average ratio of English stopwords per document.    |
|  `special_char_ratio`     | General |  Average ratio of non-alphanumeric characters per document.    |
|  `punctuation_token_ratio`     | General | Ratio of punctuation tokens to total tokens.     |
|  `max_min_length_ratio`     | General |  Ratio between the longest and shortest document lengths.    |
|  `min_class_count`     | General | Number of instances in the minority class.     |
|  `max_class_count`     | General | Number of instances in the majority class.     |
|  `avg_class_count`     | General | Average number of instances per class.     |

## 📈 Statistical Meta-Features (`StatisticalFeatures`)

These meta-features capture **distributional statistics, dispersion, asymmetry, and lexical frequency patterns** of textual datasets.

| Metric | Group | Description |
|------|:-----:|-------------|
| `std_class_count` | Statistical | Standard deviation of the number of instances per class. |
| `avg_std_ratio_class_count` | Statistical | Ratio between the mean and the standard deviation of class instance counts. |
| `avg_words_per_doc` | Statistical | Average number of words per document. |
| `median_words_per_doc` | Statistical | Median number of words per document. |
| `variance_words_per_doc` | Statistical | Variance of document lengths (in words). |
| `std_words_per_doc` | Statistical | Standard deviation of document lengths (in words). |
| `std_char_per_word` | Statistical | Standard deviation of word lengths (in characters) across the entire corpus. |
| `skewness_per_doc` | Statistical | Skewness of the document length distribution (in words). |
| `kurtosis_per_doc` | Statistical | Kurtosis of the document length distribution (in words). |
| `avg_bigram_freq_per_doc` | Statistical | Average number of bigrams per document. |
| `avg_trigram_freq_per_doc` | Statistical | Average number of trigrams per document. |
| `hapax_legomena_ratio` | Statistical | Ratio of words that occur exactly once in the corpus (hapax legomena). |
| `high_frequency_word_ratio` | Statistical | Ratio of words whose relative frequency exceeds a given threshold (default: 5%). |
| `rare_word_ratio` | Statistical | Ratio of words that occur exactly once relative to the total number of tokens. |
| `numeric_token_ratio` | Statistical | Ratio of numeric tokens (integers or decimals) to total tokens in the corpus. |
| `long_word_ratio` | Statistical | Ratio of words whose length exceeds a given threshold (default: ≥ 7 characters). |


## 📊 Information-Theoretic Meta-Features (`InfoFeatures`)

These meta-features quantify the **uncertainty, diversity, and dependency structure** between words and class labels.

| Metric  | Group | Description |
| ------------- |:-------------:|-------------|
|  `corpus_word_entropy`    | Information-Theoretic |  Shannon entropy of the word distribution over the entire corpus.   |
|  `words_per_doc_entropy`   | Information-Theoretic |  Average Shannon entropy of the word distribution computed per document.   |
|  `class_entropy`    | Information-Theoretic |  Shannon entropy of the class label distribution.   |
|  `max_class_entropy`    | Information-Theoretic |  Maximum possible class entropy for the dataset, assuming uniform class distribution.   |
|  `normalized_class_entropy`    | Information-Theoretic |  Class entropy normalized to the range \([0, 1]\).   |
|  `mean_mutual_info_words_classes`    | Information-Theoretic |  Average Mutual Information (MI) between words and class labels.   |
|  `mean_info_gain_ratio`    | Information-Theoretic |  Average Information Gain Ratio (IGR) between words and class labels.   |

## 🤖 Model-Based Meta-Features (`ModelBasedFeatures`)

These meta-features characterize the dataset through the **behavior, structure, and performance of simple machine learning models** trained on the text data.

| Metric | Group | Description |
|------|:-----:|-------------|
| `num_tree_nodes` | Model-Based | Number of nodes in a Decision Tree trained on the dataset. |
| `avg_tree_depth` | Model-Based | Maximum depth of a Decision Tree trained on the dataset. |
| `num_tree_leaves` | Model-Based | Number of leaf nodes in a Decision Tree trained on the dataset. |
| `avg_feature_importance` | Model-Based | Average feature importance assigned by a Random Forest classifier. |
| `coef_stability` | Model-Based | Standard deviation of Logistic Regression coefficients across multiple random initializations. |
| `embedding_matrix_size` | Model-Based | Estimated size of a word embedding matrix assuming 100 dimensions per unique word. |
| `sparsity` | Model-Based | Sparsity of the TF-IDF document–term matrix. |
| `ideal_clusters_for_kmeans` | Model-Based | Number of clusters that maximizes the Silhouette Score in K-Means clustering. |
| `avg_training_time` | Model-Based | Average training time of simple classifiers (Logistic Regression, Naive Bayes, Decision Tree). |
| `avg_prediction_time` | Model-Based | Average per-sample prediction time across multiple classifiers. |
| `svd_top10_singular_ratio` | Model-Based | Ratio between the sum of the top-10 singular values and the total sum of singular values from SVD. |
| `svd_total_singular_sum` | Model-Based | Sum of all singular values obtained from Truncated SVD. |
| `svd_total_explained_variance` | Model-Based | Total explained variance from Truncated SVD. |
| `svd_total_explained_ratio` | Model-Based | Total explained variance ratio from Truncated SVD. |

## 🧭 Landmarking Meta-Features (`LandmarkingFeatures`)

These meta-features characterize the dataset by measuring the **performance, robustness, and behavior of simple learning algorithms (landmarkers)** when applied to the text data.

| Metric | Group | Description |
|------|:-----:|-------------|
| `avg_acc` | Landmarking | Average classification accuracy across simple classifiers using cross-validation. |
| `avg_balanced_acc` | Landmarking | Average balanced accuracy across simple classifiers using cross-validation. |
| `avg_f1_macro` | Landmarking | Average macro-averaged F1-score across simple classifiers. |
| `avg_f1_micro` | Landmarking | Average micro-averaged F1-score across simple classifiers. |
| `avg_f1_weighted` | Landmarking | Average weighted F1-score across simple classifiers. |
| `silhouette_score_embeddings` | Landmarking | Silhouette Score obtained by applying K-Means to TF-IDF embeddings. |
| `davies_bouldin_score_embeddings` | Landmarking | Davies–Bouldin Index obtained by applying K-Means to TF-IDF embeddings. |
| `cluster_label_agreement` | Landmarking | Agreement between K-Means cluster assignments and true class labels measured by Adjusted Rand Index (ARI). |
| `linear_vs_nonlinear_gap_acc` | Landmarking | Difference in accuracy between a non-linear model (Random Forest) and a linear model (Logistic Regression). |
| `linear_vs_nonlinear_gap_balanced_acc` | Landmarking | Difference in balanced accuracy between non-linear and linear models. |
| `linear_vs_nonlinear_gap_f1_macro` | Landmarking | Difference in macro F1-score between non-linear and linear models. |
| `linear_vs_nonlinear_gap_f1_micro` | Landmarking | Difference in micro F1-score between non-linear and linear models. |
| `linear_vs_nonlinear_gap_f1_weighted` | Landmarking | Difference in weighted F1-score between non-linear and linear models. |
| `robustness_to_noise_acc_linear` | Landmarking | Decrease in accuracy of a linear model after injecting label noise. |
| `robustness_to_noise_balanced_acc_linear` | Landmarking | Decrease in balanced accuracy of a linear model after injecting label noise. |
| `robustness_to_noise_f1_macro_linear` | Landmarking | Decrease in macro F1-score of a linear model after injecting label noise. |
| `robustness_to_noise_f1_micro_linear` | Landmarking | Decrease in micro F1-score of a linear model after injecting label noise. |
| `robustness_to_noise_f1_weighted_linear` | Landmarking | Decrease in weighted F1-score of a linear model after injecting label noise. |

## 📚 Text Complexity Meta-Features (`ComplexityFeatures`)

These meta-features estimate the **linguistic and readability complexity** of the textual dataset.

| Metric | Group | Description |
|------|:-----:|-------------|
| `flesch_reading_ease` | Complexity | Flesch Reading Ease score; higher values indicate easier text. |
| `smog_index` | Complexity | SMOG readability index estimating years of education needed to understand the text. |
| `flesch_kincaid_grade` | Complexity | Flesch–Kincaid Grade Level indicating the U.S. school grade required. |
| `coleman_liau_index` | Complexity | Coleman–Liau Index based on characters per word and words per sentence. |
| `automated_readability_index` | Complexity | Automated Readability Index estimating required education level. |
| `dale_chall_readability_score` | Complexity | Dale–Chall Readability Score based on familiar word lists. |
| `difficult_words` | Complexity | Number of difficult (unfamiliar) words in the corpus. |
| `linsear_write_formula` | Complexity | Linsear Write Formula estimating readability for technical texts. |
| `gunning_fog` | Complexity | Gunning Fog Index estimating years of formal education needed. |
| `school_grade` | Complexity | Extracted school grade level derived from the TextStandard metric. |


