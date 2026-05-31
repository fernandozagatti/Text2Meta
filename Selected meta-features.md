# Meta-Feature Selection Results

This document presents the meta-features selected by each feature selection strategy evaluated in the Text2Meta paper.

## Complete Set

Number of meta-features: **83**

Contains all meta-features available in Text2Meta.

```python
from auto_nlp import MetaLearner

automl = MetaLearner(verbose=1, model_choice="rf")
automl.retrain(metrics=["avg_char_per_word", "avg_class_count", "avg_unique_words_per_doc", "english_stopword_ratio", "max_char_per_word", "max_class_count", "max_min_length_ratio", "max_words_per_doc", "min_char_per_word", "min_class_count", "min_words_per_doc", "number_of_classes", "number_of_documents", "number_of_words", "punctuation_token_ratio", "special_char_ratio", "type_token_relation", "vocabulary_size", "avg_bigram_freq_per_doc", "avg_std_ratio_class_count", "avg_trigram_freq_per_doc", "avg_words_per_doc", "hapax_legomena_ratio", "high_frequency_word_ratio", "kurtosis_per_doc", "long_word_ratio", "median_words_per_doc", "numeric_token_ratio", "rare_word_ratio", "skewness_per_doc", "std_char_per_word", "std_class_count", "std_words_per_doc", "variance_words_per_doc", "class_entropy", "corpus_word_entropy", "max_class_entropy", "mean_info_gain_ratio", "mean_mutual_info_words_classes", "normalized_class_entropy", "words_per_doc_entropy", "avg_feature_importance", "avg_prediction_time", "avg_training_time", "avg_tree_depth", "coef_stability", "embedding_matrix_size", "ideal_clusters_for_kmeans", "num_tree_leaves", "num_tree_nodes", "sparsity", "svd_top10_singular_ratio", "svd_total_explained_ratio", "svd_total_explained_variance", "svd_total_singular_sum", "avg_acc", "avg_balanced_acc", "avg_f1_macro", "avg_f1_micro", "avg_f1_weighted", "cluster_label_agreement", "davies_bouldin_score_embeddings", "linear_vs_nonlinear_gap_acc", "linear_vs_nonlinear_gap_balanced_acc", "linear_vs_nonlinear_gap_f1_macro", "linear_vs_nonlinear_gap_f1_micro", "linear_vs_nonlinear_gap_f1_weighted", "robustness_to_noise_acc_linear", "robustness_to_noise_balanced_acc_linear", "robustness_to_noise_f1_macro_linear", "robustness_to_noise_f1_micro_linear", "robustness_to_noise_f1_weighted_linear", "silhouette_score_embeddings", "automated_readability_index", "coleman_liau_index", "dale_chall_readability_score", "difficult_words", "flesch_kincaid_grade", "flesch_reading_ease", "gunning_fog", "linsear_write_formula", "school_grade", "smog_index"])
```

---

## Correlation Threshold = 0.95

Number of meta-features: **60**

Selected meta-features: 

```python
from auto_nlp import MetaLearner

automl = MetaLearner(verbose=1, model_choice="rf")
automl.retrain(metrics=["avg_char_per_word", "avg_class_count", "avg_unique_words_per_doc", "english_stopword_ratio", "max_char_per_word", "max_min_length_ratio", "min_char_per_word", "number_of_classes", "number_of_words", "punctuation_token_ratio", "special_char_ratio", "type_token_relation", "vocabulary_size", "avg_std_ratio_class_count", "avg_trigram_freq_per_doc", "hapax_legomena_ratio", "high_frequency_word_ratio", "kurtosis_per_doc", "long_word_ratio", "median_words_per_doc", "numeric_token_ratio", "skewness_per_doc", "std_char_per_word", "std_class_count", "class_entropy", "corpus_word_entropy", "mean_info_gain_ratio", "mean_mutual_info_words_classes", "normalized_class_entropy", "words_per_doc_entropy", "avg_feature_importance", "avg_prediction_time", "avg_training_time", "avg_tree_depth", "coef_stability", "embedding_matrix_size", "ideal_clusters_for_kmeans", "num_tree_leaves", "sparsity", "svd_top10_singular_ratio", "svd_total_explained_ratio", "svd_total_explained_variance", "svd_total_singular_sum", "avg_acc", "avg_balanced_acc", "cluster_label_agreement", "davies_bouldin_score_embeddings", "linear_vs_nonlinear_gap_acc", "linear_vs_nonlinear_gap_balanced_acc", "linear_vs_nonlinear_gap_f1_weighted", "robustness_to_noise_acc_linear", "robustness_to_noise_balanced_acc_linear", "robustness_to_noise_f1_macro_linear", "robustness_to_noise_f1_weighted_linear", "silhouette_score_embeddings", "automated_readability_index", "flesch_reading_ease", "linsear_write_formula", "school_grade", "smog_index"])
```

---

## Correlation Threshold = 0.50

Number of meta-features: **22**

Selected meta-features: 

```python
from auto_nlp import MetaLearner

automl = MetaLearner(verbose=1, model_choice="rf")
automl.retrain(metrics=["avg_char_per_word", "avg_class_count", "avg_unique_words_per_doc", "english_stopword_ratio", "max_char_per_word", "min_char_per_word", "number_of_classes", "number_of_words", "punctuation_token_ratio", "high_frequency_word_ratio", "numeric_token_ratio", "corpus_word_entropy", "normalized_class_entropy", "avg_feature_importance", "ideal_clusters_for_kmeans", "svd_total_explained_ratio", "davies_bouldin_score_embeddings", "linear_vs_nonlinear_gap_acc", "automated_readability_index", "flesch_reading_ease", "linsear_write_formula", "school_grade"])
```

---

## Variance Threshold = 1

Number of meta-features: **33**

Selected meta-features: 

```python
from auto_nlp import MetaLearner

automl = MetaLearner(verbose=1, model_choice="rf")
automl.retrain(metrics=["max_class_count", "min_class_count", "punctuation_token_ratio", "type_token_relation", "avg_bigram_freq_per_doc", "avg_trigram_freq_per_doc", "avg_words_per_doc", "high_frequency_word_ratio", "kurtosis_per_doc", "median_words_per_doc", "std_class_count", "std_words_per_doc", "variance_words_per_doc", "corpus_word_entropy", "max_class_entropy", "mean_info_gain_ratio", "mean_mutual_info_words_classes", "avg_prediction_time", "ideal_clusters_for_kmeans", "svd_top10_singular_ratio", "avg_acc", "avg_f1_macro", "avg_f1_micro", "cluster_label_agreement", "linear_vs_nonlinear_gap_acc", "linear_vs_nonlinear_gap_f1_micro", "linear_vs_nonlinear_gap_f1_weighted", "robustness_to_noise_acc_linear", "silhouette_score_embeddings", "coleman_liau_index", "flesch_kincaid_grade", "linsear_write_formula", "school_grade"])
```

---

## Laplacian Score (k = 15)

Number of meta-features: **15**

Selected meta-features: 

```python
from auto_nlp import MetaLearner

automl = MetaLearner(verbose=1, model_choice="rf")
automl.retrain(metrics=["hapax_legomena_ratio", "embedding_matrix_size", "flesch_reading_ease", "svd_top10_singular_ratio", "dale_chall_readability_score", "sparsity", "class_entropy", "svd_total_explained_variance", "smog_index", "svd_total_explained_ratio", "robustness_to_noise_f1_macro_linear", "number_of_documents", "max_min_length_ratio", "avg_words_per_doc", "special_char_ratio"])
```

---

## Meta-Features Never Selected

The following meta-features were not selected by any strategy, except for the first one:

### Simple

* max_words_per_doc
* min_words_per_doc
* vocabulary_size

### Statistical

* avg_std_ratio_class_count
* long_word_ratio
* rare_word_ratio
* skewness_per_doc
* std_char_per_word

### Information-Theoretic

* words_per_doc_entropy

### Model-Based

* avg_training_time
* avg_tree_depth
* coef_stability
* num_tree_leaves
* num_tree_nodes
* svd_total_singular_sum

### Landmarking

* avg_balanced_acc
* avg_f1_weighted
* linear_vs_nonlinear_gap_balanced_acc
* linear_vs_nonlinear_gap_f1_macro
* robustness_to_noise_balanced_acc_linear
* robustness_to_noise_f1_micro_linear
* robustness_to_noise_f1_weighted_linear

### Complexity

* difficult_words
* gunning_fog

---

## Stable Meta-Features

Meta-features selected by multiple selection strategies, indicating higher stability:

* avg_words_per_doc
* corpus_word_entropy
* high_frequency_word_ratio
* ideal_clusters_for_kmeans
* linear_vs_nonlinear_gap_acc
* linsear_write_formula
* punctuation_token_ratio
* school_grade
* flesch_reading_ease
* svd_total_explained_ratio
* svd_top10_singular_ratio
