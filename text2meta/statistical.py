import re
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew
from collections import Counter
import nltk
from nltk import ngrams

class StatisticalFeatures:
    def __init__(self, data, text_column, target=None):
        self.data = data
        self.text_column = text_column
        self.target = target

    def std_class_count(self):
        if self.target is None:
            raise ValueError("target é necessário para std_class_count.")
        self.data = self.data.dropna(subset=[self.target])
        class_counts = self.data[self.target].value_counts()
        return class_counts.std()

    def avg_std_ratio_class_count(self):
        if self.target is None:
            raise ValueError("target é necessário para avg_std_ratio_class_count.")
        self.data = self.data.dropna(subset=[self.target])
        class_counts = self.data[self.target].value_counts()
        mean = class_counts.mean()
        std = class_counts.std()
        return float(mean / std) if std != 0 else 0  # evita divisão por zero

    def avg_words_per_doc(self):
        word_counts = self.data[self.text_column].astype(str).apply(lambda x: len(x.split()))
        return float(word_counts.mean())

    def median_words_per_doc(self):
        word_counts = self.data[self.text_column].astype(str).apply(lambda x: len(x.split()))
        return word_counts.median()

    def variance_words_per_doc(self):
        word_counts = self.data[self.text_column].astype(str).apply(lambda x: len(x.split()))
        return word_counts.var()

    def std_words_per_doc(self):
        word_counts = self.data[self.text_column].astype(str).apply(lambda x: len(x.split()))
        return word_counts.std()

    def std_char_per_word(self):
        words = (
            self.data[self.text_column]
            .astype(str)
            .str.split()
            .explode()
        )

        if len(words) == 0:
            return 0.0

        word_lengths = words.apply(len)
        return float(word_lengths.std())

    def skewness_per_doc(self):
        if self.text_column is None:
            raise ValueError("text_column é necessário para skewness_per_document.")
        word_counts = self.data[self.text_column].astype(str).apply(lambda x: len(x.split()))
        return float(skew(word_counts))

    def kurtosis_per_doc(self):
        if self.text_column is None:
            raise ValueError("text_column é necessário para skewness_per_document.")
        word_counts = self.data[self.text_column].astype(str).apply(lambda x: len(x.split()))
        return float(kurtosis(word_counts))

    def avg_bigram_freq_per_doc(self):
        ngram_counts = self.data[self.text_column].astype(str).apply(
            lambda x: len(list(ngrams(x.split(), 2))) if len(x.split()) >= 2 else 0
        )
        return float(ngram_counts.mean())

    def avg_trigram_freq_per_doc(self):
        ngram_counts = self.data[self.text_column].astype(str).apply(
            lambda x: len(list(ngrams(x.split(), 3))) if len(x.split()) >= 3 else 0
        )
        return float(ngram_counts.mean())

    def hapax_legomena_ratio(self):
        words = self.data[self.text_column].astype(str).str.lower().str.split().explode()
        counts = words.value_counts()
        hapax = (counts == 1).sum()
        return float(hapax / len(counts)) if len(counts) > 0 else 0

    def high_frequency_word_ratio(self, threshold=0.05):
        words = self.data[self.text_column].astype(str).str.lower().str.split().explode()
        counts = words.value_counts()
        total = counts.sum()
        high_freq = (counts / total > threshold).sum()
        return float(high_freq / len(counts)) if len(counts) > 0 else 0

    def rare_word_ratio(self):
        """
        Calcula a frequência média de palavras raras (ocorrem só 1 vez no corpus).
        """
        if self.text_column is None:
            raise ValueError("text_column é necessário para rare_word_frequency.")

        all_words = " ".join(self.data[self.text_column].astype(str).tolist()).split()
        word_counts = Counter(all_words)

        rare_words = {w for w, c in word_counts.items() if c == 1}
        total_words = len(all_words)

        return len(rare_words) / total_words if total_words > 0 else 0

    def numeric_token_ratio(self):
        """
        Calcula a razão de tokens numéricos no corpus em relação ao total de tokens.
        """
        if self.text_column is None:
            raise ValueError("text_column é necessário para numeric_token_ratio.")

        # Divide cada texto em tokens e empilha em uma única Series
        tokens = self.data[self.text_column].astype(str).str.split().explode()

        # Conta total de tokens
        total_tokens = len(tokens)

        # Expressão regular para inteiros e decimais
        numeric_pattern = r'^\d+(\.\d+)?$'

        # Conta quantos são numéricos usando .str.match (vetorizado)
        numeric_tokens = tokens.str.match(numeric_pattern).sum()

        return float(numeric_tokens / total_tokens) if total_tokens > 0 else 0

    def long_word_ratio(self, threshold=7):
        all_text = self.data[self.text_column].astype(str).str.cat(sep=' ')
        words = all_text.split()
        long_words = [word for word in words if len(word) >= threshold]
        return len(long_words) / len(words) if words else 0
