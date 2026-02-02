import re
import numpy as np
import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords', quiet=True)

class SimpleFeatures:
    def __init__(self, data, text_column, target=None):
        self.data = data
        self.text_column = text_column
        self.target = target

    def number_of_documents(self):
        return len(self.data[self.text_column].dropna())

    def number_of_classes(self):
        self.data = self.data.dropna(subset=[self.target])
        if self.target is None:
            raise ValueError("target é necessário para count_classes.")
        return self.data[self.target].nunique()

    def number_of_words(self):
        return int(self.data[self.text_column].astype(str).str.split().str.len().sum())

    def avg_char_per_word(self):
        all_text = self.data[self.text_column].astype(str).str.cat(sep=' ')
        words = all_text.split()
        total_chars = sum(len(word) for word in words)
        return total_chars / len(words) if words else 0

    def min_char_per_word(self):
        min_lengths = (
            self.data[self.text_column]
            .astype(str)
            .apply(lambda x: min(len(word) for word in x.split()) if x.split() else 0)
        )
        return min_lengths.min()

    def max_char_per_word(self):
        max_lengths = (
        self.data[self.text_column]
        .astype(str)
        .apply(lambda x: max(len(word) for word in x.split()) if x.split() else 0)
    )
        return max_lengths.max()

    def max_words_per_doc(self):
        word_counts = self.data[self.text_column].astype(str).apply(lambda x: len(x.split()))
        return word_counts.max()

    def min_words_per_doc(self):
        word_counts = self.data[self.text_column].astype(str).apply(lambda x: len(x.split()))
        return word_counts.min()

    def vocabulary_size(self):
        words = self.data[self.text_column].astype(str).str.lower().str.split().explode()
        return words.nunique()

    def type_token_relation(self):
        total_words = self.number_of_words()
        total_vocabulary = self.vocabulary_size()
        return float(total_vocabulary / total_words) if total_words > 0 else 0

    def avg_unique_words_per_doc(self):
        unique_counts = self.data[self.text_column].astype(str).apply(lambda x: len(set(x.lower().split())))
        return float(unique_counts.mean())

    def english_stopword_ratio(self):
        stop_words = set(stopwords.words("english"))  # você pode mudar para 'english'

        ratios = []
        for doc in self.data[self.text_column].astype(str):
            words = doc.split()
            if len(words) == 0:  # evita divisão por zero
                ratios.append(0)
            else:
                stop_count = sum(1 for w in words if w.lower() in stop_words)
                ratios.append(stop_count / len(words))

        return sum(ratios) / len(ratios)  # média da razão de stopwords por documento

    def special_char_ratio(self):
        ratios = []
        for doc in self.data[self.text_column].astype(str):
            total_chars = len(doc)
            if total_chars == 0:  # evita divisão por zero
                ratios.append(0)
            else:
                # Conta quantos caracteres NÃO são letras ou números
                special_chars = len(re.findall(r"[^a-zA-Z0-9\s]", doc))
                ratios.append(special_chars / total_chars)

        return sum(ratios) / len(ratios)  # retorna a média da razão de caracteres especiais

    def punctuation_token_ratio(self):
        all_text = self.data[self.text_column].astype(str).str.cat(sep=' ')
        total_tokens = all_text.split()
        punct_count = sum(1 for token in total_tokens if token in string.punctuation)
        return punct_count / len(total_tokens) if total_tokens else 0

    def max_min_length_ratio(self):
        lengths = [len(str(doc).split()) for doc in self.data[self.text_column].astype(str)]
        if not lengths or min(lengths) == 0:  # evita divisão por zero
            return 0
        return max(lengths) / min(lengths)

    def min_class_count(self):
        if self.target is None:
            raise ValueError("target é necessário para min_class_count.")
        self.data = self.data.dropna(subset=[self.target])
        return self.data[self.target].value_counts().min()

    def max_class_count(self):
        if self.target is None:
            raise ValueError("target é necessário para max_class_count.")
        self.data = self.data.dropna(subset=[self.target])
        return self.data[self.target].value_counts().max()

    def avg_class_count(self):
        if self.target is None:
            raise ValueError("target é necessário para average_class_count.")
        self.data = self.data.dropna(subset=[self.target])
        class_counts = self.data[self.target].value_counts()
        return float(class_counts.mean())
