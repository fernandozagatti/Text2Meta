import re
import numpy as np
import pandas as pd
import math
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import mutual_info_classif
from scipy.stats import entropy

class InfoFeatures:
    def __init__(self, data, text_column, target=None):
        self.data = data
        self.text_column = text_column
        self.target = target

    def corpus_word_entropy(self):
        if self.text_column is None:
            raise ValueError("text_column é necessário para corpus_word_entropy.")

        all_text = " ".join(self.data[self.text_column].astype(str))
        words = all_text.split()
        if not words:
            return 0
        freqs = Counter(words)
        probs = [count / len(words) for count in freqs.values()]
        return float(entropy(probs, base=2))


    def words_per_doc_entropy(self):
        if self.text_column is None:
            raise ValueError("text_column é necessário para entropy_per_document.")

        entropies = []
        for doc in self.data[self.text_column].astype(str):
            words = doc.split()
            if not words:
                entropies.append(0)
                continue
            freqs = Counter(words)
            probs = [count / len(words) for count in freqs.values()]
            entropies.append(entropy(probs, base=2))  # base 2 = bits

        return float(sum(entropies) / len(entropies))

    def class_entropy(self):
        self.data = self.data.dropna(subset=[self.target])
        if self.target is None:
            raise ValueError("target é necessário para class_entropy.")
        counts = self.data[self.target].value_counts()
        total = len(self.data)
        probs = counts / total
        entropy = -sum(p * math.log2(p) for p in probs if p > 0)
        return entropy

    def max_class_entropy(self):
        self.data = self.data.dropna(subset=[self.target])
        if self.target is None:
            raise ValueError("target é necessário para max_class_entropy.")
        k = self.data[self.target].nunique()
        return math.log2(k) if k > 0 else 0

    def normalized_class_entropy(self):
        ent = self.class_entropy()
        max_ent = self.max_class_entropy()
        return ent / max_ent if max_ent > 0 else 0

    def mean_mutual_info_words_classes(self, max_features=1000, binary=True, random_state=0):
        """
        Calcula a informação mútua entre cada palavra (feature) e a classe e retorna a média.

        Parâmetros:
        - max_features (int): número máximo de features do CountVectorizer (top n palavras).
        - binary (bool): se True usa presença/ausência (0/1). Se False usa contagens.
        - return_series (bool): se True retorna um pd.Series com MI por termo; se False retorna apenas a média.
        - random_state (int): semente para mutual_info_classif (estocástico).

        Retorno:
        - float (média da informação mútua) ou (float, pd.Series) se return_series=True
        """
        self.data = self.data.dropna(subset=[self.target])
        if self.target is None:
            raise ValueError("target é necessário para mean_mutual_info_words_classes.")
        # prepara X e y
        texts = self.data[self.text_column].astype(str).values
        y = self.data[self.target].values

        vec = CountVectorizer(max_features=max_features, binary=binary)
        X = vec.fit_transform(texts)  # X: sparse matrix (n_samples, n_features)
        feature_names = vec.get_feature_names_out()

        # mutual_info_classif espera X (array-like/sparse) e y (1d)
        # discrete_features='auto' funciona bem; para presença (binary=True) features são discretas
        mi = mutual_info_classif(X, y, discrete_features='auto', random_state=random_state)

        # mi é um array com MI por feature
        mi = np.asarray(mi, dtype=float)

        # média (pode ser 0 se não houver features)
        mean_mi = float(mi.mean()) if mi.size > 0 else 0.0

        return mean_mi

    def mean_info_gain_ratio(self, max_features=1000, binary=True, random_state=0):
        """
        Calcula a razão de informação média (Information Gain Ratio) entre palavras e classes.

        Fórmula:
        IGR(w) = IG(w, C) / H(w),
        onde IG(w, C) é o ganho de informação (mutual info) e H(w) é a entropia da feature.

        Parâmetros:
        - max_features (int): número máximo de features do CountVectorizer (top n palavras).
        - binary (bool): se True usa presença/ausência; se False usa contagens.
        - return_series (bool): se True retorna pd.Series com IGR por termo.
        - random_state (int): semente para mutual_info_classif.

        Retorna:
        - float (média da razão de informação) ou (float, pd.Series) se return_series=True
        """
        self.data = self.data.dropna(subset=[self.target])
        if self.target is None:
            raise ValueError("target é necessário para mean_information_gain_ratio.")

        texts = self.data[self.text_column].astype(str).values
        y = self.data[self.target].values

        # Criação da matriz termo-documento
        vec = CountVectorizer(max_features=max_features, binary=binary)
        X = vec.fit_transform(texts)
        feature_names = vec.get_feature_names_out()

        # MI entre cada feature e a classe
        mi = mutual_info_classif(X, y, discrete_features='auto', random_state=random_state)
        mi = np.asarray(mi, dtype=float)

        # Entropia de cada feature (presença/ausência por doc)
        # convertemos sparse para array binário (ou contagem se binary=False)
        X_bin = (X > 0).astype(int) if binary else X.copy()
        freqs = np.asarray(X_bin.sum(axis=0)).flatten() / X.shape[0]
        H = - (freqs * np.log2(freqs + 1e-12) + (1 - freqs) * np.log2(1 - freqs + 1e-12))

        # Razão de informação
        igr = np.divide(mi, H, out=np.zeros_like(mi), where=H > 0)

        mean_igr = float(igr.mean()) if igr.size > 0 else 0.0

        return mean_igr