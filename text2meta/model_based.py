from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import numpy as np
import time

class ModelBasedFeatures:
    def __init__(self, data, text_column, target=None):
        self.text_column = text_column
        self.target = target

        if target is not None:
            data = data.dropna(subset=[target]).reset_index(drop=True)
            class_counts = data[target].value_counts()
            can_stratify = class_counts.min() >= 2
    
            if len(data) > 1000:
                if can_stratify:
                    data, _ = train_test_split(
                        data,
                        train_size=1000,
                        stratify=data[target],
                        random_state=42
                    )
                else:
                    data = data.sample(n=1000, random_state=42)
        self.data = data.reset_index(drop=True)
        
        self.clf_dt = None
        self.clf_rf = None
        self.clf_lr = None
        self.clf_nb = None
        self._svd = None

    # ==============================
    # Decision Tree
    # ==============================

    def num_tree_nodes(self):
        if self.clf_dt is None:
            self._train_model(model_type='dt')
        return self.clf_dt.tree_.node_count

    def avg_tree_depth(self):
        if self.clf_dt is None:
            self._train_model(model_type='dt')
        return self.clf_dt.tree_.max_depth

    def num_tree_leaves(self):
        if self.clf_dt is None:
            self._train_model(model_type='dt')
        return int(self.clf_dt.tree_.n_leaves)

    # ==============================
    # Random Forest Feature Importance
    # ==============================

    def avg_feature_importance(self):
        if self.clf_rf is None:
            self._train_model(model_type='rf')
        return float(np.mean(self.clf_rf.feature_importances_))

    # ==============================
    # Regressão Logística
    # ==============================

    def coef_stability(self):
        X = self._vectorize_text()
        coef_list = []
        for seed in [1, 42, 20]:
            clf = LogisticRegression(max_iter=500, random_state=seed)
            clf.fit(X, self.data[self.target])
            coef_list.append(clf.coef_.flatten())
        coef_matrix = np.vstack(coef_list)
        return float(np.std(coef_matrix))

    # ==============================
    # Word Embeddings
    # ==============================

    def embedding_matrix_size(self):
        words = self.data[self.text_column].astype(str).str.lower().str.split().explode()
        return words.nunique() * 100

    # ==============================
    # Sparsidade
    # ==============================

    def sparsity(self):
        X = self._vectorize_text()
        total_elements = X.shape[0] * X.shape[1]
        non_zero = X.nnz
        return 1 - (non_zero / total_elements)

    # ==============================
    # Clusterização
    # ==============================

    def ideal_clusters_for_kmeans(self):
        X = self._vectorize_text()
        sil_score_max = -1 #this is the minimum possible score

        for n_clusters in range(2,10):
            model = KMeans(n_clusters = n_clusters, init='k-means++', max_iter=100, n_init=1)
            try:
                labels = model.fit_predict(X)
                sil_score = silhouette_score(X, labels)
            except Exception as e:
                sil_score = -1
            if sil_score > sil_score_max:
                sil_score_max = sil_score
                best_n_clusters = n_clusters
        return best_n_clusters

    # ==============================
    # Tempo de treino e predição
    # ==============================

    def avg_training_time(self):
        models = [
            LogisticRegression(max_iter=500),
            MultinomialNB(),
            DecisionTreeClassifier(random_state=42)
        ]
        X = self._vectorize_text()
        times = []
        for model in models:
            start = time.time()
            model.fit(X, self.data[self.target])
            times.append(time.time() - start)
        return float(np.mean(times))

    def avg_prediction_time(self):
        if self.clf_lr is None:
            self._train_model(model_type='lr')
        if self.clf_nb is None:
            self._train_model(model_type='nb')
        if self.clf_dt is None:
            self._train_model(model_type='dt')

        X = self._vectorize_text()
        times = []
        for model in ['lr', 'nb', 'dt']:
            start = time.time()
            clf = getattr(self, f"clf_{model}")  # recupera o atributo dinamicamente
            clf.predict(X[:50])
            elapsed = (time.time() - start) / 50
            times.append(elapsed)
        return float(np.mean(times))

    # ==============================
    # SVD Meta-features (TF-IDF)
    # ==============================

    def svd_top10_singular_ratio(self):
        svd = self._fit_svd()
        singular_values = svd.singular_values_
        sum_top10 = np.sum(singular_values[:10])
        sum_all = np.sum(singular_values)
        return float(sum_top10 / sum_all) if sum_all > 0 else 0.0

    def svd_total_singular_sum(self):
        svd = self._fit_svd()
        return float(np.sum(svd.singular_values_))

    def svd_total_explained_variance(self):
        svd = self._fit_svd()
        return float(np.sum(svd.explained_variance_))

    def svd_total_explained_ratio(self):
        svd = self._fit_svd()
        return float(np.sum(svd.explained_variance_ratio_))

    def _train_model(self, model_type=None):
        if model_type == "dt":
            self.clf_dt = DecisionTreeClassifier(random_state=42)
            self.clf_dt.fit(self._vectorize_text(), self.data[self.target])
        if model_type == "rf":
            self.clf_rf = RandomForestClassifier(random_state=42)
            self.clf_rf.fit(self._vectorize_text(), self.data[self.target])
        if model_type == "lr":
            self.clf_lr = LogisticRegression(max_iter=500, random_state=42)
            self.clf_lr.fit(self._vectorize_text(), self.data[self.target])
        if model_type == "nb":
            self.clf_nb = MultinomialNB()
            self.clf_nb.fit(self._vectorize_text(), self.data[self.target])

    def _vectorize_text(self):
        if self.target is not None:
            self.data = self.data.dropna(subset=[self.target])
        return TfidfVectorizer(max_features=1000).fit_transform(self.data[self.text_column].astype(str).tolist())

    def _fit_svd(self, n_components=100):
        """Treina o SVD apenas uma vez e guarda o resultado."""
        if self._svd is None:
            X = self._vectorize_text()
            svd = TruncatedSVD(n_components=n_components, random_state=42)
            svd.fit(X)
            self._svd = svd
        return self._svd