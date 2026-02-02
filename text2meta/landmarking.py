from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import numpy as np
import random

class LandmarkingFeatures:
    def __init__(self, data, text_column, target=None, dt=False):
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
        
        self.kmeans = None
        self._X = None
        self._y = None
        self._cv_results = {}  # cache para os resultados de validação cruzada

    def _evaluate_model(self, clf, scoring):
        """Executa cross_val_score e usa cache se já foi avaliado."""
        key = (clf.__class__.__name__, scoring)
        if key not in self._cv_results:
            score = cross_val_score(clf, self._X, self._y, cv=3, scoring=scoring).mean()
            self._cv_results[key] = score
        return self._cv_results[key]

    # ==============================
    # Médias das métricas de desempenho
    # ==============================

    def avg_acc(self):
        models = self._get_simple_models()
        accs = [self._evaluate_model(clf, "accuracy") for clf in models.values()]
        return float(np.mean(accs))

    def avg_balanced_acc(self):
        models = self._get_simple_models()
        f1s = [self._evaluate_model(clf, "balanced_accuracy") for clf in models.values()]
        return float(np.mean(f1s))

    def avg_f1_macro(self):
        models = self._get_simple_models()
        f1s = [self._evaluate_model(clf, "f1_macro") for clf in models.values()]
        return float(np.mean(f1s))

    def avg_f1_micro(self):
        models = self._get_simple_models()
        f1s = [self._evaluate_model(clf, "f1_micro") for clf in models.values()]
        return float(np.mean(f1s))

    def avg_f1_weighted(self):
        models = self._get_simple_models()
        f1s = [self._evaluate_model(clf, "f1_weighted") for clf in models.values()]
        return float(np.mean(f1s))

    # ==============================
    # Clusterização
    # ==============================

    def silhouette_score_embeddings(self):
        self._prepare_data()
        if self.kmeans is None:
            self.kmeans = KMeans(n_clusters=len(set(self._y)), random_state=42).fit(self._X.toarray())
        return float(silhouette_score(self._X.toarray(), self.kmeans.labels_))

    def davies_bouldin_score_embeddings(self):
        self._prepare_data()
        if self.kmeans is None:
            self.kmeans = KMeans(n_clusters=len(set(self._y)), random_state=42).fit(self._X.toarray())
        return float(davies_bouldin_score(self._X.toarray(), self.kmeans.labels_))

    def cluster_label_agreement(self):
        """
        Aplica KMeans e mede concordância com rótulos reais via ARI.
        """
        self._prepare_data()
        try:
            if self.kmeans is None:
                self.kmeans = KMeans(n_clusters=len(set(self._y)), random_state=42).fit(self._X)
            preds = self.kmeans.fit_predict(self._X)
            return float(adjusted_rand_score(self._y, preds))
        except Exception:
            return 0.0

    # ==============================
    # Modelo linear vs não-linear
    # ==============================

    def linear_vs_nonlinear_gap_acc(self):
        self._prepare_data()
        lr = LogisticRegression(max_iter=500, random_state=42)
        rf = RandomForestClassifier(random_state=42)
        acc_lr = self._evaluate_model(lr, "accuracy")
        acc_rf = self._evaluate_model(rf, "accuracy")
        return float(acc_rf - acc_lr)

    def linear_vs_nonlinear_gap_balanced_acc(self):
        self._prepare_data()
        lr = LogisticRegression(max_iter=500, random_state=42)
        rf = RandomForestClassifier(random_state=42)
        acc_lr = self._evaluate_model(lr, "balanced_accuracy")
        acc_rf = self._evaluate_model(rf, "balanced_accuracy")
        return float(acc_rf - acc_lr)

    def linear_vs_nonlinear_gap_f1_macro(self):
        self._prepare_data()
        lr = LogisticRegression(max_iter=500, random_state=42)
        rf = RandomForestClassifier(random_state=42)
        acc_lr = self._evaluate_model(lr, "f1_macro")
        acc_rf = self._evaluate_model(rf, "f1_macro")
        return float(acc_rf - acc_lr)

    def linear_vs_nonlinear_gap_f1_micro(self):
        self._prepare_data()
        lr = LogisticRegression(max_iter=500, random_state=42)
        rf = RandomForestClassifier(random_state=42)
        acc_lr = self._evaluate_model(lr, "f1_micro")
        acc_rf = self._evaluate_model(rf, "f1_micro")
        return float(acc_rf - acc_lr)

    def linear_vs_nonlinear_gap_f1_weighted(self):
        self._prepare_data()
        lr = LogisticRegression(max_iter=500, random_state=42)
        rf = RandomForestClassifier(random_state=42)
        acc_lr = self._evaluate_model(lr, "f1_weighted")
        acc_rf = self._evaluate_model(rf, "f1_weighted")
        return float(acc_rf - acc_lr)

    # ==============================
    # Robustes de modelos lineares
    # ==============================

    def robustness_to_noise_acc_linear(self, noise_ratio=0.25):
        """
        Adiciona ruído nos labels e mede a queda de performance do modelo.
        """
        self._prepare_data()

        # Cache para baseline do Logistic Regression
        if not hasattr(self, "_cv_results"):
            self._cv_results = {}

        if ('LogisticRegression', 'accuracy') not in self._cv_results:
            lr = LogisticRegression(max_iter=500, random_state=42)
            baseline = self._evaluate_model(lr, "accuracy")
        else:
            baseline = self._cv_results[('LogisticRegression', 'accuracy')]

        y_noisy = self._y.copy()
        n_noise = int(len(self._y) * noise_ratio)
        noisy_idx = random.sample(range(len(self._y)), n_noise)
        y_noisy.iloc[noisy_idx] = np.random.permutation(y_noisy.iloc[noisy_idx])

        lr = LogisticRegression(max_iter=500, random_state=42)
        noisy_score = cross_val_score(lr, self._X, y_noisy, cv=3, scoring="accuracy").mean()

        return float(baseline - noisy_score)

    def robustness_to_noise_balanced_acc_linear(self, noise_ratio=0.25):
        """
        Adiciona ruído nos labels e mede a queda de performance do modelo.
        """
        self._prepare_data()

        # Cache para baseline do Logistic Regression
        if not hasattr(self, "_cv_results"):
            self._cv_results = {}

        if ('LogisticRegression', 'balanced_accuracy') not in self._cv_results:
            lr = LogisticRegression(max_iter=500, random_state=42)
            baseline = self._evaluate_model(lr, "balanced_accuracy")
        else:
            baseline = self._cv_results[('LogisticRegression', 'balanced_accuracy')]

        y_noisy = self._y.copy()
        n_noise = int(len(self._y) * noise_ratio)
        noisy_idx = random.sample(range(len(self._y)), n_noise)
        y_noisy.iloc[noisy_idx] = np.random.permutation(y_noisy.iloc[noisy_idx])

        lr = LogisticRegression(max_iter=500, random_state=42)
        noisy_score = cross_val_score(lr, self._X, y_noisy, cv=3, scoring="balanced_accuracy").mean()

        return float(baseline - noisy_score)

    def robustness_to_noise_f1_macro_linear(self, noise_ratio=0.25):
        """
        Adiciona ruído nos labels e mede a queda de performance do modelo.
        """
        self._prepare_data()

        # Cache para baseline do Logistic Regression
        if not hasattr(self, "_cv_results"):
            self._cv_results = {}

        if ('LogisticRegression', 'f1_macro') not in self._cv_results:
            lr = LogisticRegression(max_iter=500, random_state=42)
            baseline = self._evaluate_model(lr, "f1_macro")
        else:
            baseline = self._cv_results[('LogisticRegression', 'f1_macro')]

        y_noisy = self._y.copy()
        n_noise = int(len(self._y) * noise_ratio)
        noisy_idx = random.sample(range(len(self._y)), n_noise)
        y_noisy.iloc[noisy_idx] = np.random.permutation(y_noisy.iloc[noisy_idx])

        lr = LogisticRegression(max_iter=500, random_state=42)
        noisy_score = cross_val_score(lr, self._X, y_noisy, cv=3, scoring="f1_macro").mean()

        return float(baseline - noisy_score)

    def robustness_to_noise_f1_micro_linear(self, noise_ratio=0.25):
        """
        Adiciona ruído nos labels e mede a queda de performance do modelo.
        """
        self._prepare_data()

        # Cache para baseline do Logistic Regression
        if not hasattr(self, "_cv_results"):
            self._cv_results = {}

        if ('LogisticRegression', 'f1_micro') not in self._cv_results:
            lr = LogisticRegression(max_iter=500, random_state=42)
            baseline = self._evaluate_model(lr, "f1_micro")
        else:
            baseline = self._cv_results[('LogisticRegression', 'f1_micro')]

        y_noisy = self._y.copy()
        n_noise = int(len(self._y) * noise_ratio)
        noisy_idx = random.sample(range(len(self._y)), n_noise)
        y_noisy.iloc[noisy_idx] = np.random.permutation(y_noisy.iloc[noisy_idx])

        lr = LogisticRegression(max_iter=500, random_state=42)
        noisy_score = cross_val_score(lr, self._X, y_noisy, cv=3, scoring="f1_micro").mean()

        return float(baseline - noisy_score)

    def robustness_to_noise_f1_weighted_linear(self, noise_ratio=0.25):
        """
        Adiciona ruído nos labels e mede a queda de performance do modelo.
        """
        self._prepare_data()

        # Cache para baseline do Logistic Regression
        if not hasattr(self, "_cv_results"):
            self._cv_results = {}

        if ('LogisticRegression', 'f1_weighted') not in self._cv_results:
            lr = LogisticRegression(max_iter=500, random_state=42)
            baseline = self._evaluate_model(lr, "f1_weighted")
        else:
            baseline = self._cv_results[('LogisticRegression', 'f1_weighted')]

        y_noisy = self._y.copy()
        n_noise = int(len(self._y) * noise_ratio)
        noisy_idx = random.sample(range(len(self._y)), n_noise)
        y_noisy.iloc[noisy_idx] = np.random.permutation(y_noisy.iloc[noisy_idx])

        lr = LogisticRegression(max_iter=500, random_state=42)
        noisy_score = cross_val_score(lr, self._X, y_noisy, cv=3, scoring="f1_weighted").mean()

        return float(baseline - noisy_score)

    def _get_simple_models(self):
        self._prepare_data()
        return {
            "lr": LogisticRegression(max_iter=500, random_state=42),
            "nb": MultinomialNB(),
            "dt": DecisionTreeClassifier(random_state=42)
        }

    def _prepare_data(self):
        if self.target is not None:
            self.data = self.data.dropna(subset=[self.target])
        if self._X is None:
            self._X = TfidfVectorizer(max_features=1000).fit_transform(self.data[self.text_column].astype(str))
            self._y = self.data[self.target]