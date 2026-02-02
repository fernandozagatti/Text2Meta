import inspect
from tqdm import tqdm
from text2meta.simple import SimpleFeatures
from text2meta.statistical import StatisticalFeatures
from text2meta.info_theoretic import InfoFeatures
from text2meta.model_based import ModelBasedFeatures
from text2meta.landmarking import LandmarkingFeatures
from text2meta.complexity import ComplexityFeatures

class Text2Meta:
    def __init__(self, data, text_column, target=None):
        self.data = data
        self.text_column = text_column
        self.target = target

        self.modules = [
            SimpleFeatures(data, text_column, target),
            StatisticalFeatures(data, text_column, target),
            InfoFeatures(data, text_column, target),
            ModelBasedFeatures(data, text_column, target),
            LandmarkingFeatures(data, text_column, target),
            ComplexityFeatures(data, text_column, target),
        ]

    def compute(self, metrics=None, groups=None):
        """
        metrics: lista de nomes dos métodos a serem chamados.
        Exemplo: ["number_of_documents", "number_of_classes"]

        groups: lista de nomes dos grupos a serem chamados.
        Exemplo: ["general", "statistical", "info-theory", "model-based", "landmarking", "complexity"]
        """
        if metrics is None:
            metrics = []

        if len(metrics) == 0 and groups is None:
            metrics = ["number_of_documents", "number_of_classes"]

        if groups is not None:
            if "general" in groups:
                methods = [
                    name for name, func in inspect.getmembers(SimpleFeatures, predicate=inspect.isfunction)
                    if not name.startswith("_")
                ]
                metrics.extend(methods)
            if "statistical" in groups:
                methods = [
                    name for name, func in inspect.getmembers(StatisticalFeatures, predicate=inspect.isfunction)
                    if not name.startswith("_")
                ]
                metrics.extend(methods)
            if "info-theory" in groups:
                methods = [
                    name for name, func in inspect.getmembers(InfoFeatures, predicate=inspect.isfunction)
                    if not name.startswith("_")
                ]
                metrics.extend(methods)
            if "model-based" in groups:
                methods = [
                    name for name, func in inspect.getmembers(ModelBasedFeatures, predicate=inspect.isfunction)
                    if not name.startswith("_")
                ]
                metrics.extend(methods)
            if "landmarking" in groups:
                methods = [
                    name for name, func in inspect.getmembers(LandmarkingFeatures, predicate=inspect.isfunction)
                    if not name.startswith("_")
                ]
                metrics.extend(methods)
            if "complexity" in groups:
                methods = [
                    name for name, func in inspect.getmembers(ComplexityFeatures, predicate=inspect.isfunction)
                    if not name.startswith("_")
                ]
                metrics.extend(methods)

        metrics = list(dict.fromkeys(metrics))
        results = {}
        #for metric in tqdm(metrics, desc="Calculando métricas"):
        for metric in metrics:
            found = False
            for module in self.modules:
                method = getattr(module, metric, None)
                if method is not None:
                    results[metric] = method()
                    found = True
                    break
            if not found:
                raise ValueError(f"Metric '{metric}' not found.")
        return results
