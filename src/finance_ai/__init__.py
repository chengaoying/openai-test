"""Finance AI package for fetching public market data and building analysis models."""

from .features import FeatureDataset, compute_features
from .models import LinearRegressionModel

__all__ = ["FeatureDataset", "compute_features", "LinearRegressionModel"]

try:  # pragma: no cover - optional dependency for offline tests
    from .data_sources import DataSourceError, SinaFinanceClient, XueqiuClient

    __all__.extend(["DataSourceError", "SinaFinanceClient", "XueqiuClient"])
except ModuleNotFoundError:  # requests is optional for unit tests
    pass
