"""Feature engineering helpers for OHLCV data."""
from __future__ import annotations

import datetime as _dt
import math
from dataclasses import dataclass
from typing import Iterable, Mapping, MutableMapping, Sequence


@dataclass
class FeatureDataset:
    """Container holding engineered features ready for modeling."""

    feature_names: Sequence[str]
    features: list[list[float]]
    target: list[float]
    datetimes: list[_dt.datetime]

    def __post_init__(self) -> None:
        if len(self.features) != len(self.target):
            raise ValueError("Features and target must have the same length")
        if len(self.features) != len(self.datetimes):
            raise ValueError("Datetimes must align with features")

    def split(self, ratio: float) -> tuple["FeatureDataset", "FeatureDataset"]:
        """Split the dataset into two parts according to ``ratio``."""

        if not 0.0 < ratio < 1.0:
            raise ValueError("Split ratio must be between 0 and 1")
        index = int(len(self.datetimes) * ratio)
        first = FeatureDataset(
            feature_names=self.feature_names,
            features=self.features[:index],
            target=self.target[:index],
            datetimes=self.datetimes[:index],
        )
        second = FeatureDataset(
            feature_names=self.feature_names,
            features=self.features[index:],
            target=self.target[index:],
            datetimes=self.datetimes[index:],
        )
        return first, second


def _ensure_rows(
    data: Sequence[Mapping[str, object]] | Mapping[str, Iterable[object]]
) -> list[MutableMapping[str, object]]:
    if hasattr(data, "to_dict"):
        # Support pandas objects without requiring the dependency at runtime.
        try:  # pragma: no cover - executed only when pandas is installed
            records = data.to_dict(orient="records")  # type: ignore[call-arg]
        except TypeError:  # pragma: no cover - pandas < 1.1 signature
            records = data.to_dict("records")  # type: ignore[call-arg]
        return [dict(row) for row in records]

    if isinstance(data, Mapping):
        keys = list(data.keys())
        values = list(data.values())
        rows = []
        for items in zip(*values):
            rows.append({key: value for key, value in zip(keys, items)})
        return rows

    return [dict(row) for row in data]


def _as_datetime(value: object) -> _dt.datetime:
    if isinstance(value, _dt.datetime):
        return value
    if isinstance(value, str):
        try:
            return _dt.datetime.fromisoformat(value)
        except ValueError as exc:  # pragma: no cover - unexpected format
            raise ValueError(f"Invalid datetime string: {value}") from exc
    raise TypeError(f"Unsupported datetime type: {type(value)!r}")


def compute_features(
    data: Sequence[Mapping[str, object]] | Mapping[str, Iterable[object]],
    horizon: int = 5,
) -> FeatureDataset:
    """Compute technical features from tabular OHLCV data.

    The function accepts any iterable of mappings containing ``datetime``, ``close``
    and ``volume`` keys. A :class:`FeatureDataset` is returned with the engineered
    feature matrix and aligned target returns.
    """

    rows = _ensure_rows(data)
    if not rows:
        raise ValueError("Input data is empty")

    for field in ("datetime", "close", "volume"):
        if field not in rows[0]:
            raise ValueError(f"Missing required column: {field}")

    rows.sort(key=lambda item: _as_datetime(item["datetime"]))

    datetimes = [_as_datetime(row["datetime"]) for row in rows]
    closes = [float(row["close"]) for row in rows]
    volumes = [float(row["volume"]) for row in rows]

    n = len(rows)
    if n <= horizon:
        raise ValueError("Not enough rows to compute target returns")

    return_1d = [0.0] * n
    for idx in range(1, n):
        prev = closes[idx - 1]
        if prev:
            return_1d[idx] = (closes[idx] - prev) / prev

    return_5d = [0.0] * n
    for idx in range(5, n):
        base = closes[idx - 5]
        if base:
            return_5d[idx] = (closes[idx] - base) / base

    volume_z = [0.0] * n
    window = 20
    for idx in range(n):
        start = max(0, idx - window + 1)
        series = volumes[start : idx + 1]
        mean = sum(series) / len(series)
        variance = sum((value - mean) ** 2 for value in series) / len(series)
        std = math.sqrt(variance)
        if std > 0:
            volume_z[idx] = (volumes[idx] - mean) / std

    def _moving_average(period: int) -> list[float]:
        result = [0.0] * n
        for idx in range(n):
            start = max(0, idx - period + 1)
            window_values = closes[start : idx + 1]
            result[idx] = sum(window_values) / len(window_values)
        return result

    ma_5 = _moving_average(5)
    ma_10 = _moving_average(10)
    ma_ratio = []
    for m5, m10 in zip(ma_5, ma_10):
        if m10:
            ma_ratio.append(m5 / m10)
        else:
            ma_ratio.append(1.0)

    target = [0.0] * (n - horizon)
    feature_matrix: list[list[float]] = []
    datetime_values: list[_dt.datetime] = []

    feature_names = ["return_1d", "return_5d", "volume_zscore", "ma_ratio"]
    for idx in range(n - horizon):
        future = closes[idx + horizon]
        base = closes[idx]
        target[idx] = (future - base) / base if base else 0.0
        feature_matrix.append(
            [
                return_1d[idx],
                return_5d[idx],
                volume_z[idx],
                ma_ratio[idx],
            ]
        )
        datetime_values.append(datetimes[idx + horizon])

    return FeatureDataset(
        feature_names=feature_names,
        features=feature_matrix,
        target=target,
        datetimes=datetime_values,
    )
