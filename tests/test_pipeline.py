import json
import pathlib

from finance_ai.features import compute_features
from finance_ai.models import LinearRegressionModel


def load_sample() -> list[dict]:
    path = pathlib.Path(__file__).parents[1] / "src" / "sample_data.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    for row in payload:
        row["datetime"] = parse_datetime(row["datetime"])
    return payload


def parse_datetime(value: str):
    import datetime

    if isinstance(value, datetime.datetime):
        return value
    return datetime.datetime.fromisoformat(value)


def test_feature_pipeline_runs():
    rows = load_sample()
    dataset = compute_features(rows)

    model = LinearRegressionModel().fit(dataset.features, dataset.target)
    preds = model.predict(dataset.features[:10])

    assert len(preds) == 10
    assert all(not _is_nan(value) for value in preds)


def _is_nan(value: float) -> bool:
    return value != value
