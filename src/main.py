"""Command line interface for downloading data and training a simple model."""
from __future__ import annotations

import argparse
import csv
import datetime as _datetime
import json
import pathlib
import sys
from typing import Optional

_SRC_DIR = pathlib.Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

try:  # pragma: no cover - optional dependency for offline usage
    from finance_ai.data_sources import (
        DataSourceError,
        SinaFinanceClient,
        XueqiuClient,
        bars_to_rows,
    )
except ModuleNotFoundError:  # requests may be unavailable in offline environments
    class _RequestsMissingError(RuntimeError):
        pass

    DataSourceError = _RequestsMissingError
    SinaFinanceClient = None  # type: ignore[assignment]
    XueqiuClient = None  # type: ignore[assignment]

    def bars_to_rows(*_args, **_kwargs):  # type: ignore[no-redef]
        raise _RequestsMissingError(
            "Remote data sources require the 'requests' package to be installed."
        )
from finance_ai.features import FeatureDataset, compute_features
from finance_ai.models import LinearRegressionModel


def _load_offline_sample() -> list[dict]:
    sample_path = pathlib.Path(__file__).with_name("sample_data.json")
    payload = json.loads(sample_path.read_text(encoding="utf-8"))
    return _normalise_rows(payload)


def _normalise_rows(rows: list[dict]) -> list[dict]:
    normalised = []
    for row in rows:
        row = dict(row)
        if "datetime" in row:
            row["datetime"] = _parse_datetime(row["datetime"])
        normalised.append(row)
    return normalised


def _parse_datetime(value: object) -> _datetime.datetime:
    if isinstance(value, _datetime.datetime):
        return value
    if isinstance(value, str):
        return _datetime.datetime.fromisoformat(value)
    raise TypeError(f"Unsupported datetime type: {type(value)!r}")


def download_data(
    source: str, symbol: str, count: int, token: Optional[str]
) -> list[dict]:
    if source == "sina":
        if SinaFinanceClient is None:
            raise DataSourceError("Sina data source requires the 'requests' package")
        client = SinaFinanceClient()
        bars = client.fetch_daily_bars(symbol, count=count)
        return bars_to_rows(bars)

    if source == "xueqiu":
        if XueqiuClient is None:
            raise DataSourceError("Xueqiu data source requires the 'requests' package")
        if not token:
            raise ValueError("Xueqiu requires a token provided via --token")
        client = XueqiuClient(token)
        bars = client.fetch_daily_bars(symbol, count=count)
        return bars_to_rows(bars)

    if source == "offline":
        return _load_offline_sample()

    raise ValueError(f"Unknown data source: {source}")


def build_model(
    rows: list[dict], horizon: int
) -> tuple[LinearRegressionModel, FeatureDataset, list[dict]]:
    dataset = compute_features(rows, horizon=horizon)

    split_index = int(len(dataset.target) * 0.8)
    X_train = dataset.features[:split_index]
    y_train = dataset.target[:split_index]
    X_test = dataset.features[split_index:]
    y_test = dataset.target[split_index:]

    model = LinearRegressionModel().fit(X_train, y_train)
    predictions = model.predict(X_test)

    evaluation_rows = [
        {
            "datetime": dt.isoformat(),
            "actual": float(actual),
            "predicted": float(pred),
        }
        for dt, actual, pred in zip(
            dataset.datetimes[split_index:], y_test, predictions, strict=False
        )
    ]

    test_dataset = FeatureDataset(
        feature_names=dataset.feature_names,
        features=X_test,
        target=y_test,
        datetimes=dataset.datetimes[split_index:],
    )

    return model, test_dataset, evaluation_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("symbol", help="Ticker symbol (e.g. sh600000)")
    parser.add_argument(
        "--source",
        choices=["sina", "xueqiu", "offline"],
        default="offline",
        help="Data source to query",
    )
    parser.add_argument(
        "--count", type=int, default=200, help="Number of bars to download"
    )
    parser.add_argument(
        "--token", help="Authentication token for Xueqiu", default=None
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=5,
        help="Prediction horizon in trading days",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path("output"),
        help="Directory to store artifacts",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        rows = download_data(args.source, args.symbol, args.count, args.token)
    except (DataSourceError, ValueError) as exc:
        raise SystemExit(f"Failed to download data: {exc}") from exc

    model, dataset, evaluation_rows = build_model(rows, horizon=args.horizon)
    args.output.mkdir(parents=True, exist_ok=True)

    evaluation_path = args.output / "evaluation.csv"
    with evaluation_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["datetime", "actual", "predicted"])
        writer.writeheader()
        writer.writerows(evaluation_rows)

    model_params = {
        "intercept": model.intercept_,
        "coefficients": list(model.coefficients_) if model.coefficients_ is not None else None,
        "feature_order": list(dataset.feature_names),
    }
    model_path = args.output / "model.json"
    model_path.write_text(json.dumps(model_params, indent=2), encoding="utf-8")

    print(f"Saved evaluation metrics to {evaluation_path}")
    print(f"Saved model parameters to {model_path}")


if __name__ == "__main__":
    main()
