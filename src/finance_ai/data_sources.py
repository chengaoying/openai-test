"""Utilities for downloading public market data from Sina Finance and Xueqiu."""
from __future__ import annotations

import datetime as _dt
import json
import time
from dataclasses import dataclass
from typing import Iterable, List

import requests


class DataSourceError(RuntimeError):
    """Exception raised when a remote data source cannot be reached."""


@dataclass
class OHLCVBar:
    """Represents a single OHLCV bar."""

    datetime: _dt.datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


class SinaFinanceClient:
    """Client for the unofficial Sina Finance k-line JSON API."""

    BASE_URL = (
        "https://quotes.sina.cn/cn/api/openapi.php/StockV2Service.getKLineData"
    )

    def fetch_daily_bars(self, symbol: str, count: int = 120) -> List[OHLCVBar]:
        """Fetch recent daily k-line data.

        Parameters
        ----------
        symbol:
            Ticker with exchange prefix, e.g. ``sh600000`` or ``sz000002``.
        count:
            Number of data points to request. Sina caps this to roughly 1000.
        """

        params = {
            "symbol": symbol,
            "scale": 240,  # 240 minutes = daily
            "ma": "no",
            "datalen": max(count, 60),
        }
        try:
            response = requests.get(self.BASE_URL, params=params, timeout=10)
            response.raise_for_status()
        except requests.RequestException as exc:  # pragma: no cover - network failure path
            raise DataSourceError(f"Failed to call Sina Finance API: {exc}") from exc

        payload = response.json()
        if payload.get("result") is None:
            raise DataSourceError(
                f"Unexpected payload from Sina Finance API: {json.dumps(payload)[:1000]}"
            )
        raw_items = payload["result"]["data"]["data"]

        bars: List[OHLCVBar] = []
        for item in raw_items:
            dt_str, open_, high, low, close, volume = item[:6]
            dt = _dt.datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
            bars.append(
                OHLCVBar(
                    datetime=dt,
                    open=float(open_),
                    high=float(high),
                    low=float(low),
                    close=float(close),
                    volume=float(volume),
                )
            )
        bars.sort(key=lambda b: b.datetime)
        return bars[-count:]


class XueqiuClient:
    """Minimal client for the unofficial Xueqiu k-line API.

    Unlike the Sina endpoint, Xueqiu requires an ``xq_a_token`` cookie. Users of this
    library are expected to supply the token by logging into Xueqiu in a browser and
    copying the cookie value into the client configuration.
    """

    BASE_URL = "https://stock.xueqiu.com/v5/stock/chart/kline.json"

    def __init__(self, token: str):
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/122 Safari/537.36"
            )
        })
        self._session.cookies.set("xq_a_token", token)

    def fetch_daily_bars(
        self, symbol: str, count: int = 120, indicator: str = "kline"
    ) -> List[OHLCVBar]:
        """Fetch daily bars from the Xueqiu API.

        Parameters
        ----------
        symbol:
            Symbol string such as ``SH600000`` or ``SZ000002``.
        count:
            Negative values request data backwards from ``begin``.
        indicator:
            Indicator field, defaults to ``kline`` which yields OHLC data.
        """

        end_time = int(time.time() * 1000)
        params = {
            "symbol": symbol,
            "period": "day",
            "type": "before",
            "indicator": indicator,
            "begin": end_time,
            "count": -abs(count),
        }
        try:
            response = self._session.get(self.BASE_URL, params=params, timeout=10)
            response.raise_for_status()
        except requests.RequestException as exc:  # pragma: no cover - network failure path
            raise DataSourceError(f"Failed to call Xueqiu API: {exc}") from exc

        payload = response.json()
        if payload.get("data") is None:
            raise DataSourceError(
                f"Unexpected payload from Xueqiu API: {json.dumps(payload)[:1000]}"
            )

        items = payload["data"].get("item", [])
        bars: List[OHLCVBar] = []
        for item in items:
            timestamp, open_, high, low, close, volume = item[:6]
            dt = _dt.datetime.fromtimestamp(timestamp / 1000, tz=_dt.timezone.utc)
            bars.append(
                OHLCVBar(
                    datetime=dt.astimezone(),
                    open=float(open_),
                    high=float(high),
                    low=float(low),
                    close=float(close),
                    volume=float(volume),
                )
            )
        bars.sort(key=lambda b: b.datetime)
        return bars[-count:]


def bars_to_rows(bars: Iterable[OHLCVBar]) -> List[dict]:
    """Convert OHLCV bars to dictionaries suitable for DataFrame creation."""

    return [
        {
            "datetime": bar.datetime,
            "open": bar.open,
            "high": bar.high,
            "low": bar.low,
            "close": bar.close,
            "volume": bar.volume,
        }
        for bar in bars
    ]
