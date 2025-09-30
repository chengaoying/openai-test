"""Simple modeling utilities for return prediction."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple


@dataclass
class LinearRegressionModel:
    """Ordinary least squares regression implemented with pure Python math."""

    coefficients_: List[float] | None = None
    intercept_: float | None = None

    def fit(self, X: Iterable[Iterable[float]], y: Iterable[float]) -> "LinearRegressionModel":
        X_mat = [list(map(float, row)) for row in X]
        if not X_mat:
            raise ValueError("Design matrix X is empty")
        y_vec = [float(value) for value in y]
        if len(y_vec) != len(X_mat):
            raise ValueError("X and y must contain the same number of rows")

        X_augmented = [[1.0] + row for row in X_mat]
        xtx = _matmul(_transpose(X_augmented), X_augmented)
        xty = _matvec(_transpose(X_augmented), y_vec)
        beta = _solve_linear_system(xtx, xty)
        self.intercept_ = beta[0]
        self.coefficients_ = beta[1:]
        return self

    def predict(self, X: Iterable[Iterable[float]]) -> List[float]:
        if self.coefficients_ is None or self.intercept_ is None:
            raise ValueError("Model has not been fitted")
        rows = [list(map(float, row)) for row in X]
        if not rows:
            return []
        return [self.intercept_ + _dot(row, self.coefficients_) for row in rows]

    def fit_predict(
        self,
        X_train: Iterable[Iterable[float]],
        y_train: Iterable[float],
        X_test: Iterable[Iterable[float]],
    ) -> Tuple["LinearRegressionModel", List[float]]:
        model = self.fit(X_train, y_train)
        return model, model.predict(X_test)


def _transpose(matrix: Sequence[Sequence[float]]) -> List[List[float]]:
    if not matrix:
        return []
    return [list(row) for row in zip(*matrix)]


def _matmul(a: Sequence[Sequence[float]], b: Sequence[Sequence[float]]) -> List[List[float]]:
    if not a or not b:
        return []
    result: List[List[float]] = []
    b_t = _transpose(b)
    for row in a:
        result_row = []
        for col in b_t:
            result_row.append(_dot(row, col))
        result.append(result_row)
    return result


def _matvec(matrix: Sequence[Sequence[float]], vec: Sequence[float]) -> List[float]:
    return [_dot(row, vec) for row in matrix]


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _solve_linear_system(matrix: Sequence[Sequence[float]], values: Sequence[float]) -> List[float]:
    # Gaussian elimination with partial pivoting
    size = len(matrix)
    aug = [list(row) + [values[idx]] for idx, row in enumerate(matrix)]

    for col in range(size):
        pivot_row = max(range(col, size), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot_row][col]) < 1e-12:
            raise ValueError("Matrix is singular and cannot be solved")
        if pivot_row != col:
            aug[col], aug[pivot_row] = aug[pivot_row], aug[col]

        pivot = aug[col][col]
        aug[col] = [value / pivot for value in aug[col]]

        for row in range(size):
            if row == col:
                continue
            factor = aug[row][col]
            aug[row] = [current - factor * pivot_value for current, pivot_value in zip(aug[row], aug[col])]

    return [row[-1] for row in aug]
