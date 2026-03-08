"""Pure-stdlib binary classification and summary metrics."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import median


@dataclass(slots=True)
class CalibrationBin:
    lower: float
    upper: float
    count: int
    avg_pred: float
    actual_rate: float

    def to_dict(self) -> dict[str, float | int]:
        return {
            "lower": self.lower,
            "upper": self.upper,
            "count": self.count,
            "avg_pred": self.avg_pred,
            "actual_rate": self.actual_rate,
        }


def quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize_distribution(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p10": None,
            "p90": None,
            "pct_positive": None,
        }
    mean_value = sum(values) / len(values)
    positive = sum(1 for value in values if value > 0)
    return {
        "count": len(values),
        "mean": mean_value,
        "median": median(values),
        "p10": quantile(values, 0.10),
        "p90": quantile(values, 0.90),
        "pct_positive": positive / len(values),
    }


def brier_score(y_true: list[int], y_prob: list[float]) -> float | None:
    if not y_true:
        return None
    errors = ((prob - truth) ** 2 for truth, prob in zip(y_true, y_prob, strict=True))
    return sum(errors) / len(y_true)


def auroc(y_true: list[int], y_prob: list[float]) -> float | None:
    positives = sum(y_true)
    negatives = len(y_true) - positives
    if positives == 0 or negatives == 0:
        return None

    ranked = sorted(zip(y_prob, y_true, strict=True), key=lambda item: item[0])
    rank = 1
    rank_sum_positive = 0.0
    index = 0
    while index < len(ranked):
        j = index
        while j < len(ranked) and ranked[j][0] == ranked[index][0]:
            j += 1
        average_rank = (rank + (rank + (j - index) - 1)) / 2.0
        positives_in_group = sum(label for _, label in ranked[index:j])
        rank_sum_positive += positives_in_group * average_rank
        rank += j - index
        index = j

    return (rank_sum_positive - positives * (positives + 1) / 2.0) / (positives * negatives)


def average_precision(y_true: list[int], y_prob: list[float]) -> float | None:
    positives = sum(y_true)
    if positives == 0:
        return None

    ranked = sorted(
        zip(y_prob, y_true, strict=True),
        key=lambda item: item[0],
        reverse=True,
    )
    true_positives = 0
    precision_sum = 0.0
    for index, (_, label) in enumerate(ranked, start=1):
        if label:
            true_positives += 1
            precision_sum += true_positives / index
    return precision_sum / positives


def calibration_bins(
    y_true: list[int],
    y_prob: list[float],
    *,
    num_bins: int = 10,
) -> list[CalibrationBin]:
    bins: list[list[tuple[int, float]]] = [[] for _ in range(num_bins)]
    for truth, prob in zip(y_true, y_prob, strict=True):
        clipped = min(max(prob, 0.0), 1.0)
        idx = min(int(clipped * num_bins), num_bins - 1)
        bins[idx].append((truth, clipped))

    results: list[CalibrationBin] = []
    for idx, entries in enumerate(bins):
        lower = idx / num_bins
        upper = (idx + 1) / num_bins
        if not entries:
            results.append(CalibrationBin(lower, upper, 0, 0.0, 0.0))
            continue
        avg_pred = sum(prob for _, prob in entries) / len(entries)
        actual_rate = sum(truth for truth, _ in entries) / len(entries)
        results.append(CalibrationBin(lower, upper, len(entries), avg_pred, actual_rate))
    return results
