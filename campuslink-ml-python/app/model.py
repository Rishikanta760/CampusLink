from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

FEATURES = (
    "verifiedSkills",
    "academics",
    "projects",
    "aptitude",
    "communication",
    "interview",
)


class ScoreNormalizer(BaseEstimator, TransformerMixin):
    """Keep the existing score transformation: six 0–100 values divided by 100."""

    def fit(self, values: Any, labels: Any = None) -> "ScoreNormalizer":
        return self

    def transform(self, values: Any) -> np.ndarray:
        return np.asarray(values, dtype=float) / 100.0


def make_model() -> Pipeline:
    return Pipeline(
        [
            ("normalize_scores", ScoreNormalizer()),
            (
                "logistic_regression",
                LogisticRegression(
                    solver="lbfgs",
                    max_iter=2000,
                    C=100.0,
                    random_state=2026,
                ),
            ),
        ]
    )


def load_artifacts(model_dir: str | Path, allow_demo: bool = False) -> tuple[Pipeline, dict[str, Any]] | None:
    directory = Path(model_dir)
    model_path = directory / "placement.model.joblib"
    metadata_path = directory / "placement.metadata.json"
    preprocess_path = directory / "placement.preprocessing.json"
    if not model_path.is_file() or not metadata_path.is_file() or not preprocess_path.is_file():
        return None

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("featureOrder") != list(FEATURES):
        raise ValueError("Model feature order does not match the API schema.")
    if metadata.get("provenance") not in {"historical", "unverified_demo"}:
        raise ValueError("Model provenance is missing or invalid.")
    if metadata.get("provenance") == "unverified_demo" and not allow_demo:
        raise ValueError("Unverified demo models are disabled.")

    preprocessing = json.loads(preprocess_path.read_text(encoding="utf-8"))
    expected_transform = "divide each 0-100 evidence score by 100"
    if preprocessing.get("featureOrder") != list(FEATURES) or preprocessing.get("transform") != expected_transform:
        raise ValueError("Preprocessing artifact does not match the API feature schema.")

    estimator = joblib.load(model_path)
    if not isinstance(estimator, Pipeline) or "logistic_regression" not in estimator.named_steps:
        raise ValueError("Saved model artifact has an unsupported type.")
    return estimator, metadata
