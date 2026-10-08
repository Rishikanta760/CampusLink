from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, brier_score_loss, f1_score, precision_score, recall_score

from app.model import FEATURES, make_model


def parse_date(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def normalize_row(source: dict[str, Any], provenance: str) -> tuple[dict[str, Any] | None, str | None]:
    student_id = source.get("studentId", source.get("student_id"))
    cohort = source.get("cohort")
    evidence_at = source.get("evidenceAt", source.get("evidence_at"))
    outcome_at = source.get("outcomeAt", source.get("outcome_at"))
    raw_label = source.get("placed", source.get("placement_status"))
    placed = 1 if raw_label in (1, "Placed") else 0 if raw_label in (0, "Unplaced") else None
    source_tag = source.get("outcomeSource")

    if not isinstance(student_id, (str, int)) or not str(student_id).strip():
        return None, "missing student id"
    if not isinstance(cohort, str) or not cohort.strip():
        return None, "missing cohort"
    evidence_date, outcome_date = parse_date(evidence_at), parse_date(outcome_at)
    if evidence_date is None or outcome_date is None:
        return None, "missing or invalid evidence/outcome date"
    if evidence_date >= outcome_date:
        return None, "evidence was not before outcome"
    if placed is None:
        return None, "unrecognized placement outcome"
    if provenance == "historical" and source_tag != "observed":
        return None, "historical rows must have outcomeSource='observed'"
    if provenance == "unverified_demo" and source_tag not in (None, "unverified_demo"):
        return None, "demo rows must not be mislabeled as historical"

    row: dict[str, Any] = {
        "studentId": str(student_id),
        "cohort": cohort.strip(),
        "evidenceAt": evidence_at,
        "outcomeAt": outcome_at,
        "outcomeSource": "observed" if provenance == "historical" else "unverified_demo",
        "placed": placed,
    }
    for feature in FEATURES:
        value = source.get(feature)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not np.isfinite(value) or not 0 <= value <= 100:
            return None, "missing or out-of-range score (0..100)"
        row[feature] = float(value)
    return row, None


def score_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float | int]:
    predicted = (probabilities >= 0.5).astype(int)
    positives = int(labels.sum())
    return {
        "accuracy": float(accuracy_score(labels, predicted)),
        "precision": float(precision_score(labels, predicted, zero_division=0)),
        "recall": float(recall_score(labels, predicted, zero_division=0)),
        "f1": float(f1_score(labels, predicted, zero_division=0)),
        "brier": float(brier_score_loss(labels, probabilities)),
        "testCount": int(len(labels)),
        "positiveCount": positives,
        "negativeCount": int(len(labels) - positives),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the CampusLink placement logistic-regression model.")
    parser.add_argument("--input", required=True, help="Input JSONL training dataset")
    parser.add_argument("--output", default="models", help="Directory for saved model artifacts")
    parser.add_argument("--provenance", choices=("historical", "unverified_demo"), default="historical")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output)
    exclusions: Counter[str] = Counter()
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(input_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            source = json.loads(line)
        except json.JSONDecodeError as error:
            raise SystemExit(f"Invalid JSON on input line {line_number}: {error}") from error
        normalized, reason = normalize_row(source, args.provenance)
        if reason:
            if args.provenance == "historical" and reason != "missing or out-of-range score (0..100)":
                raise SystemExit(f"Input line {line_number}: {reason}")
            exclusions[reason] += 1
            continue
        rows.append(normalized)

    if len(rows) < 60:
        raise SystemExit(f"Need at least 60 valid rows; found {len(rows)}.")
    cohorts = sorted({row["cohort"] for row in rows})
    if len(cohorts) < 3:
        raise SystemExit("Need at least three cohorts so the latest cohort can be held out.")

    test_cohort = cohorts[-1]
    test_rows = [row for row in rows if row["cohort"] == test_cohort]
    test_students = {row["studentId"] for row in test_rows}
    training_rows = [row for row in rows if row["cohort"] != test_cohort and row["studentId"] not in test_students]
    purged_rows = [row for row in rows if row["cohort"] != test_cohort and row["studentId"] in test_students]
    if len({row["placed"] for row in training_rows}) < 2 or len({row["placed"] for row in test_rows}) < 2:
        raise SystemExit("Training and held-out sets both need placed and unplaced examples.")

    x_train = np.asarray([[row[name] for name in FEATURES] for row in training_rows], dtype=float)
    y_train = np.asarray([row["placed"] for row in training_rows], dtype=int)
    x_test = np.asarray([[row[name] for name in FEATURES] for row in test_rows], dtype=float)
    y_test = np.asarray([row["placed"] for row in test_rows], dtype=int)

    estimator = make_model()
    estimator.fit(x_train, y_train)
    probabilities = estimator.predict_proba(x_test)[:, 1]
    baseline_probability = float(y_train.mean())
    baseline_probabilities = np.full(len(y_test), baseline_probability)

    fingerprint_data = json.dumps(rows, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    fingerprint = hashlib.sha256(fingerprint_data).hexdigest()[:12]
    version_prefix = "campuslink-logistic-py-v1" if args.provenance == "historical" else "campuslink-logistic-py-demo-v1"
    model_version = f"{version_prefix}-{fingerprint}"
    output_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(estimator, output_dir / "placement.model.joblib")
    normalized_path = output_dir / "placement-training.jsonl"
    normalized_path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
    metadata = {
        "version": model_version,
        "provenance": args.provenance,
        "trainedAt": datetime.now().astimezone().isoformat(),
        "featureOrder": list(FEATURES),
        "split": {
            "strategy": "latest-cohort-with-student-group-isolation",
            "trainingCohorts": [cohort for cohort in cohorts if cohort != test_cohort],
            "testCohorts": [test_cohort],
            "trainingStudentCount": len({row["studentId"] for row in training_rows}),
            "testStudentCount": len(test_students),
            "purgedRowsForGroupIsolation": len(purged_rows),
        },
        "evaluation": {
            "model": score_metrics(y_test, probabilities),
            "majorityBaseline": score_metrics(y_test, baseline_probabilities),
            "trainingCount": len(training_rows),
        },
    }
    (output_dir / "placement.metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (output_dir / "placement.preprocessing.json").write_text(
        json.dumps({"version": "score-normalization-v1", "featureOrder": list(FEATURES),
                    "transform": "divide each 0-100 evidence score by 100", "missingValues": "rejected"}, indent=2),
        encoding="utf-8",
    )
    report = {
        "warning": "Unverified demo data is illustrative only." if args.provenance == "unverified_demo" else None,
        "acceptedRows": len(rows),
        "excludedRows": sum(exclusions.values()),
        "exclusions": dict(exclusions),
        "modelFile": str((output_dir / "placement.model.joblib").resolve()),
        "metadataFile": str((output_dir / "placement.metadata.json").resolve()),
        "modelVersion": model_version,
        "provenance": args.provenance,
        "split": metadata["split"],
        "evaluation": metadata["evaluation"],
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
