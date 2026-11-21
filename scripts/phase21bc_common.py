"""Shared, read-only calculations for the Phase 21B-C frozen-evidence audit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "phase21b_correction"
PRED = ROOT / "results" / "predictions" / "phase21b"
DATA = ROOT / "data" / "processed" / "judipl" / "phase18c"
CLASSES = ["Accepted", "Other", "Rejected"]
PROB_COLUMNS = ["prob_accepted", "prob_other", "prob_rejected"]
N = 6311


def load_manifest() -> dict:
    return json.loads((ROOT / "results" / "reports" / "phase21b_duplicate_exclusion_manifest.json").read_text(encoding="utf-8"))


def eligible_truth() -> pd.DataFrame:
    manifest = load_manifest()
    df = pd.read_parquet(DATA / "three_class" / "test_masked.parquet")
    excluded = set(map(str, manifest["excluded_test_ids"]))
    df["id"] = df["id"].astype(str).str.zfill(10)
    return df.loc[~df["id"].isin(excluded), ["id", "label"]].reset_index(drop=True)


def load_predictions(model: str) -> pd.DataFrame:
    path = PRED / f"{model.lower()}_test_predictions.csv"
    df = pd.read_csv(path, dtype={"id": str})
    df["id"] = df["id"].astype(str).str.zfill(10)
    return df


def validate_prediction_frame(df: pd.DataFrame, model: str, expected_ids: set[str]) -> list[dict]:
    errors: list[dict] = []
    required = {"id", "predicted_label", *PROB_COLUMNS}
    missing = sorted(required - set(df.columns))
    if missing:
        errors.append({"model": model, "check": "required_columns", "status": "FAIL", "detail": str(missing)})
        return errors
    ids = set(df["id"])
    if len(df) != len(expected_ids):
        errors.append({"model": model, "check": "row_count", "status": "FAIL", "detail": f"{len(df)} != {len(expected_ids)}"})
    if df["id"].duplicated().any():
        errors.append({"model": model, "check": "duplicate_ids", "status": "FAIL", "detail": str(df.loc[df["id"].duplicated(), "id"].tolist())})
    if ids != expected_ids:
        errors.append({"model": model, "check": "eligible_ids", "status": "FAIL", "detail": f"missing={sorted(expected_ids-ids)[:5]}, unexpected={sorted(ids-expected_ids)[:5]}"})
    probs = df[PROB_COLUMNS].to_numpy(dtype=float)
    finite = np.isfinite(probs).all()
    bounds = ((probs >= 0) & (probs <= 1)).all()
    sums = np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)
    argmax = np.array(CLASSES)[np.argmax(probs, axis=1)]
    labels = df["predicted_label"].to_numpy()
    checks = [
        ("probabilities_finite", finite, "all finite"),
        ("probabilities_in_0_1", bounds, "all within [0,1]"),
        ("probabilities_sum_to_one", sums, "row sums within 1e-5"),
        ("predicted_label_argmax", np.array_equal(argmax, labels), "label equals probability argmax"),
        ("predicted_labels_valid", set(labels).issubset(CLASSES), "canonical labels only"),
    ]
    for check, ok, detail in checks:
        if not ok:
            errors.append({"model": model, "check": check, "status": "FAIL", "detail": detail})
    return errors


def aligned_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    truth = eligible_truth()
    m4, m8 = load_predictions("m4"), load_predictions("m8")
    expected = set(truth["id"])
    errors = validate_prediction_frame(m4, "M4", expected) + validate_prediction_frame(m8, "M8", expected)
    if set(m4["id"]) != set(m8["id"]):
        errors.append({"model": "M4/M8", "check": "cross_model_ids", "status": "FAIL", "detail": "ID sets differ"})
    if errors:
        raise ValueError("Prediction alignment failed: " + json.dumps(errors))
    merged = truth.merge(m4, on="id", validate="one_to_one").merge(m8, on="id", suffixes=("_m4", "_m8"), validate="one_to_one")
    return merged, m4, m8


def classification_metrics(merged: pd.DataFrame, model: str) -> tuple[dict, list[dict], np.ndarray]:
    y = merged["label"].to_numpy()
    pred = merged[f"predicted_label_{model.lower()}"].to_numpy()
    probs = merged[[f"{c}_{model.lower()}" for c in PROB_COLUMNS]].to_numpy(dtype=float)
    probs = probs / probs.sum(axis=1, keepdims=True)
    y_onehot = pd.get_dummies(pd.Categorical(y, categories=CLASSES)).to_numpy()
    cm = confusion_matrix(y, pred, labels=CLASSES)
    per_class = []
    for i, c in enumerate(CLASSES):
        per_class.append({
            "model_id": model, "class": c,
            "precision": precision_score(y, pred, labels=CLASSES, average=None, zero_division=0)[i],
            "recall": recall_score(y, pred, labels=CLASSES, average=None, zero_division=0)[i],
            "f1_score": f1_score(y, pred, labels=CLASSES, average=None, zero_division=0)[i],
            "support": int(cm[i].sum()), "false_positives": int(cm[:, i].sum() - cm[i, i]),
            "false_negatives": int(cm[i].sum() - cm[i, i]),
        })
    try:
        auc_ovr = roc_auc_score(y_onehot, probs, multi_class="ovr", average="macro")
        auc_ovo = roc_auc_score(y_onehot, probs, multi_class="ovo", average="macro")
    except ValueError:
        raise
    rec = {
        "model_id": model, "model_name": "Section-Aware TF-IDF Classifier" if model == "M4" else "Hybrid Multi-Class Fusion Ensemble",
        "eligible_test_size": len(y), "accuracy": accuracy_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred), "macro_precision": precision_score(y, pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y, pred, average="macro", zero_division=0), "macro_f1": f1_score(y, pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y, pred, average="weighted", zero_division=0), "roc_auc_ovr": auc_ovr, "roc_auc_ovo": auc_ovo,
        "pr_auc_macro": average_precision_score(y_onehot, probs, average="macro"), "log_loss": log_loss(y_onehot, probs),
        "brier_score_mean": float(np.mean(np.sum((probs - y_onehot) ** 2, axis=1))),
        "brier_score_sum_classes": float(np.mean((probs - y_onehot) ** 2, axis=0).sum()),
        "confusion_matrix": json.dumps(cm.tolist()),
    }
    return rec, per_class, cm


def calibration(merged: pd.DataFrame, model: str, bins: int = 10) -> dict:
    y = merged["label"].to_numpy()
    probs = merged[[f"{c}_{model.lower()}" for c in PROB_COLUMNS]].to_numpy(dtype=float)
    probs = probs / probs.sum(axis=1, keepdims=True)
    pred = np.array(CLASSES)[np.argmax(probs, axis=1)]
    conf = probs.max(axis=1)
    boundaries = np.linspace(0, 1, bins + 1)
    ece = 0.0
    rows = []
    for i in range(bins):
        mask = (conf > boundaries[i]) & (conf <= boundaries[i + 1])
        if mask.any():
            acc, mean_conf = float((pred[mask] == y[mask]).mean()), float(conf[mask].mean())
            ece += mask.mean() * abs(acc - mean_conf)
            rows.append((i + 1, int(mask.sum()), acc, mean_conf))
    onehot = pd.get_dummies(pd.Categorical(y, categories=CLASSES)).to_numpy()
    return {"model_id": model, "brier_score": float(np.mean(np.sum((probs - onehot) ** 2, axis=1))),
            "brier_convention": "mean per-document sum across classes", "log_loss": float(log_loss(onehot, probs)),
            "ece": float(ece), "ece_bins": bins, "binning": "10 equal-width bins; (left,right]", "confidence": "maximum class probability",
            "mean_confidence": float(conf.mean()), "calibration_bins": json.dumps(rows)}


def mcnemar(merged: pd.DataFrame) -> dict:
    y = merged["label"].to_numpy()
    c4 = merged["predicted_label_m4"].to_numpy() == y
    c8 = merged["predicted_label_m8"].to_numpy() == y
    n11, n00, n01, n10 = int((c4 & c8).sum()), int((~c4 & ~c8).sum()), int((~c4 & c8).sum()), int((c4 & ~c8).sum())
    discordant = n01 + n10
    chi2 = ((abs(n01 - n10) - 1) ** 2 / discordant) if discordant else 0.0
    p_cc = float(stats.chi2.sf(chi2, 1))
    p_exact = float(stats.binomtest(min(n01, n10), discordant, 0.5).pvalue) if discordant else 1.0
    return {"n11_both_correct": n11, "n00_both_incorrect": n00, "n01_m4_incorrect_m8_correct": n01,
            "n10_m4_correct_m8_incorrect": n10, "n": n11 + n00 + n01 + n10,
            "m4_correct": n11 + n10, "m8_correct": n11 + n01, "net_m8_minus_m4": n01 - n10,
            "chi2_continuity_corrected": chi2, "p_value_continuity_corrected": p_cc,
            "p_value_exact_binomial": p_exact, "discordant": discordant}


def bootstrap(merged: pd.DataFrame, iterations: int = 1000, seed: int = 42) -> pd.DataFrame:
    y, p4, p8 = merged["label"].to_numpy(), merged["predicted_label_m4"].to_numpy(), merged["predicted_label_m8"].to_numpy()
    rng = np.random.RandomState(seed)
    values = {"Accuracy": [], "Balanced Accuracy": [], "Macro-F1": [], "Weighted-F1": []}
    for _ in range(iterations):
        idx = rng.randint(0, len(y), len(y))
        for name, fn in [
            ("Accuracy", lambda a, b: accuracy_score(a, b)),
            ("Balanced Accuracy", lambda a, b: balanced_accuracy_score(a, b)),
            ("Macro-F1", lambda a, b: f1_score(a, b, average="macro", zero_division=0)),
            ("Weighted-F1", lambda a, b: f1_score(a, b, average="weighted", zero_division=0)),
        ]:
            values[name].append(fn(y[idx], p8[idx]) - fn(y[idx], p4[idx]))
    rows = []
    point = classification_metrics(merged, "M4")[0], classification_metrics(merged, "M8")[0]
    key_map = {"Accuracy": "accuracy", "Balanced Accuracy": "balanced_accuracy", "Macro-F1": "macro_f1", "Weighted-F1": "weighted_f1"}
    for name, arr in values.items():
        a = np.asarray(arr)
        p = float(np.mean(a <= 0))
        rows.append({"metric": name, "point_estimate_delta_m8_minus_m4": point[1][key_map[name]] - point[0][key_map[name]],
                     "ci_95_lower": float(np.percentile(a, 2.5)), "ci_95_upper": float(np.percentile(a, 97.5)),
                     "bootstrap_p_value_pr_delta_le_zero": p, "iterations": iterations, "seed": seed,
                     "method": "paired document-index resampling; percentile CI"})
    return pd.DataFrame(rows)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write(df: pd.DataFrame, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / name, index=False)
