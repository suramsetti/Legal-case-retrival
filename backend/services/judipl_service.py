"""Read-only Jud-IPL inference and evaluation metadata service."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd

from src.models.phase20_multiclass_models import CANONICAL_CLASSES


class JudIPLService:
    """Loads frozen Phase 20 classifiers and Phase 21B-C result artifacts."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.checkpoint_dir = project_root / "models" / "checkpoints" / "phase20"
        self.correction_dir = project_root / "results" / "phase21b_correction"
        self.data_path = project_root / "data" / "processed" / "judipl" / "phase18c" / "three_class" / "test_masked.parquet"
        self._models: Dict[str, Any] = {}
        self._metadata = self._load_metadata()

    def _load_metadata(self) -> Dict[str, Any]:
        data = pd.read_parquet(self.data_path)
        self._data = data
        manifest_path = self.project_root / "results" / "reports" / "phase21b_duplicate_exclusion_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        eligible = len(data) - len(manifest["excluded_test_ids"])
        return {
            "dataset_id": "jud-ipl",
            "name": "Jud-IPL",
            "task": "Legal Judgment Outcome Classification",
            "eligible_test_size": eligible,
            "original_test_size": len(data),
            "excluded_duplicate_count": len(manifest["excluded_test_ids"]),
            "classes": CANONICAL_CLASSES,
            "models": [
                {"id": "M4", "name": "Section-Aware TF-IDF", "role": "Primary"},
                {"id": "M8", "name": "Hybrid Late Fusion", "role": "Comparison"},
            ],
        }

    def _load_models(self) -> None:
        if self._models:
            return
        self._models["M4"] = joblib.load(self.checkpoint_dir / "m4_section_aware_tfidf.joblib")
        self._models["M1"] = joblib.load(self.checkpoint_dir / "m1_tfidf_lr_unweighted.joblib")
        self._models["M3"] = joblib.load(self.checkpoint_dir / "m3_char_ngram_lr.joblib")
        self._models["M5"] = joblib.load(self.checkpoint_dir / "m5_citation_aware_lr.joblib")
        bundle = joblib.load(self.checkpoint_dir / "m8_hybrid_fusion.joblib")
        self._weights = bundle["weights"]

    def metadata(self) -> Dict[str, Any]:
        return self._metadata

    def metrics(self) -> Dict[str, Any]:
        path = self.correction_dir / "corrected_statistical_comparison.csv"
        calibration_path = self.correction_dir / "corrected_calibration_metrics.csv"
        report_path = self.correction_dir / "corrected_classification_report.csv"
        return {
            "dataset": self._metadata,
            "models": pd.read_csv(path).replace({np.nan: None}).to_dict(orient="records"),
            "calibration": pd.read_csv(calibration_path).replace({np.nan: None}).to_dict(orient="records"),
            "classification_report": pd.read_csv(report_path).replace({np.nan: None}).to_dict(orient="records"),
        }

    def explorer(self, page: int = 1, page_size: int = 20, search: str = "") -> Dict[str, Any]:
        """Return a bounded, read-only view of the existing Jud-IPL test metadata."""
        data = pd.read_parquet(self.data_path)
        if search:
            needle = search.casefold()
            mask = data.astype(str).apply(
                lambda column: column.str.casefold().str.contains(needle, regex=False, na=False)
            ).any(axis=1)
            data = data.loc[mask]
        total = len(data)
        start = (page - 1) * page_size
        rows = data.iloc[start:start + page_size]
        fields = ["id", "label", "case_category", "case_type", "date_of_judgment",
                  "bench", "word_length", "sentence_count", "statute_count", "citation_count"]
        return {
            "dataset": "jud-ipl",
            "page": page,
            "page_size": page_size,
            "total": total,
            "rows": rows[fields].fillna("").to_dict(orient="records"),
            "available_fields": fields,
        }

    def get_document(self, doc_id: str):
        """Return full row metadata for a single judgment by ID."""
        if getattr(self, "_data", None) is None:
            self._data = pd.read_parquet(self.data_path)
        data = self._data
        s_doc = str(doc_id).strip()
        mask = (data["id"].astype(str) == s_doc) | (data["id"].astype(str).str.lstrip("0") == s_doc.lstrip("0"))
        matched = data[mask]
        if matched.empty:
            return None
        row = matched.iloc[0].to_dict()
        return {k: (str(v) if pd.notna(v) else "") for k, v in row.items()}


    @staticmethod
    def _canonical_probabilities(model: Any, texts: List[str]) -> np.ndarray:
        raw = model.predict_proba(texts)
        columns = {label: index for index, label in enumerate(model.classes_)}
        return np.column_stack([raw[:, columns[label]] for label in CANONICAL_CLASSES])

    def predict(self, text: str, model_id: str = "M4") -> Dict[str, Any]:
        if not text or not text.strip():
            raise ValueError("Judgment text cannot be empty.")
        if model_id not in {"M4", "M8"}:
            raise ValueError("model_id must be M4 or M8.")
        self._load_models()
        texts = [text.strip()]
        m4 = self._canonical_probabilities(self._models["M4"], texts)
        if model_id == "M4":
            probabilities = m4
        else:
            component_probs = {
                "m1": self._canonical_probabilities(self._models["M1"], texts),
                "m4": m4,
                "m5": self._canonical_probabilities(self._models["M5"], texts),
                "m3": self._canonical_probabilities(self._models["M3"], texts),
            }
            probabilities = sum(self._weights[key] * component_probs[key] for key in component_probs)
        probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
        row = probabilities[0]
        index = int(np.argmax(row))
        return {
            "dataset": "jud-ipl",
            "model_id": model_id,
            "model_name": next(model["name"] for model in self._metadata["models"] if model["id"] == model_id),
            "predicted_label": CANONICAL_CLASSES[index],
            "probabilities": {label: float(row[i]) for i, label in enumerate(CANONICAL_CLASSES)},
            "word_count": len(text.split()),
        }
