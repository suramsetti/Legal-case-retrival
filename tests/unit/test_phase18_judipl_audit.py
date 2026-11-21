"""
Phase 18 Unit Tests: Jud-IPL Forensic Audit and Task Feasibility.

Verifies:
1. Strict scope compliance: no IL-TUR dependencies.
2. Jud-IPL dataset structure (42,006 rows, 19 columns).
3. Label set integrity (Accepted, Rejected, Other).
4. Proof sentence coverage (100%).
5. Column completeness and data types.
6. Task feasibility verdicts (rejection of ungrounded retrieval, validation of LJP).
7. Leakage vector detection (exact full-text duplicates).
"""

import sys
import json
from pathlib import Path
import pytest
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DOCS_PATH = PROJECT_ROOT / "data" / "processed" / "judipl" / "documents.parquet"
COL_AUDIT_PATH = PROJECT_ROOT / "results" / "tables" / "phase18_judipl_column_audit.csv"
TASK_FEASIBILITY_PATH = PROJECT_ROOT / "results" / "tables" / "phase18_judipl_task_feasibility.csv"
LABEL_DIST_PATH = PROJECT_ROOT / "results" / "tables" / "phase18_judipl_label_distribution.csv"


class TestPhase18DatasetIntegrity:

    @pytest.fixture(autouse=True)
    def setup_data(self):
        assert DOCS_PATH.exists(), f"Jud-IPL documents.parquet not found at {DOCS_PATH}"
        self.df = pd.read_parquet(DOCS_PATH)

    def test_total_row_count(self):
        assert len(self.df) == 42006, f"Expected 42,006 rows, got {len(self.df)}"

    def test_all_unique_ids(self):
        assert self.df["id"].nunique() == 42006, "All 42,006 Kanoon IDs must be unique"

    def test_expected_columns(self):
        expected_cols = [
            "id", "raw_id", "raw_name", "case_category", "case_type",
            "label", "petitioner", "respondent", "date_of_judgment",
            "bench", "citation_meta", "proof_sentence", "char_length",
            "word_length", "sentence_count", "statute_count",
            "citation_count", "full_text", "source"
        ]
        for col in expected_cols:
            assert col in self.df.columns, f"Missing required column: {col}"

    def test_source_identifier(self):
        assert (self.df["source"] == "Jud-IPL").all(), "All records must have source='Jud-IPL'"

    def test_label_values(self):
        valid_labels = {"Accepted", "Rejected", "Other", ""}
        actual_labels = set(self.df["label"].unique())
        assert actual_labels.issubset(valid_labels), f"Unexpected labels found: {actual_labels - valid_labels}"

    def test_proof_sentence_coverage(self):
        non_empty = (self.df["proof_sentence"] != "") & (self.df["proof_sentence"].notnull())
        assert non_empty.sum() >= 42000, f"Expected >=42,000 proof sentences, got {non_empty.sum()}"

    def test_exact_duplicate_text_detection(self):
        # There are 1,028 exact full_text duplicates in the 42,006 cases
        dups = self.df.duplicated(subset=["full_text"]).sum()
        assert dups == 1028, f"Expected exactly 1,028 full text duplicates, found {dups}"


class TestPhase18TaskFeasibility:

    @pytest.fixture(autouse=True)
    def setup_tables(self):
        assert TASK_FEASIBILITY_PATH.exists(), f"Missing {TASK_FEASIBILITY_PATH}"
        self.task_df = pd.read_csv(TASK_FEASIBILITY_PATH)

    def test_retrieval_task_rejected(self):
        """Precedent retrieval must be rejected due to lack of ground-truth relevance pairs."""
        t5_rows = self.task_df[self.task_df["task_id"] == "T5"]
        assert len(t5_rows) == 1, "T5 must be documented in task feasibility table"
        verdict = t5_rows.iloc[0]["feasibility_verdict"]
        assert "NOT DEFENSIBLE" in verdict or "REJECTED" in verdict, \
            f"T5 must be rejected without gold relevance labels, got {verdict}"

    def test_outcome_prediction_accepted(self):
        """Legal outcome prediction (LJP) must be accepted as primary supervised task."""
        t1_rows = self.task_df[self.task_df["task_id"] == "T1"]
        assert len(t1_rows) == 1, "T1 must be documented in task feasibility table"
        verdict = t1_rows.iloc[0]["feasibility_verdict"]
        assert "FEASIBLE" in verdict, f"T1 must be accepted as feasible, got {verdict}"
        assert t1_rows.iloc[0]["available_samples"] >= 40000

    def test_holding_extraction_accepted(self):
        """Legal holding extraction must be accepted as secondary supervised task."""
        t2_rows = self.task_df[self.task_df["task_id"] == "T2"]
        assert len(t2_rows) == 1
        assert "FEASIBLE" in t2_rows.iloc[0]["feasibility_verdict"]


class TestPhase18NoILTURContamination:

    def test_no_iltur_in_audit_files(self):
        for path in [COL_AUDIT_PATH, TASK_FEASIBILITY_PATH, LABEL_DIST_PATH]:
            content = open(path, encoding="utf-8").read()
            assert "IL-TUR-PCR" not in content, f"Contamination in {path}"

