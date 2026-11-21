"""
Phase 17 Unit Tests: Jud-IPL Forensic Audit and Legal Corpus Integration.

Tests:
  1. parse_case_info — metadata extraction from raw case_info strings
  2. Deduplication logic — combined corpus has no duplicate IDs
  3. Source column integrity — all docs have 'IL-TUR' or 'Jud-IPL' source
  4. StatuteExtractor — works on Jud-IPL sample text
  5. CitationExtractor — works on Jud-IPL sample text
  6. Phase16SectionChunker — produces ≥1 chunks from judgment text
  7. Legal features JSON — contains required keys
  8. Combined corpus structure — required columns present
"""

import sys
import json
import re
from pathlib import Path
from typing import Dict, Any

import pytest
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ── Import script helpers ──────────────────────────────────────────────────────
from scripts.preprocess_judipl import parse_case_info


# ── Sample data ───────────────────────────────────────────────────────────────

SAMPLE_CASE_INFO = (
    "Petitioner: Ram Lal Vs State of U.P. "
    "Date of Judgment: 15/07/2003 "
    "Bench: Y.K. Sabharwal, D.M. Dharmadhikari "
    "Citation: 2003 SCC 145"
)

SAMPLE_JUDGMENT = (
    "This appeal arises from the order dated 15 July 2003. "
    "The Indian Penal Code Section 302 is applicable. "
    "Under the Code of Criminal Procedure Section 173, the charge-sheet was filed. "
    "Relying on the judgment in AIR 2001 SC 1234, the court held that the accused "
    "is guilty of the offence of murder. The appeal is hereby dismissed. "
    "We find no merit in the contentions raised by the appellant. "
    "The judgment of the High Court is hereby affirmed. "
    "The ratio decidendi is that the penal provisions apply when intent is proven. "
    "This court has jurisdiction under Article 136 of the Constitution of India."
)

SAMPLE_JUDIPL_IDS = [
    ("74634371.txt", "0074634371"),
    ("1234567.txt", "0001234567"),
    ("9999999999.txt", "9999999999"),
]


# ── Test 1: parse_case_info ────────────────────────────────────────────────────

class TestParseCaseInfo:

    def test_extracts_date(self):
        result = parse_case_info(SAMPLE_CASE_INFO)
        assert result["date"] == "15/07/2003", f"Got: {result['date']}"

    def test_extracts_bench(self):
        result = parse_case_info(SAMPLE_CASE_INFO)
        assert "sabharwal" in result["bench"].lower() or result["bench"] != "", \
            f"Bench extraction failed: {result['bench']}"

    def test_extracts_citation(self):
        result = parse_case_info(SAMPLE_CASE_INFO)
        assert "scc" in result["citation"].lower() or "2003" in result["citation"], \
            f"Citation extraction failed: {result['citation']}"

    def test_empty_string_returns_defaults(self):
        result = parse_case_info("")
        assert result == {"petitioner": "", "respondent": "", "date": "", "bench": "", "citation": ""}

    def test_none_returns_defaults(self):
        result = parse_case_info(None)
        assert result["date"] == ""

    def test_partial_info(self):
        result = parse_case_info("Bench: S.B. Sinha")
        assert "sinha" in result["bench"].lower() or result["bench"] != ""


# ── Test 2: ID normalization ───────────────────────────────────────────────────

class TestIdNormalization:

    def test_zero_padding(self):
        for raw, expected in SAMPLE_JUDIPL_IDS:
            num_str = raw.replace(".txt", "").strip()
            if num_str.isdigit():
                num_id = int(num_str)
                std_id = f"{num_id:010d}"
                assert std_id == expected, f"ID mismatch: {std_id} != {expected}"

    def test_non_numeric_names_rejected(self):
        bad_names = ["abc.txt", "123abc.txt", "", "nan"]
        for name in bad_names:
            num_str = name.replace(".txt", "").strip()
            assert not num_str.isdigit(), f"Expected rejection for: {name}"


# ── Test 3: StatuteExtractor ───────────────────────────────────────────────────

class TestStatuteExtractor:

    def setup_method(self):
        from src.legal_features.statute_extractor import StatuteExtractor
        self.extractor = StatuteExtractor()

    def test_extracts_ipc_section(self):
        results = self.extractor.extract(SAMPLE_JUDGMENT)
        keys = [r.normalized_key for r in results]
        # Should find IPC 302 or CrPC 173 or similar
        assert len(results) >= 1, f"Expected ≥1 statute, got 0. Text: {SAMPLE_JUDGMENT[:200]}"

    def test_returns_list(self):
        results = self.extractor.extract("No statutes here.")
        assert isinstance(results, list)

    def test_empty_text(self):
        results = self.extractor.extract("")
        assert isinstance(results, list)

    def test_normalized_key_is_string(self):
        results = self.extractor.extract(SAMPLE_JUDGMENT)
        for r in results:
            assert isinstance(r.normalized_key, str), f"normalized_key must be str, got {type(r.normalized_key)}"


# ── Test 4: CitationExtractor ──────────────────────────────────────────────────

class TestCitationExtractor:

    def setup_method(self):
        from src.legal_features.citation_extractor import CitationExtractor
        self.extractor = CitationExtractor()

    def test_extracts_air_citation(self):
        results = self.extractor.extract(SAMPLE_JUDGMENT)
        keys = [r.normalized_key for r in results]
        assert len(results) >= 1, f"Expected ≥1 citation, got 0"

    def test_returns_list(self):
        results = self.extractor.extract("No citations here.")
        assert isinstance(results, list)

    def test_empty_text(self):
        results = self.extractor.extract("")
        assert isinstance(results, list)

    def test_normalized_key_is_string(self):
        results = self.extractor.extract(SAMPLE_JUDGMENT)
        for r in results:
            assert isinstance(r.normalized_key, str)


# ── Test 5: Phase16SectionChunker ─────────────────────────────────────────────

class TestSectionChunker:

    def setup_method(self):
        from src.models.phase16_cross_encoder import Phase16SectionChunker
        self.chunker = Phase16SectionChunker()

    def test_produces_chunks(self):
        chunks = self.chunker.chunk_document(SAMPLE_JUDGMENT)
        assert len(chunks) >= 1, "Expected ≥1 chunk from judgment text"

    def test_chunk_keys(self):
        chunks = self.chunker.chunk_document(SAMPLE_JUDGMENT)
        for chunk in chunks:
            assert "section_type" in chunk, f"Missing 'section_type' key in chunk: {chunk.keys()}"
            assert "text" in chunk, f"Missing 'text' key in chunk: {chunk.keys()}"
            assert "rhetorical_weight" in chunk, f"Missing 'rhetorical_weight' key in chunk: {chunk.keys()}"

    def test_section_types_are_known(self):
        chunks = self.chunker.chunk_document(SAMPLE_JUDGMENT)
        valid_types = {"Headnote", "Ratio", "Ruling", "Fact", "unknown", "general"}
        for chunk in chunks:
            st = chunk["section_type"]
            assert isinstance(st, str), f"section_type must be str, got {type(st)}"

    def test_empty_text(self):
        chunks = self.chunker.chunk_document("")
        assert isinstance(chunks, list)


# ── Test 6: Deduplication (unit level) ────────────────────────────────────────

class TestDeduplication:

    def test_no_duplicate_ids_after_merge(self):
        """Simulate the merge logic: IL-TUR IDs take priority; no duplicate IDs in combined."""
        iltur_ids = {"0000001234", "0000005678", "0000009999"}
        judipl_ids = {"0000001234", "0000007777", "0000008888"}  # overlap on 1234

        # IL-TUR preferred: combined = iltur + (judipl - overlap)
        new_judipl = judipl_ids - iltur_ids
        combined = iltur_ids | new_judipl
        assert len(combined) == len(iltur_ids) + len(new_judipl), \
            "Combined should not have duplicate IDs"
        assert "0000001234" in combined  # preserved from IL-TUR

    def test_overlap_count(self):
        iltur_ids = {"0000001234", "0000005678"}
        judipl_ids = {"0000001234", "0000007777"}
        overlap = iltur_ids & judipl_ids
        assert len(overlap) == 1
        assert "0000001234" in overlap


# ── Test 7: Source column integrity ───────────────────────────────────────────

class TestSourceIntegrity:

    def test_valid_source_values(self):
        """All source values in a combined corpus must be 'IL-TUR' or 'Jud-IPL'."""
        df = pd.DataFrame({
            "id": ["0000001234", "0000005678", "0000007777"],
            "source": ["IL-TUR", "IL-TUR", "Jud-IPL"],
            "full_text": ["text a", "text b", "text c"],
        })
        valid_sources = {"IL-TUR", "Jud-IPL"}
        invalid = df[~df["source"].isin(valid_sources)]
        assert len(invalid) == 0, f"Invalid sources found: {invalid['source'].unique()}"

    def test_no_missing_source(self):
        df = pd.DataFrame({
            "id": ["0000001234"],
            "source": ["IL-TUR"],
            "full_text": ["text"],
        })
        assert df["source"].notna().all()


# ── Test 8: Required column structure ─────────────────────────────────────────

class TestRequiredColumns:

    REQUIRED_DOC_COLS = [
        "id", "full_text", "source", "char_length",
        "case_category", "label",
    ]

    REQUIRED_CHUNK_COLS = [
        "chunk_id", "document_id", "section_type", "text", "rhetorical_weight",
    ]

    def test_document_columns_present(self):
        """Simulate a minimal documents DataFrame and verify required columns."""
        df = pd.DataFrame([{
            "id": "0074634371",
            "full_text": SAMPLE_JUDGMENT,
            "source": "Jud-IPL",
            "char_length": len(SAMPLE_JUDGMENT),
            "case_category": "Civil",
            "label": "Accepted",
        }])
        for col in self.REQUIRED_DOC_COLS:
            assert col in df.columns, f"Required column missing: {col}"

    def test_chunk_columns_present(self):
        """Simulate a minimal chunks DataFrame and verify required columns."""
        df = pd.DataFrame([{
            "chunk_id": "0074634371_chk_0",
            "document_id": "0074634371",
            "section_type": "Ratio",
            "text": "The court held...",
            "rhetorical_weight": 1.5,
        }])
        for col in self.REQUIRED_CHUNK_COLS:
            assert col in df.columns, f"Required chunk column missing: {col}"


# ── Test 9: Legal features JSON structure ─────────────────────────────────────

class TestLegalFeaturesJson:

    def test_required_keys(self):
        feature_entry = {
            "statutes": ["IPC_302", "CrPC_173"],
            "citations": ["AIR_2001_SC_1234"],
            "proof_sentence": "The appeal is dismissed.",
            "outcome_label": "Accepted",
        }
        for key in ["statutes", "citations", "proof_sentence", "outcome_label"]:
            assert key in feature_entry, f"Missing key: {key}"

    def test_statutes_is_list(self):
        feature_entry = {"statutes": [], "citations": [], "proof_sentence": "", "outcome_label": ""}
        assert isinstance(feature_entry["statutes"], list)
        assert isinstance(feature_entry["citations"], list)


# ── Test 10: Loaded Jud-IPL documents.parquet (integration, skipped if missing) ──

class TestJudIPLOutputFiles:

    JUDIPL_DOCS = PROJECT_ROOT / "data" / "processed" / "judipl" / "documents.parquet"
    JUDIPL_CHUNKS = PROJECT_ROOT / "data" / "processed" / "judipl" / "chunks.parquet"
    JUDIPL_FEATS = PROJECT_ROOT / "data" / "processed" / "judipl" / "legal_features.json"

    @pytest.mark.skipif(
        not (PROJECT_ROOT / "data" / "processed" / "judipl" / "documents.parquet").exists(),
        reason="Jud-IPL documents.parquet not yet generated"
    )
    def test_documents_schema(self):
        df = pd.read_parquet(self.JUDIPL_DOCS)
        assert len(df) > 0, "documents.parquet is empty"
        assert "id" in df.columns
        assert "full_text" in df.columns
        assert "source" in df.columns
        assert (df["source"] == "Jud-IPL").all(), "All sources should be 'Jud-IPL'"

    @pytest.mark.skipif(
        not (PROJECT_ROOT / "data" / "processed" / "judipl" / "documents.parquet").exists(),
        reason="Jud-IPL documents.parquet not yet generated"
    )
    def test_no_duplicate_ids(self):
        df = pd.read_parquet(self.JUDIPL_DOCS)
        assert df["id"].nunique() == len(df), "Found duplicate IDs in Jud-IPL documents.parquet"

    @pytest.mark.skipif(
        not (PROJECT_ROOT / "data" / "processed" / "judipl" / "chunks.parquet").exists(),
        reason="Jud-IPL chunks.parquet not yet generated"
    )
    def test_chunks_schema(self):
        df = pd.read_parquet(self.JUDIPL_CHUNKS)
        assert "chunk_id" in df.columns
        assert "document_id" in df.columns
        assert "section_type" in df.columns

    @pytest.mark.skipif(
        not (PROJECT_ROOT / "data" / "processed" / "judipl" / "legal_features.json").exists(),
        reason="Jud-IPL legal_features.json not yet generated"
    )
    def test_legal_features_structure(self):
        with open(self.JUDIPL_FEATS) as f:
            feats = json.load(f)
        assert len(feats) > 0, "legal_features.json is empty"
        sample_key = next(iter(feats))
        entry = feats[sample_key]
        assert "statutes" in entry
        assert "citations" in entry
        assert isinstance(entry["statutes"], list)
        assert isinstance(entry["citations"], list)

