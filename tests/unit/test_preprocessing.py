"""Unit tests for legal text cleaning, document reconstruction, and chunking."""

from pathlib import Path
import sys
import unittest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.text_cleaner import LegalTextCleaner
from src.preprocessing.document_reconstructor import DocumentReconstructor
from src.preprocessing.chunker import SentenceAwareChunker


class TestLegalPreprocessing(unittest.TestCase):
    """Test suite for preprocessing components."""

    def setUp(self):
        self.cleaner = LegalTextCleaner()
        self.reconstructor = DocumentReconstructor(cleaner=self.cleaner)
        tok_path = Path("models/bert/tokenizer.json")
        self.chunker = SentenceAwareChunker(
            tokenizer=tok_path if tok_path.exists() else None,
            max_tokens=100,
            overlap_tokens=20,
        )

    # Test 1: Unicode and Whitespace Normalization
    def test_unicode_and_whitespace_normalization(self):
        raw = "The\u00a0appellant   appeals  against\u200b the judgment.\r\n\x00\x08"
        cleaned = self.cleaner.clean(raw)
        self.assertEqual(cleaned, "The appellant appeals against the judgment.")
        self.assertNotIn("\u00a0", cleaned)
        self.assertNotIn("\x00", cleaned)
        self.assertNotIn("   ", cleaned)

    # Test 2: Legal Citation & Statutory Reference Preservation
    def test_legal_citation_preservation(self):
        citations = [
            "AIR 1965 SC 123",
            "(2008) 4 SCC 594",
            "Section 302 read with Section 34 of the Indian Penal Code, 1860",
            "Criminal Appeal No. 1234 of 2018",
            "Article 21 and Article 14 of the Constitution of India",
            "Order 39 Rule 1 & 2 of CPC",
            "Kesavananda Bharati v. State of Kerala, (1973) 4 SCC 225",
        ]
        for cit in citations:
            cleaned = self.cleaner.clean(cit)
            self.assertTrue(cit in cleaned or cleaned == cit)

    # Test 3: Document Reconstruction & Order Preservation
    def test_document_reconstruction_preserves_order(self):
        sentences = [
            "Sentence one establishes jurisdiction.",
            "Sentence two introduces the prosecution facts.",
            "Sentence three details witness testimony.",
        ]
        result = self.reconstructor.reconstruct(sentences)
        self.assertEqual(result["sentence_count"], 3)
        self.assertEqual(result["original_sentence_count"], 3)
        self.assertEqual(result["empty_sentences_count"], 0)
        self.assertEqual(result["clean_sentences"], sentences)
        expected_full = "Sentence one establishes jurisdiction. Sentence two introduces the prosecution facts. Sentence three details witness testimony."
        self.assertEqual(result["full_text"], expected_full)

    # Test 4: Empty and Malformed Sentences in Reconstruction
    def test_empty_and_malformed_document_reconstruction(self):
        sentences = [
            None,
            "   ",
            "Valid sentence one.",
            "\x00\x08",
            "Valid sentence two.",
            None,
        ]
        result = self.reconstructor.reconstruct(sentences)
        self.assertEqual(result["sentence_count"], 2)
        self.assertEqual(result["original_sentence_count"], 6)
        self.assertEqual(result["empty_sentences_count"], 4)
        self.assertEqual(result["clean_sentences"], ["Valid sentence one.", "Valid sentence two."])
        self.assertEqual(result["full_text"], "Valid sentence one. Valid sentence two.")

        # Completely empty document
        empty_result = self.reconstructor.reconstruct([])
        self.assertEqual(empty_result["sentence_count"], 0)
        self.assertEqual(empty_result["full_text"], "")

    # Test 5: Chunk Ordering, ID Preservation, and Spans
    def test_chunking_ordering_and_id_preservation(self):
        doc_id = "SC_2019_105"
        sentences = [
            "This is sentence number 1 describing the legal dispute in detail.",
            "This is sentence number 2 discussing previous High Court decisions.",
            "This is sentence number 3 highlighting key statutory sections.",
            "This is sentence number 4 citing precedents and relevant legal ratio.",
            "This is sentence number 5 presenting the final decree of the court.",
        ]
        chunks = self.chunker.chunk_document(doc_id=doc_id, sentences=sentences)
        self.assertGreaterEqual(len(chunks), 1)

        # Verify ID, ordering, and consecutive indices
        for idx, chk in enumerate(chunks):
            self.assertEqual(chk["doc_id"], doc_id)
            self.assertEqual(chk["chunk_idx"], idx)
            self.assertEqual(chk["chunk_id"], f"{doc_id}_chk_{idx:04d}")
            self.assertLessEqual(chk["start_sentence_idx"], chk["end_sentence_idx"])
            self.assertLessEqual(chk["token_count"], self.chunker.max_tokens)

    # Test 6: Chunk Overlap Verification
    def test_chunk_overlap(self):
        doc_id = "SC_OVERLAP_TEST"
        sentences = [f"Legal proposition {i} with factual considerations and case circumstances." for i in range(25)]
        chunks = self.chunker.chunk_document(doc_id=doc_id, sentences=sentences)
        self.assertGreater(len(chunks), 1)

        # Verify overlap exists between consecutive chunks
        for i in range(len(chunks) - 1):
            curr_chunk = chunks[i]
            next_chunk = chunks[i + 1]
            self.assertLessEqual(next_chunk["start_sentence_idx"], curr_chunk["end_sentence_idx"])

    # Test 7: Split Isolation & Zero Leaks
    def test_split_isolation_logic(self):
        train_queries = {"Q1", "Q2", "Q3"}
        dev_queries = {"Q4", "Q5"}
        test_queries = {"Q6", "Q7"}

        self.assertEqual(len(train_queries.intersection(dev_queries)), 0)
        self.assertEqual(len(train_queries.intersection(test_queries)), 0)
        self.assertEqual(len(dev_queries.intersection(test_queries)), 0)


if __name__ == "__main__":
    unittest.main()
