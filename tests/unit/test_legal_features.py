"""
Unit Tests for Legal Knowledge Feature Extraction Pipeline.

Tests:
- StatuteExtractor: IPC, CrPC, CPC, Constitution, and special Acts extraction & normalization
- CitationExtractor: AIR, SCC, SCR, and appeal number extraction
- LegalEntityExtractor: Courts, judges, parties, counsels, organizations, and locations
- RhetoricalAnalyzer: Sentence classification, discourse analysis, and distribution similarity
- LegalFeatureEncoder: Document encoding and pairwise similarity calculation
"""

import unittest
import numpy as np

from src.legal_features.statute_extractor import StatuteExtractor, StatuteReference
from src.legal_features.citation_extractor import CitationExtractor, LegalCitation
from src.legal_features.entity_extractor import LegalEntityExtractor, LegalEntity
from src.legal_features.rhetorical_analyzer import RhetoricalAnalyzer, SentenceRole, RHETORICAL_ROLES
from src.legal_features.legal_feature_encoder import (
    LegalFeatureEncoder,
    DocumentLegalFeatures,
)


class TestStatuteExtractor(unittest.TestCase):
    def setUp(self):
        self.extractor = StatuteExtractor()

    def test_ipc_extraction(self):
        text = "The appellant was convicted under Section 302 read with Section 34 of the Indian Penal Code."
        refs = self.extractor.extract(text)
        keys = self.extractor.get_normalized_keys(refs)
        self.assertIn("IPC:302", keys)
        self.assertIn("IPC:34", keys)

    def test_crpc_and_cpc_extraction(self):
        text = (
            "The petition was filed under Section 482 of the Code of Criminal Procedure. "
            "In civil proceedings, an application under Order VII Rule 11 of the Code of Civil Procedure was dismissed."
        )
        refs = self.extractor.extract(text)
        keys = self.extractor.get_normalized_keys(refs)
        self.assertIn("CRPC:482", keys)
        self.assertIn("CPC:OVII_R11", keys)

    def test_constitution_and_special_acts(self):
        text = (
            "The appellant invoked Article 21 and Article 226 of the Constitution of India. "
            "A complaint under Section 138 of the Negotiable Instruments Act was also initiated."
        )
        refs = self.extractor.extract(text)
        keys = self.extractor.get_normalized_keys(refs)
        self.assertIn("CONST:ART_21", keys)
        self.assertIn("CONST:ART_226", keys)
        self.assertIn("NIACT:138", keys)

    def test_statute_overlap(self):
        text_a = "Convicted under Section 302 IPC and Section 307 IPC."
        text_b = "Charges framed under Section 302 of the Indian Penal Code and Section 34 IPC."
        refs_a = self.extractor.extract(text_a)
        refs_b = self.extractor.extract(text_b)
        overlap = self.extractor.compute_overlap(refs_a, refs_b)
        self.assertGreater(overlap["statute_jaccard"], 0.0)
        self.assertEqual(overlap["statute_shared_count"], 1.0)  # Shared IPC:302


class TestCitationExtractor(unittest.TestCase):
    def setUp(self):
        self.extractor = CitationExtractor()

    def test_scc_and_air_extraction(self):
        text = (
            "Reliance was placed on the decision in (1973) 4 SCC 225 and AIR 1980 SC 1160. "
            "Also see [1962] 2 SCR 100."
        )
        cits = self.extractor.extract(text)
        keys = self.extractor.get_normalized_keys(cits)
        self.assertIn("SCC:1973:4:225", keys)
        self.assertIn("AIR:1980:SC:1160", keys)
        self.assertIn("SCR:1962:2:100", keys)

    def test_appeal_numbers(self):
        text = "This judgment arises out of Criminal Appeal No. 92 of 2015."
        cits = self.extractor.extract(text)
        keys = self.extractor.get_normalized_keys(cits)
        self.assertIn("APPEAL:CRIMINAL_APPEAL:92:2015", keys)

    def test_citation_overlap(self):
        text_a = "Refer to (1973) 4 SCC 225 and AIR 1994 SC 268."
        text_b = "As laid down in (1973) 4 SCC 225 and [1975] 1 SCR 120."
        cits_a = self.extractor.extract(text_a)
        cits_b = self.extractor.extract(text_b)
        overlap = self.extractor.compute_overlap(cits_a, cits_b)
        self.assertGreater(overlap["citation_jaccard"], 0.0)
        self.assertEqual(overlap["citation_shared_count"], 1.0)


class TestLegalEntityExtractor(unittest.TestCase):
    def setUp(self):
        self.extractor = LegalEntityExtractor()

    def test_courts_and_judges(self):
        text = (
            "IN THE SUPREME COURT OF INDIA\n"
            "Hon'ble Mr. Justice D.Y. Chandrachud delivered the judgment. "
            "The High Court of Judicature at Bombay had earlier upheld the conviction."
        )
        entities = self.extractor.extract(text)
        categories = {e.category for e in entities}
        self.assertIn("COURT", categories)
        self.assertIn("JUDGE", categories)

    def test_organizations_and_locations(self):
        text = "The appeal was contested between the State of Maharashtra and Union of India in New Delhi."
        entities = self.extractor.extract(text)
        categories = {e.category for e in entities}
        self.assertIn("ORG", categories)
        self.assertIn("LOC", categories)

    def test_entity_overlap(self):
        text_a = "In the Supreme Court of India, Justice Chandrachud decided the dispute against State of Maharashtra."
        text_b = "Before the Supreme Court of India, State of Maharashtra was represented by counsel."
        ents_a = self.extractor.extract(text_a)
        ents_b = self.extractor.extract(text_b)
        overlap = self.extractor.compute_overlap(ents_a, ents_b)
        self.assertGreater(overlap["entity_jaccard"], 0.0)
        self.assertGreater(overlap["court_overlap"], 0.0)
        self.assertGreater(overlap["org_overlap"], 0.0)


class TestRhetoricalAnalyzer(unittest.TestCase):
    def setUp(self):
        self.analyzer = RhetoricalAnalyzer()

    def test_sentence_classification(self):
        sent_fact = "The case of the prosecution is that on 15th August 2012, the deceased was attacked."
        role_fact, _ = self.analyzer.classify_sentence(sent_fact, position_ratio=0.1)
        self.assertEqual(role_fact, "Fact")

        sent_rpc = "In the result, the appeal is dismissed and the conviction is upheld."
        role_rpc, _ = self.analyzer.classify_sentence(sent_rpc, position_ratio=0.95)
        self.assertEqual(role_rpc, "RulingByPresentCourt")

        sent_stat = "A plain reading of Section 302 of the Indian Penal Code stipulates that whoever commits murder shall be punished."
        role_stat, _ = self.analyzer.classify_sentence(sent_stat, position_ratio=0.4)
        self.assertEqual(role_stat, "Statute")

    def test_document_distribution(self):
        sentences = [
            "The case of the prosecution is that the incident occurred in broad daylight.",
            "The short question for determination is whether the High Court was justified in granting bail.",
            "Learned counsel appearing on behalf of the appellant contended that the evidence was insufficient.",
            "A plain reading of Section 482 of the Code of Criminal Procedure empowers the High Court.",
            "In our considered opinion, the view taken by the High Court is unassailable.",
            "In the result, the appeal is dismissed without any order as to costs.",
        ]
        sentence_roles, dist = self.analyzer.analyze_document(sentences)
        self.assertEqual(len(sentence_roles), len(sentences))
        self.assertAlmostEqual(float(np.sum(dist)), 1.0, places=4)

        # Self-similarity should be 1.0
        sim = self.analyzer.compute_distribution_similarity(dist, dist)
        self.assertAlmostEqual(sim["rhetorical_cosine"], 1.0, places=4)
        self.assertAlmostEqual(sim["rhetorical_js_sim"], 1.0, places=4)


class TestLegalFeatureEncoder(unittest.TestCase):
    def setUp(self):
        self.encoder = LegalFeatureEncoder()

    def test_document_encoding_and_pairwise(self):
        text_q = (
            "REPORTABLE IN THE SUPREME COURT OF INDIA\n"
            "Hon'ble Justice Chandrachud.\n"
            "The appellant was convicted under Section 302 IPC read with Section 34 IPC. "
            "Reliance was placed on (1973) 4 SCC 225. "
            "In our considered opinion, the conviction must be set aside. "
            "In the result, the appeal is allowed."
        )
        text_c = (
            "IN THE SUPREME COURT OF INDIA\n"
            "Hon'ble Justice Chandrachud.\n"
            "The accused faced charges under Section 302 of the Indian Penal Code. "
            "The Court referred to (1973) 4 SCC 225 and AIR 1980 SC 1160. "
            "We hold that the evidence is conclusive. "
            "In the result, the appeal is dismissed."
        )

        doc_q = self.encoder.encode_document("Q1", text_q)
        doc_c = self.encoder.encode_document("C1", text_c)

        self.assertGreater(doc_q.num_statutes, 0)
        self.assertGreater(doc_q.num_citations, 0)
        self.assertGreater(doc_q.num_entities, 0)
        self.assertEqual(len(doc_q.rhetorical_distribution), len(RHETORICAL_ROLES))

        pairwise = self.encoder.compute_pairwise_features(doc_q, doc_c)
        self.assertGreater(pairwise["statute_jaccard"], 0.0)
        self.assertGreater(pairwise["citation_jaccard"], 0.0)
        self.assertGreater(pairwise["entity_jaccard"], 0.0)
        self.assertGreater(pairwise["judge_overlap"], 0.0)
        self.assertGreater(pairwise["rhetorical_cosine"], 0.0)
        self.assertGreater(pairwise["composite_legal_score"], 0.0)


if __name__ == "__main__":
    unittest.main()

