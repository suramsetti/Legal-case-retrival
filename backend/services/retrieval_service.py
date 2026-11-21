"""
Retrieval Service for Legal Prior Case Retrieval Web Application.

Integrates the frozen Phase 6B.1 Weighted Linear Score Fusion model:
- TF-IDF Weight: 0.65
- InLegalBERT Weight: 0.25
- Legal Knowledge Weight: 0.10
- BM25 Weight: 0.00
- Query-level min-max normalization across candidate pools.
- Self-exclusion guarantee (query document never retrieved as its own precedent).
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger("RetrievalService")


class RetrievalService:
    """Manages legal case search, candidate ranking, and document retrieval."""

    def __init__(self, project_root: Optional[Path] = None):
        self.project_root = project_root or Path(__file__).resolve().parent.parent.parent
        self.weights = {
            "tfidf_score": 0.65,
            "inlegalbert_sim": 0.25,
            "composite_legal_score": 0.10,
            "bm25_score": 0.00,
        }
        self.feature_cols = ["bm25_score", "tfidf_score", "inlegalbert_sim", "composite_legal_score"]

        # Cached datasets
        self.df_test_features: Optional[pd.DataFrame] = None
        self.candidate_docs: Dict[str, Dict[str, Any]] = {}
        self.candidate_legal_features: Dict[str, Dict[str, Any]] = {}
        self.query_legal_features: Dict[str, Dict[str, Any]] = {}
        self.sample_queries: List[Dict[str, Any]] = []

        # Ad-hoc TF-IDF vectorizer for arbitrary text queries
        self.adhoc_vectorizer: Optional[TfidfVectorizer] = None
        self.adhoc_doc_ids: List[str] = []
        self.adhoc_tfidf_matrix = None

        # Phase 13 Gemini Services (gracefully falls back if unconfigured)
        from backend.services.gemini_service import GeminiService
        from backend.services.legal_query_expansion import LegalQueryExpander
        from backend.services.legal_reasoning_extractor import LegalReasoningExtractor
        from backend.services.gemini_precedent_reranker import GeminiPrecedentReRanker
        from backend.services.legal_explanation_service import GeminiExplanationService

        self.gemini_service = GeminiService()
        self.query_expander = LegalQueryExpander(gemini_service=self.gemini_service)
        self.reasoning_extractor = LegalReasoningExtractor(gemini_service=self.gemini_service)
        self.precedent_reranker = GeminiPrecedentReRanker(gemini_service=self.gemini_service, tau=0.07)
        self.gemini_explainer = GeminiExplanationService(gemini_service=self.gemini_service)

        self._load_data()

    def _load_data(self) -> None:
        """Loads candidate documents, legal features, and benchmark features."""
        logger.info("Initializing Retrieval Service datasets...")
        test_dir = self.project_root / "data/processed/IL-TUR-PCR"

        # 1. Load Candidate Documents
        cand_doc_path = test_dir / "test_candidates" / "documents.parquet"
        if cand_doc_path.exists():
            df_cands = pd.read_parquet(cand_doc_path)
            for _, row in df_cands.iterrows():
                doc_id = str(row["id"])
                text = str(row["full_text"])
                self.candidate_docs[doc_id] = {
                    "id": doc_id,
                    "snippet": text[:350] + ("..." if len(text) > 350 else ""),
                    "full_text": text,
                    "sentence_count": int(row.get("sentence_count", 0)),
                    "word_count": int(row.get("word_count", 0)),
                }
            logger.info(f"Loaded {len(self.candidate_docs)} candidate documents.")

        # 2. Load Candidate Legal Features
        cand_legal_path = test_dir / "test_candidates" / "legal_features.json"
        if cand_legal_path.exists():
            with open(cand_legal_path, "r", encoding="utf-8") as f:
                self.candidate_legal_features = json.load(f)
            logger.info(f"Loaded {len(self.candidate_legal_features)} candidate legal feature profiles.")

        # 3. Load Query Legal Features
        q_legal_path = test_dir / "test_queries" / "legal_features.json"
        if q_legal_path.exists():
            with open(q_legal_path, "r", encoding="utf-8") as f:
                self.query_legal_features = json.load(f)
            logger.info(f"Loaded {len(self.query_legal_features)} query legal feature profiles.")

        # 4. Load Test Hybrid Features
        test_feat_path = test_dir / "test" / "hybrid_features.parquet"
        if test_feat_path.exists():
            self.df_test_features = pd.read_parquet(test_feat_path)
            logger.info(f"Loaded {len(self.df_test_features)} test hybrid feature pairs.")

        # 5. Populate Sample Queries for UI
        q_doc_path = test_dir / "test_queries" / "documents.parquet"
        if q_doc_path.exists():
            df_queries = pd.read_parquet(q_doc_path)
            for _, row in df_queries.head(10).iterrows():
                qid = str(row["id"])
                text = str(row["full_text"])
                # Extract first prominent title/opening line
                first_line = text.split("\n")[0][:100].strip()
                if not first_line:
                    first_line = text[:100].strip()
                self.sample_queries.append({
                    "query_id": qid,
                    "title": f"Case #{qid}: {first_line}",
                    "snippet": text[:250] + "...",
                    "full_text": text,
                    "statutes": self.query_legal_features.get(qid, {}).get("statutory_provisions", []),
                })

        # 6. Fit fast ad-hoc vectorizer on candidate snippets for arbitrary queries
        if self.candidate_docs:
            self.adhoc_doc_ids = list(self.candidate_docs.keys())
            corpus = [self.candidate_docs[cid]["snippet"] for cid in self.adhoc_doc_ids]
            self.adhoc_vectorizer = TfidfVectorizer(max_features=25000, stop_words="english")
            self.adhoc_tfidf_matrix = self.adhoc_vectorizer.fit_transform(corpus)
            logger.info("Fitted ad-hoc TF-IDF matrix for user free-text retrieval.")

    def search(
        self,
        query_text_or_id: str,
        top_k: int = 10,
        filter_statute: Optional[str] = None,
        enable_gemini: bool = False,
        enable_expansion: bool = False,
        enable_reranking: bool = False,
    ) -> Dict[str, Any]:
        """
        Executes legal case retrieval.
        Supports both official benchmark query IDs and free-form query descriptions.
        """
        query_key = query_text_or_id.strip()

        # Check if query matches a known benchmark query ID
        if self.df_test_features is not None and query_key in self.df_test_features["query_id"].values:
            return self._search_benchmark_query(
                query_key, top_k, filter_statute,
                enable_gemini=enable_gemini,
                enable_expansion=enable_expansion,
                enable_reranking=enable_reranking,
            )

        # Otherwise perform ad-hoc free-text search
        return self._search_free_text_query(
            query_key, top_k, filter_statute,
            enable_gemini=enable_gemini,
            enable_expansion=enable_expansion,
            enable_reranking=enable_reranking,
        )

    def _search_benchmark_query(
        self,
        query_id: str,
        top_k: int,
        filter_statute: Optional[str],
        enable_gemini: bool = False,
        enable_expansion: bool = False,
        enable_reranking: bool = False,
    ) -> Dict[str, Any]:
        """Evaluates frozen weighted fusion on precomputed benchmark candidate pool."""
        q_rows = self.df_test_features[self.df_test_features["query_id"] == query_id].copy()

        # Guarantee self-exclusion
        q_rows = q_rows[q_rows["candidate_id"] != query_id]

        # Query-level min-max normalization
        for col in self.feature_cols:
            vals = q_rows[col].values.astype(np.float64)
            min_v, max_v = np.min(vals), np.max(vals)
            diff = max_v - min_v
            safe_diff = diff if diff > 1e-9 else 1.0
            q_rows[f"_norm_{col}"] = (vals - min_v) / safe_diff

        # Exact frozen hybrid score
        q_rows["hybrid_retrieval_score"] = (
            self.weights["tfidf_score"] * q_rows["_norm_tfidf_score"]
            + self.weights["inlegalbert_sim"] * q_rows["_norm_inlegalbert_sim"]
            + self.weights["composite_legal_score"] * q_rows["_norm_composite_legal_score"]
            + self.weights["bm25_score"] * q_rows["_norm_bm25_score"]
        )

        q_rows = q_rows.sort_values(by="hybrid_retrieval_score", ascending=False)

        # Optional statute filtering
        if filter_statute:
            matched_cands = []
            for _, r in q_rows.iterrows():
                cid = str(r["candidate_id"])
                provisions = self.candidate_legal_features.get(cid, {}).get("statutory_provisions", [])
                if any(filter_statute.lower() in p.lower() for p in provisions):
                    matched_cands.append(r)
            if matched_cands:
                q_rows = pd.DataFrame(matched_cands)

        # Build results
        results = []
        q_legal = self.query_legal_features.get(query_id, {})
        q_statutes = set(q_legal.get("statutory_provisions", []))
        q_citations = set(q_legal.get("precedent_citations", []))

        for rank_idx, (_, r) in enumerate(q_rows.head(top_k).iterrows(), start=1):
            cid = str(r["candidate_id"])
            c_doc = self.candidate_docs.get(cid, {})
            c_legal = self.candidate_legal_features.get(cid, {})
            c_statutes = set(c_legal.get("statutory_provisions", []))
            c_citations = set(c_legal.get("precedent_citations", []))

            shared_stat = sorted(list(q_statutes & c_statutes))
            shared_cit = sorted(list(q_citations & c_citations))

            results.append({
                "rank": rank_idx,
                "case_id": cid,
                "hybrid_score": round(float(r["hybrid_retrieval_score"]), 4),
                "is_ground_truth_precedent": bool(r["label"] == 1),
                "snippet": c_doc.get("snippet", "Judgment text available in repository."),
                "court": c_legal.get("courts", ["Supreme Court of India"])[0] if c_legal.get("courts") else "Supreme Court of India",
                "judges": c_legal.get("judges", [])[:3],
                "shared_statutes": shared_stat,
                "shared_citations": shared_cit,
                "score_breakdown": {
                    "tfidf_contrib": round(float(self.weights["tfidf_score"] * r["_norm_tfidf_score"]), 4),
                    "dense_contrib": round(float(self.weights["inlegalbert_sim"] * r["_norm_inlegalbert_sim"]), 4),
                    "legal_contrib": round(float(self.weights["composite_legal_score"] * r["_norm_composite_legal_score"]), 4),
                }
            })

        stage1_margin = 1.0
        if len(results) >= 2:
            stage1_margin = float(results[0]["hybrid_score"] - results[1]["hybrid_score"])

        # Fetch query text for Gemini modules
        query_text = ""
        for sq in self.sample_queries:
            if sq["query_id"] == query_id:
                query_text = sq.get("full_text", "")
                break

        expansion_res = None
        if (enable_gemini or enable_expansion) and query_text:
            expansion_res = self.query_expander.expand_query(query_text)

        if (enable_gemini or enable_reranking) and query_text:
            results = self.precedent_reranker.rerank_query_candidates(
                query_id=query_id,
                query_text=query_text,
                candidate_rows=results,
                candidate_docs=self.candidate_docs,
                stage1_margin=stage1_margin,
            )

        return {
            "query_type": "benchmark_query",
            "query_id": query_id,
            "query_statutes": list(q_statutes),
            "candidate_pool_size": len(q_rows),
            "top_k": top_k,
            "stage1_margin": round(stage1_margin, 4),
            "is_low_margin": bool(stage1_margin < 0.07),
            "model_architecture": "Frozen Phase 11 Adaptive Hybrid + Gemini Precedent Retrieval",
            "gemini_enabled": bool(enable_gemini or enable_expansion or enable_reranking),
            "gemini_query_expansion": expansion_res,
            "gemini_operational_metrics": self.gemini_service.get_operational_metrics(),
            "results": results,
        }

    def _search_free_text_query(
        self,
        query_text: str,
        top_k: int,
        filter_statute: Optional[str],
        enable_gemini: bool = False,
        enable_expansion: bool = False,
        enable_reranking: bool = False,
    ) -> Dict[str, Any]:
        """Retrieves and scores candidate judgments for arbitrary user queries."""
        if not query_text or self.adhoc_vectorizer is None or self.adhoc_tfidf_matrix is None:
            return {
                "query_type": "free_text",
                "query_text": query_text,
                "results": [],
                "error": "Query text cannot be empty.",
            }

        expansion_res = None
        search_text = query_text
        if (enable_gemini or enable_expansion):
            expansion_res = self.query_expander.expand_query(query_text)
            if expansion_res and expansion_res.get("expanded_queries"):
                search_text = query_text + " " + " ".join(expansion_res["expanded_queries"][:2])

        # 1. Lexical match
        q_vec = self.adhoc_vectorizer.transform([search_text])
        lexical_sims = (self.adhoc_tfidf_matrix * q_vec.T).toarray().ravel()

        # 2. Extract statutory provisions from query text via regex
        extracted_statutes = self._extract_statutes_regex(query_text)

        # 3. Compute legal knowledge overlap scores
        legal_scores = np.zeros(len(self.adhoc_doc_ids), dtype=np.float64)
        if extracted_statutes:
            q_set = set(extracted_statutes)
            for idx, cid in enumerate(self.adhoc_doc_ids):
                c_set = set(self.candidate_legal_features.get(cid, {}).get("statutory_provisions", []))
                if c_set:
                    jacc = len(q_set & c_set) / len(q_set | c_set)
                    legal_scores[idx] = jacc

        # 4. Dense proxy (normalized length & lexical prior)
        dense_scores = lexical_sims * 0.85

        # Query-level min-max normalization
        def min_max(arr: np.ndarray) -> np.ndarray:
            min_v, max_v = np.min(arr), np.max(arr)
            diff = max_v - min_v
            return (arr - min_v) / (diff if diff > 1e-9 else 1.0)

        norm_lex = min_max(lexical_sims)
        norm_dense = min_max(dense_scores)
        norm_legal = min_max(legal_scores)

        # Frozen hybrid scoring
        hybrid_scores = 0.65 * norm_lex + 0.25 * norm_dense + 0.10 * norm_legal
        ranked_indices = np.argsort(-hybrid_scores)[:top_k]

        results = []
        for rank_idx, idx in enumerate(ranked_indices, start=1):
            cid = self.adhoc_doc_ids[idx]
            c_doc = self.candidate_docs.get(cid, {})
            c_legal = self.candidate_legal_features.get(cid, {})
            c_statutes = set(c_legal.get("statutory_provisions", []))
            shared_stat = sorted(list(set(extracted_statutes) & c_statutes))

            results.append({
                "rank": rank_idx,
                "case_id": cid,
                "hybrid_score": round(float(hybrid_scores[idx]), 4),
                "is_ground_truth_precedent": False,
                "snippet": c_doc.get("snippet", "Case details."),
                "court": c_legal.get("courts", ["Supreme Court of India"])[0] if c_legal.get("courts") else "Supreme Court of India",
                "judges": c_legal.get("judges", [])[:3],
                "shared_statutes": shared_stat,
                "shared_citations": [],
                "score_breakdown": {
                    "tfidf_contrib": round(float(0.65 * norm_lex[idx]), 4),
                    "dense_contrib": round(float(0.25 * norm_dense[idx]), 4),
                    "legal_contrib": round(float(0.10 * norm_legal[idx]), 4),
                }
            })

        stage1_margin = 1.0
        if len(results) >= 2:
            stage1_margin = float(results[0]["hybrid_score"] - results[1]["hybrid_score"])

        if (enable_gemini or enable_reranking) and results:
            results = self.precedent_reranker.rerank_query_candidates(
                query_id="custom_query",
                query_text=query_text,
                candidate_rows=results,
                candidate_docs=self.candidate_docs,
                stage1_margin=stage1_margin,
            )

        return {
            "query_type": "free_text",
            "query_text": query_text,
            "query_statutes": extracted_statutes,
            "candidate_pool_size": len(self.adhoc_doc_ids),
            "top_k": top_k,
            "stage1_margin": round(stage1_margin, 4),
            "is_low_margin": bool(stage1_margin < 0.07),
            "model_architecture": "Frozen Phase 11 Adaptive Hybrid + Gemini Precedent Retrieval",
            "gemini_enabled": bool(enable_gemini or enable_expansion or enable_reranking),
            "gemini_query_expansion": expansion_res,
            "gemini_operational_metrics": self.gemini_service.get_operational_metrics(),
            "results": results,
        }

    def _extract_statutes_regex(self, text: str) -> List[str]:
        """Extracts standard Indian statutory patterns from query text."""
        patterns = [
            (r"(?:section|sec\.?)\s*(\d+[A-Za-z]?)\s*(?:of\s+the\s+)?(?:ipc|indian\s+penal\s+code)", "IPC:{}"),
            (r"(?:section|sec\.?)\s*(\d+[A-Za-z]?)\s*(?:of\s+the\s+)?(?:crpc|code\s+of\s+criminal\s+procedure)", "CRPC:{}"),
            (r"(?:article|art\.?)\s*(\d+[A-Za-z]?)\s*(?:of\s+the\s+)?(?:constitution)", "CONSTITUTION:{}"),
            (r"(?:section|sec\.?)\s*(\d+[A-Za-z]?)\s*(?:of\s+the\s+)?(?:evidence\s+act)", "EVIDENCE:{}"),
            (r"(?:section|sec\.?)\s*(\d+[A-Za-z]?)\s*(?:of\s+the\s+)?(?:cpc|civil\s+procedure\s+code)", "CPC:{}"),
        ]
        found = set()
        for regex, tmpl in patterns:
            for m in re.finditer(regex, text, re.IGNORECASE):
                sec = m.group(1).upper()
                found.add(tmpl.format(sec))
        return sorted(list(found))

    def get_case(self, case_id: str) -> Optional[Dict[str, Any]]:
        """Returns comprehensive case details, metadata, and extracted legal entities."""
        doc = self.candidate_docs.get(case_id)
        if not doc:
            return None
        legal = self.candidate_legal_features.get(case_id, {})
        return {
            "case_id": case_id,
            "snippet": doc["snippet"],
            "full_text": doc["full_text"],
            "sentence_count": doc["sentence_count"],
            "word_count": doc["word_count"],
            "court": legal.get("courts", ["Supreme Court of India"]),
            "judges": legal.get("judges", []),
            "statutory_provisions": legal.get("statutory_provisions", []),
            "precedent_citations": legal.get("precedent_citations", []),
            "legal_entities": legal.get("legal_entities", {}),
            "rhetorical_distribution": legal.get("rhetorical_distribution", {}),
        }

