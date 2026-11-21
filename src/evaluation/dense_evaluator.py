"""
Dense Evaluator for Legal Prior Case Retrieval.

Orchestrates dense semantic embedding generation, caching, and evaluation
across IL-TUR PCR splits using transformer backbones (InLegalBERT, BERT).
Enforces query self-exclusion, evaluates multiple chunk aggregation strategies,
and produces comparative benchmark reports against Phase 3 lexical baselines.
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd
import torch

# Path setup
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.embeddings.model_loader import ModelLoader
from src.embeddings.embedding_generator import EmbeddingGenerator
from src.retrieval.dense_retriever import DenseRetriever
from src.evaluation.retrieval_metrics import (
    evaluate_query_ranking,
    aggregate_retrieval_metrics,
)


class DenseEvaluator:
    """Manages dense semantic embedding generation and retrieval benchmark evaluation."""

    def __init__(
        self,
        model_name: str = "inlegalbert",
        processed_data_dir: Path = PROJECT_ROOT / "data" / "processed" / "IL-TUR-PCR",
        embeddings_dir: Path = PROJECT_ROOT / "models" / "embeddings",
        results_dir: Path = PROJECT_ROOT / "results",
        batch_size: int = 32,
        device: Optional[str] = None,
    ):
        self.model_name = model_name
        self.model_slug = model_name.lower().replace("/", "_").replace("-", "_")
        self.processed_data_dir = Path(processed_data_dir)
        self.embeddings_dir = Path(embeddings_dir) / self.model_slug
        self.results_dir = Path(results_dir)
        self.eval_dir = self.results_dir / "evaluation"
        self.tables_dir = self.results_dir / "tables"
        self.reports_dir = self.results_dir / "reports"
        self.batch_size = batch_size

        for d in [self.embeddings_dir, self.eval_dir, self.tables_dir, self.reports_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # Determine active device
        active_device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        if active_device == "cpu":
            num_threads = min(8, torch.get_num_threads())
            torch.set_num_threads(num_threads)
            print(f"Configured PyTorch CPU threads: {num_threads}")

        print(f"Initializing DenseEvaluator for model: {self.model_name}...")
        self.model, self.tokenizer, self.device = ModelLoader.load_model_and_tokenizer(
            self.model_name, device=active_device
        )
        self.generator = EmbeddingGenerator(
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
            batch_size=self.batch_size,
        )
        self.metadata = ModelLoader.get_model_metadata(self.model_name)
        print(f"Model loaded on {self.device}. Hidden dim: {self.metadata['hidden_size']}")

    def get_or_generate_fulldoc_embeddings(
        self,
        split_name: str,
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Loads or generates full-document embeddings (head 512 tokens) with masked mean pooling.
        
        Args:
            split_name: e.g. 'dev_queries', 'dev_candidates'
            
        Returns:
            (embeddings_array, docs_df)
        """
        split_dir = self.processed_data_dir / split_name
        save_dir = self.embeddings_dir / split_name
        save_dir.mkdir(parents=True, exist_ok=True)

        npy_path = save_dir / "fulldoc_embeddings.npy"
        meta_path = save_dir / "fulldoc_metadata.parquet"

        docs_file = split_dir / "documents.parquet"
        docs_df = pd.read_parquet(docs_file)

        if npy_path.exists() and meta_path.exists():
            print(f"Loading cached full-doc embeddings for {split_name} from {npy_path}...")
            embeddings = np.load(npy_path)
            if len(embeddings) == len(docs_df):
                return embeddings, docs_df

        print(f"Generating full-doc embeddings for {split_name} ({len(docs_df)} documents)...")
        texts = docs_df["full_text"].values
        embeddings = self.generator.encode_texts(
            texts, show_progress=True, progress_interval=100
        )

        np.save(npy_path, embeddings)

        metadata_df = docs_df[["id"]].copy()
        metadata_df["split"] = split_name
        metadata_df["model"] = self.metadata["hf_id"]
        metadata_df["embedding_dim"] = self.metadata["hidden_size"]
        metadata_df.to_parquet(meta_path, index=False)

        print(f"Successfully saved {split_name} full-doc embeddings ({embeddings.shape}) to {npy_path}")
        return embeddings, docs_df

    def get_or_generate_split_embeddings(
        self,
        split_name: str,
        is_candidates: bool = False,
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Loads cached embeddings or generates them using the transformer backbone.
        
        Args:
            split_name: e.g. 'dev_queries', 'dev_candidates'
            is_candidates: True for candidate pools, False for query sets
            
        Returns:
            (embeddings_array, chunks_or_docs_df)
        """
        split_dir = self.processed_data_dir / split_name
        save_dir = self.embeddings_dir / split_name
        save_dir.mkdir(parents=True, exist_ok=True)

        npy_path = save_dir / "embeddings.npy"
        meta_path = save_dir / "metadata.parquet"

        # Load chunk parquet
        chunks_file = split_dir / "chunks.parquet"
        chunks_df = pd.read_parquet(chunks_file)

        if npy_path.exists() and meta_path.exists():
            print(f"Loading cached embeddings for {split_name} from {npy_path}...")
            embeddings = np.load(npy_path)
            if len(embeddings) == len(chunks_df):
                return embeddings, chunks_df

        print(f"Generating embeddings for {split_name} ({len(chunks_df)} chunks)...")
        embeddings, _ = self.generator.encode_and_save_split(
            split_name=split_name,
            chunks_df=chunks_df,
            output_dir=save_dir,
            model_name=self.metadata["hf_id"],
        )
        return embeddings, chunks_df

    def evaluate_split(
        self,
        split: str = "dev",
        evaluate_chunks: bool = True,
    ) -> Dict[str, Any]:
        """
        Evaluates dense retrieval on a partition ('dev' or 'test').
        
        Args:
            split: 'dev' or 'test'
            evaluate_chunks: Whether to generate and evaluate chunk-level embeddings.
            
        Returns:
            Dict containing benchmark evaluation metrics.
        """
        print(f"\n==========================================")
        print(f"Dense Semantic Evaluation: {split.upper()} [{self.model_name}]")
        print(f"==========================================")

        t_start = time.time()
        q_split = f"{split}_queries"
        c_split = f"{split}_candidates"

        # Load raw metadata
        q_docs = pd.read_parquet(self.processed_data_dir / q_split / "documents.parquet")
        c_docs = pd.read_parquet(self.processed_data_dir / c_split / "documents.parquet")

        cand_doc_ids = c_docs["id"].values.tolist()
        cand_id_set = set(cand_doc_ids)

        # 1. Full-document dense baseline
        t0 = time.time()
        c_doc_embs, _ = self.get_or_generate_fulldoc_embeddings(c_split)
        q_doc_embs, _ = self.get_or_generate_fulldoc_embeddings(q_split)
        t_fulldoc_emb = time.time() - t0

        # Audit valid queries and zero-relevant queries
        valid_queries = []
        zero_relevant_queries = []

        for idx, row in q_docs.iterrows():
            qid = str(row["id"])
            raw_rel = row["relevant_candidates"]
            if raw_rel is None:
                rel_list = []
            elif isinstance(raw_rel, np.ndarray):
                rel_list = [str(x) for x in raw_rel]
            else:
                rel_list = [str(x) for x in list(raw_rel)]

            resolvable_rel = [cid for cid in rel_list if cid in cand_id_set]

            q_info = {
                "idx": idx,
                "id": qid,
                "relevant_candidates": resolvable_rel,
                "resolvable_count": len(resolvable_rel),
            }

            if len(resolvable_rel) == 0:
                zero_relevant_queries.append(q_info)
            else:
                valid_queries.append(q_info)

        print(f"Total Queries: {len(q_docs)}")
        print(f"  Valid Queries: {len(valid_queries)} | Zero-Relevant in Pool: {len(zero_relevant_queries)}")
        print(f"Candidate Documents: {len(cand_doc_ids)} | Dimensions: {self.metadata['hidden_size']}")

        split_results: Dict[str, Any] = {
            "split": split,
            "model_name": self.model_name,
            "hidden_dimension": self.metadata["hidden_size"],
            "total_queries": len(q_docs),
            "valid_queries_count": len(valid_queries),
            "zero_relevant_queries_count": len(zero_relevant_queries),
            "zero_relevant_query_ids": [q["id"] for q in zero_relevant_queries],
            "candidate_pool_size": len(cand_doc_ids),
            "models": {},
        }

        # Evaluate Dense_FullDoc
        print("\nEvaluating Dense_FullDoc baseline...")
        t0 = time.time()
        valid_indices = [q["idx"] for q in valid_queries]
        valid_q_embs = q_doc_embs[valid_indices]

        # Inner product cosine similarity: (n_queries, n_candidates)
        doc_sim_matrix = np.dot(valid_q_embs, c_doc_embs.T)

        from src.retrieval.document_retriever import DocumentRetriever
        doc_retriever = DocumentRetriever(candidate_doc_ids=cand_doc_ids)

        fulldoc_query_metrics = []
        for i, q in enumerate(valid_queries):
            qid = q["id"]
            rel_set = set(q["relevant_candidates"])
            scores = doc_sim_matrix[i].copy()
            ranked = doc_retriever.rank_candidates(scores, query_id=qid, top_n=10)
            retrieved_ids = [doc_id for doc_id, _ in ranked]
            metrics = evaluate_query_ranking(retrieved_ids, rel_set)
            fulldoc_query_metrics.append(metrics)

        agg_fulldoc = aggregate_retrieval_metrics(fulldoc_query_metrics)
        t_eval_fulldoc = time.time() - t0

        split_results["models"]["Dense_FullDoc"] = {
            "strategy": "Dense_FullDoc",
            "scope": "doc",
            "aggregation": "direct",
            "metrics": agg_fulldoc,
            "eval_runtime_seconds": round(t_eval_fulldoc, 3),
        }
        print(
            f"  [Dense_FullDoc         ] P@1: {agg_fulldoc['P@1']:.4f} | "
            f"MRR: {agg_fulldoc['MRR']:.4f} | nDCG@10: {agg_fulldoc['nDCG@10']:.4f} | "
            f"R@10: {agg_fulldoc['R@10']:.4f}"
        )

        # Chunk evaluation if enabled
        if evaluate_chunks:
            # 1. Candidate chunk embeddings
            t0 = time.time()
            cand_embeddings, cand_chunks_df = self.get_or_generate_split_embeddings(
                c_split, is_candidates=True
            )
            t_cand_emb = time.time() - t0

            # 2. Query chunk embeddings
            t0 = time.time()
            query_embeddings, query_chunks_df = self.get_or_generate_split_embeddings(
                q_split, is_candidates=False
            )
            t_query_emb = time.time() - t0

            cand_chunk_doc_ids = cand_chunks_df["doc_id"].values.tolist()

            # Initialize DenseRetriever
            dense_retriever = DenseRetriever(
                candidate_embeddings=cand_embeddings,
                candidate_chunk_doc_ids=cand_chunk_doc_ids,
                candidate_doc_ids=cand_doc_ids,
            )

            # Query document vectors from query chunk embeddings (mean pool)
            unique_qids = q_docs["id"].values.tolist()
            query_doc_embeddings = np.zeros((len(unique_qids), self.metadata["hidden_size"]), dtype=np.float32)
            q_chunk_doc_ids = query_chunks_df["doc_id"].values
            q_doc_id_to_idx = {qid: idx for idx, qid in enumerate(unique_qids)}

            q_chunk_groups: Dict[int, List[int]] = {idx: [] for idx in range(len(unique_qids))}
            for c_idx, qid in enumerate(q_chunk_doc_ids):
                if qid in q_doc_id_to_idx:
                    q_chunk_groups[q_doc_id_to_idx[qid]].append(c_idx)

            for d_idx, c_indices in q_chunk_groups.items():
                if len(c_indices) > 0:
                    mean_vec = query_embeddings[c_indices].mean(axis=0)
                    norm = np.linalg.norm(mean_vec)
                    query_doc_embeddings[d_idx] = mean_vec / max(norm, 1e-12)

            valid_query_vectors = query_doc_embeddings[valid_indices]
            sim_matrix = dense_retriever.score_query_embeddings(valid_query_vectors)

            strategies = [
                ("Dense_Chunk_Max", "max", 1),
                ("Dense_Chunk_Top3Mean", "top_k_mean", 3),
                ("Dense_Chunk_Top5Mean", "top_k_mean", 5),
            ]

            for strat_name, agg_type, k_agg in strategies:
                t_eval_start = time.time()
                query_metrics = []

                for i, q in enumerate(valid_queries):
                    qid = q["id"]
                    rel_set = set(q["relevant_candidates"])
                    chunk_sims = sim_matrix[i]

                    doc_scores = dense_retriever.doc_retriever.aggregate_chunk_scores(
                        chunk_sims, strategy=agg_type, top_k=k_agg
                    )
                    ranked = dense_retriever.doc_retriever.rank_candidates(
                        doc_scores, query_id=qid, top_n=10
                    )
                    retrieved_ids = [doc_id for doc_id, _ in ranked]

                    metrics = evaluate_query_ranking(retrieved_ids, rel_set)
                    query_metrics.append(metrics)

                agg_metrics = aggregate_retrieval_metrics(query_metrics)
                eval_runtime = time.time() - t_eval_start

                split_results["models"][strat_name] = {
                    "strategy": strat_name,
                    "scope": "chunk",
                    "aggregation": agg_type,
                    "top_k": k_agg,
                    "metrics": agg_metrics,
                    "eval_runtime_seconds": round(eval_runtime, 3),
                }

                print(
                    f"  [{strat_name:22s}] P@1: {agg_metrics['P@1']:.4f} | "
                    f"MRR: {agg_metrics['MRR']:.4f} | nDCG@10: {agg_metrics['nDCG@10']:.4f} | "
                    f"R@10: {agg_metrics['R@10']:.4f}"
                )

        split_results["total_split_runtime_seconds"] = round(time.time() - t_start, 2)
        return split_results
