"""End-to-end preprocessing and chunking pipeline for IL-TUR PCR dataset.

Processes all six dataset splits, reconstructs and normalizes legal judgments,
generates sentence-aware transformer chunks, calculates comprehensive statistics,
and stores clean Parquet outputs and analytical reports.
"""

import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

mpl_dir = PROJECT_ROOT / "logs" / "mpl"
mpl_dir.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(mpl_dir.resolve())

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.preprocessing.chunker import SentenceAwareChunker
from src.preprocessing.document_reconstructor import DocumentReconstructor
from src.preprocessing.text_cleaner import LegalTextCleaner


class PreprocessingPipeline:
    """Manages the full preprocessing, reconstruction, chunking, and statistics lifecycle."""

    def __init__(
        self,
        raw_data_dir: Path = Path("data/external/IL-TUR-PCR"),
        processed_data_dir: Path = Path("data/processed/IL-TUR-PCR"),
        results_tables_dir: Path = Path("results/tables"),
        results_figures_dir: Path = Path("results/figures"),
        tokenizer_path: Optional[Path] = Path("models/bert/tokenizer.json"),
        max_tokens: int = 512,
        overlap_tokens: int = 64,
    ):
        self.raw_data_dir = Path(raw_data_dir)
        self.processed_data_dir = Path(processed_data_dir)
        self.results_tables_dir = Path(results_tables_dir)
        self.results_figures_dir = Path(results_figures_dir)

        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        self.results_tables_dir.mkdir(parents=True, exist_ok=True)
        self.results_figures_dir.mkdir(parents=True, exist_ok=True)

        self.cleaner = LegalTextCleaner()
        self.reconstructor = DocumentReconstructor(cleaner=self.cleaner)
        self.chunker = SentenceAwareChunker(
            tokenizer=tokenizer_path if tokenizer_path and tokenizer_path.exists() else None,
            max_tokens=max_tokens,
            overlap_tokens=overlap_tokens,
        )

        self.splits = [
            "train_queries",
            "train_candidates",
            "dev_queries",
            "dev_candidates",
            "test_queries",
            "test_candidates",
        ]

        self.partition_pairs = {
            "train_queries": "train_candidates",
            "dev_queries": "dev_candidates",
            "test_queries": "test_candidates",
        }

    def run(self) -> Dict[str, Any]:
        """Executes the complete preprocessing pipeline across all splits."""
        print("Starting IL-TUR PCR Preprocessing and Chunking Pipeline...")

        # Step 1: Pre-collect candidate IDs for pool-membership checks
        candidate_pools = {}
        for c_split in ["train_candidates", "dev_candidates", "test_candidates"]:
            raw_c_file = self.raw_data_dir / c_split / f"{c_split}.parquet"
            table = pq.read_table(raw_c_file, columns=["id"])
            candidate_pools[c_split] = set(table.to_pandas()["id"].astype(str).tolist())
            print(f"Loaded candidate pool [{c_split}]: {len(candidate_pools[c_split])} IDs")

        pipeline_stats: Dict[str, Any] = {}
        all_token_counts: List[int] = []
        all_chunk_counts: List[int] = []

        # Step 2: Process each split
        for split in self.splits:
            print(f"\nProcessing split: {split}...")
            raw_file = self.raw_data_dir / split / f"{split}.parquet"
            if not raw_file.exists():
                raise FileNotFoundError(f"Raw data file missing: {raw_file}")

            table = pq.read_table(raw_file)
            df = table.to_pandas()

            doc_rows: List[Dict[str, Any]] = []
            chunk_rows: List[Dict[str, Any]] = []

            is_query = "queries" in split
            paired_c_split = self.partition_pairs.get(split)
            target_pool = candidate_pools.get(paired_c_split, set()) if is_query else set()

            raw_word_counts = []
            clean_word_counts = []
            sentence_counts = []
            char_counts = []
            chunks_per_doc = []

            total_docs = len(df)
            print(f"Starting {split}: {total_docs} documents...", flush=True)

            for i, (_, row) in enumerate(df.iterrows()):
                if (i + 1) % 250 == 0 or (i + 1) == total_docs:
                    print(f"  [{split}] Processed {i + 1}/{total_docs} docs ({len(chunk_rows)} chunks accumulated)...", flush=True)

                doc_id = str(row["id"])
                raw_sentences = row["text"]

                # Reconstruct & normalize
                reconstructed = self.reconstructor.reconstruct(raw_sentences)
                full_text = reconstructed["full_text"]
                clean_sents = reconstructed["clean_sentences"]

                raw_sents_list = raw_sentences.tolist() if isinstance(raw_sentences, np.ndarray) else list(raw_sentences or [])
                raw_words = sum(len(str(s).split()) for s in raw_sents_list)
                clean_words = reconstructed["word_count"]

                raw_word_counts.append(raw_words)
                clean_word_counts.append(clean_words)
                sentence_counts.append(reconstructed["sentence_count"])
                char_counts.append(reconstructed["character_count"])

                # Handle relevant_candidates for queries
                rels = row.get("relevant_candidates")
                if rels is not None:
                    if isinstance(rels, np.ndarray):
                        rels_list = [str(x) for x in rels.tolist()]
                    elif isinstance(rels, (list, tuple)):
                        rels_list = [str(x) for x in rels]
                    else:
                        rels_list = []
                else:
                    rels_list = []

                # Pool check & zero-relevant flag
                in_pool = [cid for cid in rels_list if cid in target_pool] if is_query else []
                has_zero_in_pool = (len(in_pool) == 0) if is_query else False

                doc_rows.append(
                    {
                        "id": doc_id,
                        "full_text": full_text,
                        "clean_sentences": clean_sents,
                        "sentence_count": reconstructed["sentence_count"],
                        "word_count": reconstructed["word_count"],
                        "character_count": reconstructed["character_count"],
                        "relevant_candidates": rels_list,
                        "has_zero_relevant_in_pool": has_zero_in_pool,
                        "resolvable_candidates_count": len(in_pool) if is_query else 0,
                    }
                )

                # Chunking
                doc_chunks = self.chunker.chunk_document(doc_id=doc_id, sentences=clean_sents)
                chunks_per_doc.append(len(doc_chunks))
                chunk_rows.extend(doc_chunks)

                for chk in doc_chunks:
                    all_token_counts.append(chk["token_count"])

            all_chunk_counts.extend(chunks_per_doc)

            # Convert to DataFrames
            doc_df = pd.DataFrame(doc_rows)
            chunk_df = pd.DataFrame(chunk_rows)

            # Save processed Parquet files
            split_dir = self.processed_data_dir / split
            split_dir.mkdir(parents=True, exist_ok=True)

            doc_out_path = split_dir / "documents.parquet"
            chunk_out_path = split_dir / "chunks.parquet"

            doc_df.to_parquet(doc_out_path, index=False)
            chunk_df.to_parquet(chunk_out_path, index=False)

            # Record split statistics
            raw_words_arr = np.array(raw_word_counts, dtype=np.int64)
            clean_words_arr = np.array(clean_word_counts, dtype=np.int64)
            sents_arr = np.array(sentence_counts, dtype=np.int64)
            chars_arr = np.array(char_counts, dtype=np.int64)
            chunks_arr = np.array(chunks_per_doc, dtype=np.int64)

            split_stat = {
                "document_count": int(len(doc_df)),
                "total_chunks": int(len(chunk_df)),
                "avg_chunks_per_doc": round(float(np.mean(chunks_arr)), 2) if len(chunks_arr) > 0 else 0.0,
                "median_chunks_per_doc": float(np.median(chunks_arr)) if len(chunks_arr) > 0 else 0.0,
                "max_chunks_per_doc": int(np.max(chunks_arr)) if len(chunks_arr) > 0 else 0,
                "avg_clean_word_count": round(float(np.mean(clean_words_arr)), 2) if len(clean_words_arr) > 0 else 0.0,
                "median_clean_word_count": float(np.median(clean_words_arr)) if len(clean_words_arr) > 0 else 0.0,
                "max_clean_word_count": int(np.max(clean_words_arr)) if len(clean_words_arr) > 0 else 0,
                "avg_sentence_count": round(float(np.mean(sents_arr)), 2) if len(sents_arr) > 0 else 0.0,
                "empty_documents_count": int(np.sum(clean_words_arr == 0)),
                "extremely_long_docs_gt_10k_words": int(np.sum(clean_words_arr > 10000)),
                "extremely_long_docs_gt_20k_words": int(np.sum(clean_words_arr > 20000)),
                "net_word_count_change": int(np.sum(clean_words_arr) - np.sum(raw_words_arr)),
            }

            if is_query:
                split_stat["queries_with_zero_relevant_in_pool"] = int(doc_df["has_zero_relevant_in_pool"].sum())

            pipeline_stats[split] = split_stat
            print(f"Finished {split}: {len(doc_df)} documents -> {len(chunk_df)} chunks ({split_stat['avg_chunks_per_doc']} chunks/doc)")

        # Step 3: Compute global chunking and configuration report
        token_arr = np.array(all_token_counts, dtype=np.int64)
        chunk_dist_stat = {
            "total_chunks_across_dataset": int(len(token_arr)),
            "max_tokens_config": self.chunker.max_tokens,
            "overlap_tokens_config": self.chunker.overlap_tokens,
            "avg_chunk_tokens": round(float(np.mean(token_arr)), 2) if len(token_arr) > 0 else 0.0,
            "median_chunk_tokens": float(np.median(token_arr)) if len(token_arr) > 0 else 0.0,
            "min_chunk_tokens": int(np.min(token_arr)) if len(token_arr) > 0 else 0,
            "max_chunk_tokens": int(np.max(token_arr)) if len(token_arr) > 0 else 0,
            "p25_tokens": float(np.percentile(token_arr, 25)) if len(token_arr) > 0 else 0.0,
            "p75_tokens": float(np.percentile(token_arr, 75)) if len(token_arr) > 0 else 0.0,
            "p95_tokens": float(np.percentile(token_arr, 95)) if len(token_arr) > 0 else 0.0,
        }

        full_summary = {
            "pipeline_name": "IL-TUR PCR Preprocessing and Chunking Pipeline",
            "chunker_configuration": {
                "max_tokens": self.chunker.max_tokens,
                "overlap_tokens": self.chunker.overlap_tokens,
                "special_tokens_reserve": self.chunker.special_tokens_reserve,
                "tokenizer": "bert-base-uncased (via models/bert/tokenizer.json)",
                "sentence_boundary_preservation": True,
            },
            "split_statistics": pipeline_stats,
            "chunk_token_distribution": chunk_dist_stat,
        }

        # Step 4: Save JSON summary table
        summary_table_path = self.results_tables_dir / "preprocessing_summary.json"
        with open(summary_table_path, "w", encoding="utf-8") as f:
            json.dump(full_summary, f, indent=2)
        print(f"\nSaved preprocessing summary table to {summary_table_path}")

        # Step 5: Generate distribution figures
        self._generate_figures(token_arr, all_chunk_counts)

        return full_summary

    def _generate_figures(self, token_counts: np.ndarray, chunks_per_doc: List[int]) -> None:
        """Generates distribution plots for token lengths and chunk counts."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Plot 1: Chunk Token Length Distribution
        axes[0].hist(token_counts, bins=40, color="#1f77b4", edgecolor="black", alpha=0.7)
        axes[0].axvline(np.mean(token_counts), color="red", linestyle="--", label=f"Mean: {np.mean(token_counts):.1f}")
        axes[0].axvline(np.median(token_counts), color="green", linestyle=":", label=f"Median: {np.median(token_counts):.1f}")
        axes[0].set_title("Chunk Token Length Distribution (BERT Tokenizer)")
        axes[0].set_xlabel("Tokens per Chunk")
        axes[0].set_ylabel("Frequency")
        axes[0].legend()
        axes[0].grid(True, linestyle="--", alpha=0.5)

        # Plot 2: Chunks per Document Distribution (Capped at 99th percentile for clean display)
        chk_arr = np.array(chunks_per_doc, dtype=np.int64)
        cap = int(np.percentile(chk_arr, 98))
        axes[1].hist(chk_arr[chk_arr <= cap], bins=35, color="#2ca02c", edgecolor="black", alpha=0.7)
        axes[1].axvline(np.mean(chk_arr), color="red", linestyle="--", label=f"Mean: {np.mean(chk_arr):.1f}")
        axes[1].axvline(np.median(chk_arr), color="green", linestyle=":", label=f"Median: {np.median(chk_arr):.1f}")
        axes[1].set_title(f"Chunks per Document (Capped at 98th %: {cap} chunks)")
        axes[1].set_xlabel("Chunks per Document")
        axes[1].set_ylabel("Frequency")
        axes[1].legend()
        axes[1].grid(True, linestyle="--", alpha=0.5)

        plt.tight_layout()
        fig_path = self.results_figures_dir / "chunking_distributions.png"
        plt.savefig(fig_path, dpi=300)
        plt.close()
        print(f"Saved distribution plot to {fig_path}")


if __name__ == "__main__":
    pipeline = PreprocessingPipeline()
    pipeline.run()
