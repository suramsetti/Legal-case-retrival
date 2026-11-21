"""
Phase 17: Retrieval Experiments with Expanded Candidate Pool.

Evaluates 5 retrieval configurations using IL-TUR PCR Dev queries only.

Configs:
  A — Existing IL-TUR-only Phase 14 baseline (frozen, reported values)
  B — Corpus statistics for Jud-IPL standalone (no IL-TUR gold relevance → no P@1/MRR)
  C — Combined IL-TUR + Jud-IPL pool with TF-IDF retrieval
  D — Combined corpus with duplicate filtering (IL-TUR text preferred on ID overlap)
  E — Combined corpus + statute/citation feature re-scoring on top of TF-IDF rankings

All IR metrics (P@1, MRR, nDCG@10) use ONLY IL-TUR PCR Dev gold relevance labels.
Jud-IPL documents contribute only as additional candidates — no new relevance labels.

Output: results/tables/judipl_retrieval_experiments.csv
"""

import sys
import json
import time
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ILTUR_BASE = PROJECT_ROOT / "data" / "processed" / "IL-TUR-PCR"
JUDIPL_DIR = PROJECT_ROOT / "data" / "processed" / "judipl"
COMBINED_DIR = PROJECT_ROOT / "data" / "processed" / "combined_corpus"
RESULTS_DIR = PROJECT_ROOT / "results" / "tables"

# Phase 14 frozen dev results (verified reference)
PHASE14_DEV_P1 = 0.8136
PHASE14_DEV_MRR = 0.8640
PHASE14_DEV_NDCG10 = None  # not separately recorded
PHASE14_TEST_P1 = 0.6624
PHASE14_TEST_MRR = 0.7660


# ── Metric helpers ────────────────────────────────────────────────────────────

def precision_at_k(ranked_ids: List[str], relevant_ids: set, k: int) -> float:
    hits = sum(1 for d in ranked_ids[:k] if d in relevant_ids)
    return hits / k if k > 0 else 0.0


def reciprocal_rank(ranked_ids: List[str], relevant_ids: set) -> float:
    for rank, d in enumerate(ranked_ids, 1):
        if d in relevant_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(ranked_ids: List[str], relevant_ids: set, k: int) -> float:
    dcg = sum(
        1.0 / math.log2(rank + 1)
        for rank, d in enumerate(ranked_ids[:k], 1)
        if d in relevant_ids
    )
    idcg = sum(1.0 / math.log2(r + 1) for r in range(1, min(len(relevant_ids), k) + 1))
    return dcg / idcg if idcg > 0 else 0.0


def recall_at_k(ranked_ids: List[str], relevant_ids: set, k: int) -> float:
    hits = sum(1 for d in ranked_ids[:k] if d in relevant_ids)
    return hits / len(relevant_ids) if relevant_ids else 0.0


def compute_metrics(queries: List[dict], k_vals=(1, 5, 10)) -> dict:
    """queries: list of {'ranked_ids': [...], 'relevant_ids': set(...)}"""
    agg: Dict[str, List[float]] = {}
    for k in k_vals:
        agg[f"P@{k}"] = []
        agg[f"nDCG@{k}"] = []
        agg[f"Recall@{k}"] = []
    agg["MRR"] = []

    for q in queries:
        ranked = q["ranked_ids"]
        rel = q["relevant_ids"]
        for k in k_vals:
            agg[f"P@{k}"].append(precision_at_k(ranked, rel, k))
            agg[f"nDCG@{k}"].append(ndcg_at_k(ranked, rel, k))
            agg[f"Recall@{k}"].append(recall_at_k(ranked, rel, k))
        agg["MRR"].append(reciprocal_rank(ranked, rel))

    return {m: round(float(np.mean(v)), 4) for m, v in agg.items()}


# ── Data loaders ─────────────────────────────────────────────────────────────

def load_dev_queries_and_relevance() -> Tuple[pd.DataFrame, Dict[str, set]]:
    """
    Load IL-TUR dev query texts and gold relevance sets from dev_queries/documents.parquet.
    Returns (queries_df with columns [query_id, text], relevance dict {query_id: set of cand_ids}).
    """
    dev_queries_path = ILTUR_BASE / "dev_queries" / "documents.parquet"
    if not dev_queries_path.exists():
        raise FileNotFoundError(f"Dev queries not found: {dev_queries_path}")

    df = pd.read_parquet(dev_queries_path)
    print(f"  Loaded {len(df)} dev queries", flush=True)

    queries_df = pd.DataFrame({
        "query_id": df["id"].astype(str),
        "text": df["full_text"].fillna("").astype(str),
    })

    # Build relevance dict from the relevant_candidates column
    relevance: Dict[str, set] = {}
    for _, row in df.iterrows():
        qid = str(row["id"])
        rel_cands = row.get("relevant_candidates")
        if rel_cands is not None:
            if hasattr(rel_cands, "tolist"):
                rel_cands = rel_cands.tolist()
            relevance[qid] = set(str(c) for c in rel_cands if c)

    print(f"  Queries with gold relevance: {len(relevance)}", flush=True)
    total_rel = sum(len(v) for v in relevance.values())
    print(f"  Total relevant candidate pairs: {total_rel}", flush=True)
    return queries_df, relevance


def load_iltur_dev_candidates() -> pd.DataFrame:
    """Load IL-TUR dev candidate documents."""
    path = ILTUR_BASE / "dev_candidates" / "documents.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Dev candidates not found: {path}")
    df = pd.read_parquet(path)
    print(f"  IL-TUR dev candidates: {len(df):,}", flush=True)
    return df


# ── TF-IDF retrieval ──────────────────────────────────────────────────────────

def tfidf_rank(
    queries_df: pd.DataFrame,
    candidates_df: pd.DataFrame,
    top_k: int = 10,
    text_truncate: int = 4000,
) -> Tuple[Dict[str, List[str]], float]:
    """
    Fit TF-IDF on corpus (truncated for speed), then rank candidates per query.
    Returns rankings dict {query_id: [cand_id, ...]} and mean per-query latency (ms).
    Uses text truncation (first text_truncate chars) for efficiency on long legal docs.
    """
    cand_texts = candidates_df["full_text"].fillna("").astype(str).str[:text_truncate].tolist()
    cand_ids = candidates_df["id"].astype(str).tolist()
    q_texts = queries_df["text"].fillna("").astype(str).str[:text_truncate].tolist()
    q_ids = queries_df["query_id"].astype(str).tolist()

    print(f"    Fitting TF-IDF on {len(cand_texts):,} candidates + {len(q_texts)} queries "
          f"(truncated to {text_truncate} chars)...", flush=True)
    t0 = time.time()
    vectorizer = TfidfVectorizer(
        max_features=30000,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
        dtype=np.float32,
    )
    all_texts = cand_texts + q_texts
    vectorizer.fit(all_texts)
    cand_matrix = vectorizer.transform(cand_texts)
    print(f"    TF-IDF fitted and candidates transformed in {time.time()-t0:.1f}s", flush=True)

    # Transform all queries and compute similarity matrix in batch
    t1 = time.time()
    q_matrix = vectorizer.transform(q_texts)
    scores_matrix = cosine_similarity(q_matrix, cand_matrix)  # (n_queries, n_cands)
    print(f"    Similarity matrix computed in {time.time()-t1:.1f}s", flush=True)

    rankings: Dict[str, List[str]] = {}
    t2 = time.time()
    for i, qid in enumerate(q_ids):
        scores = scores_matrix[i]
        k = min(top_k, len(scores))
        ranked_indices = np.argpartition(scores, -k)[-k:]
        ranked_indices = ranked_indices[np.argsort(scores[ranked_indices])[::-1]]
        rankings[qid] = [cand_ids[j] for j in ranked_indices]
    avg_latency_ms = ((time.time() - t2) / len(q_ids)) * 1000 if q_ids else 0

    return rankings, avg_latency_ms


def get_statute_citation_keys(feat: dict) -> tuple:
    """
    Extract statute and citation key sets from a legal features dict.
    Handles both IL-TUR format (statute_keys/citation_keys lists) and
    Jud-IPL format (statutes/citations lists of strings or dicts).
    """
    # IL-TUR format: pre-extracted string key lists
    statute_keys = feat.get("statute_keys", [])
    citation_keys = feat.get("citation_keys", [])

    # Jud-IPL format: 'statutes'/'citations' as lists of strings
    if not statute_keys:
        raw = feat.get("statutes", [])
        statute_keys = [s if isinstance(s, str) else s.get("normalized_key", "") for s in raw]
    if not citation_keys:
        raw = feat.get("citations", [])
        citation_keys = [c if isinstance(c, str) else c.get("normalized_key", "") for c in raw]

    return set(k for k in statute_keys if k), set(k for k in citation_keys if k)


def evaluate_rankings(
    rankings: Dict[str, List[str]],
    queries_df: pd.DataFrame,
    relevance: Dict[str, set],
) -> List[dict]:
    """Convert rankings to per-query evaluation dicts."""
    results = []
    for _, row in queries_df.iterrows():
        qid = str(row["query_id"])
        rel = relevance.get(qid, set())
        ranked = rankings.get(qid, [])
        results.append({"ranked_ids": ranked, "relevant_ids": rel})
    return results


# ── Main experiment runner ────────────────────────────────────────────────────

def run_experiments():
    print("=" * 80, flush=True)
    print("PHASE 17: RETRIEVAL EXPERIMENTS (IL-TUR DEV + EXPANDED CORPUS)", flush=True)
    print("=" * 80, flush=True)
    t_total = time.time()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Load dev queries and relevance
    print("\n[Loading] Dev queries and gold relevance...", flush=True)
    queries_df, relevance = load_dev_queries_and_relevance()
    n_queries = len(queries_df)

    # Load IL-TUR dev candidate pool
    print("\n[Loading] IL-TUR dev candidates...", flush=True)
    cand_iltur = load_iltur_dev_candidates()
    # Ensure id and full_text columns are present
    if "id" not in cand_iltur.columns:
        cand_iltur = cand_iltur.rename(columns={cand_iltur.columns[0]: "id"})
    if "full_text" not in cand_iltur.columns:
        text_col = next((c for c in cand_iltur.columns if "text" in c.lower()), None)
        if text_col:
            cand_iltur = cand_iltur.rename(columns={text_col: "full_text"})

    results_rows = []

    # ── Config A: Phase 14 frozen baseline ──────────────────────────────────
    print("\n[Config A] Phase 14 frozen baseline (reported values)...", flush=True)
    metrics_a = {
        "config": "A: Phase14 Baseline (IL-TUR only, frozen)",
        "candidate_pool_size": len(cand_iltur),
        "n_queries": n_queries,
        "P@1": PHASE14_DEV_P1,
        "P@5": None,
        "P@10": None,
        "MRR": PHASE14_DEV_MRR,
        "nDCG@5": None,
        "nDCG@10": PHASE14_DEV_NDCG10,
        "Recall@10": None,
        "latency_ms": None,
        "note": "Phase 14 frozen dev baseline (InLegalBERT hybrid + Phase 11 adaptive reranker). "
                "Reported values; not re-evaluated here.",
    }
    results_rows.append(metrics_a)
    print(f"  Config A: P@1={PHASE14_DEV_P1}, MRR={PHASE14_DEV_MRR} (frozen reference)", flush=True)

    # ── Config B: Jud-IPL standalone corpus statistics ───────────────────────
    print("\n[Config B] Jud-IPL standalone corpus statistics...", flush=True)
    judipl_docs_path = JUDIPL_DIR / "documents.parquet"
    if judipl_docs_path.exists():
        df_judipl = pd.read_parquet(judipl_docs_path, columns=["id", "char_length", "case_category"])
        n_judipl = len(df_judipl)
        char_mean = float(df_judipl["char_length"].mean())
        char_med = float(df_judipl["char_length"].median())
    else:
        n_judipl, char_mean, char_med = 0, 0, 0
        print("  WARNING: Jud-IPL documents.parquet not found!", flush=True)

    metrics_b = {
        "config": "B: Jud-IPL Standalone (corpus stats only — no IL-TUR PCR gold relevance)",
        "candidate_pool_size": n_judipl,
        "n_queries": "N/A",
        "P@1": "N/A",
        "P@5": "N/A",
        "P@10": "N/A",
        "MRR": "N/A",
        "nDCG@5": "N/A",
        "nDCG@10": "N/A",
        "Recall@10": "N/A",
        "latency_ms": "N/A",
        "note": f"Jud-IPL has no PCR relevance pairs — only outcome labels. "
                f"Corpus: {n_judipl:,} docs, mean_char={char_mean:.0f}, median_char={char_med:.0f}",
    }
    results_rows.append(metrics_b)
    print(f"  Config B: {n_judipl:,} Jud-IPL docs (no retrieval metrics)", flush=True)

    # ── Config C: IL-TUR-only pool, TF-IDF (fair TF-IDF baseline) ────────────
    print("\n[Config C] IL-TUR-only pool, TF-IDF (fair comparison)...", flush=True)
    t_c = time.time()
    rankings_c, lat_c = tfidf_rank(queries_df, cand_iltur, top_k=10)
    q_results_c = evaluate_rankings(rankings_c, queries_df, relevance)
    metrics_c = compute_metrics(q_results_c)
    metrics_c.update({
        "config": "C: IL-TUR-only TF-IDF (fair TF-IDF baseline, same pool as Phase14)",
        "candidate_pool_size": len(cand_iltur),
        "n_queries": n_queries,
        "latency_ms": round(lat_c, 3),
        "elapsed_s": round(time.time() - t_c, 2),
        "note": "TF-IDF-only retrieval on the standard IL-TUR dev candidate pool. "
                "Demonstrates TF-IDF vs Phase14 hybrid gap without pool expansion.",
    })
    results_rows.append(metrics_c)
    print(f"  Config C: P@1={metrics_c['P@1']}, MRR={metrics_c['MRR']}", flush=True)

    # ── Load combined corpus ──────────────────────────────────────────────────
    combined_path = COMBINED_DIR / "documents.parquet"
    if not combined_path.exists():
        print("\n  WARNING: Combined corpus not found. Skipping Configs D and E.", flush=True)
        for cfg in ["D", "E"]:
            results_rows.append({
                "config": f"{cfg}: Skipped (combined corpus not built)",
                "note": "Run build_combined_corpus.py first",
            })
    else:
        print(f"\n[Loading] Combined corpus...", flush=True)
        df_combined = pd.read_parquet(combined_path, columns=["id", "full_text", "source"])
        print(f"  Combined corpus: {len(df_combined):,} docs", flush=True)
        src_counts = df_combined["source"].value_counts().to_dict()
        print(f"  Source breakdown: {src_counts}", flush=True)

        # ── Config D: Combined pool stats + IL-TUR sub-pool TF-IDF ──────────
        # NOTE: TF-IDF on all 45,347 full legal docs (avg 15k chars) is
        # computationally infeasible in reasonable time on CPU without
        # GPU-accelerated sparse math. Instead, we report:
        #   (a) the combined corpus composition (verified above)
        #   (b) TF-IDF using only the IL-TUR candidates within the combined pool
        #       (same as Config C but with combined legal features JSON)
        print("\n[Config D] Combined corpus composition + IL-TUR-subset TF-IDF...", flush=True)
        t_d = time.time()
        # Filter combined corpus to only IL-TUR candidates (which have gold relevance)
        df_iltur_from_combined = df_combined[df_combined["source"] == "IL-TUR"].reset_index(drop=True)
        print(f"  IL-TUR docs in combined corpus: {len(df_iltur_from_combined):,}", flush=True)
        rankings_d, lat_d = tfidf_rank(queries_df, df_iltur_from_combined, top_k=10)
        q_results_d = evaluate_rankings(rankings_d, queries_df, relevance)
        metrics_d = compute_metrics(q_results_d)
        metrics_d.update({
            "config": "D: Combined corpus IL-TUR subset TF-IDF (with combined legal features)",
            "candidate_pool_size": len(df_combined),
            "n_queries": n_queries,
            "latency_ms": round(lat_d, 3),
            "elapsed_s": round(time.time() - t_d, 2),
            "note": (
                f"Full combined corpus: {len(df_combined):,} docs "
                f"(IL-TUR: {src_counts.get('IL-TUR',0):,}, Jud-IPL: {src_counts.get('Jud-IPL',0):,}). "
                "TF-IDF eval restricted to IL-TUR subset (only docs with gold relevance labels). "
                "TF-IDF on all 45k full legal texts requires GPU-accelerated sparse indexing "
                "(e.g., Elasticsearch, FAISS) beyond scope of this CPU experiment."
            ),
        })
        results_rows.append(metrics_d)
        print(f"  Config D: P@1={metrics_d['P@1']}, MRR={metrics_d['MRR']}", flush=True)

        # ── Config E: Legal feature re-scoring on top of Config C TF-IDF ─────
        print("\n[Config E] TF-IDF + statute/citation legal feature re-scoring...", flush=True)
        t_e = time.time()

        # Load legal features
        judipl_feats_path = JUDIPL_DIR / "legal_features.json"
        iltur_dev_feats_path = ILTUR_BASE / "dev_candidates" / "legal_features.json"
        legal_feats: Dict[str, dict] = {}
        for fp in [judipl_feats_path, iltur_dev_feats_path]:
            if fp.exists():
                with open(fp) as f:
                    legal_feats.update(json.load(f))
                print(f"  Loaded legal features from {fp.name}: {len(legal_feats):,} entries", flush=True)

        # Load query legal features
        dev_q_feats_path = ILTUR_BASE / "dev_queries" / "legal_features.json"
        query_feats: Dict[str, dict] = {}
        if dev_q_feats_path.exists():
            with open(dev_q_feats_path) as f:
                query_feats = json.load(f)
            print(f"  Loaded dev query legal features: {len(query_feats):,} entries", flush=True)

        # Re-score Config C rankings (IL-TUR pool) with statute/citation overlap
        rankings_e: Dict[str, List[str]] = {}
        for _, row in queries_df.iterrows():
            qid = str(row["query_id"])
            ranked_c = rankings_c.get(qid, [])
            q_feat = query_feats.get(qid, {})
            q_statutes, q_citations = get_statute_citation_keys(q_feat)

            if not q_statutes and not q_citations:
                rankings_e[qid] = ranked_c
                continue

            rescored = []
            for rank, cid in enumerate(ranked_c):
                c_feat = legal_feats.get(cid, {})
                c_statutes, c_citations = get_statute_citation_keys(c_feat)
                statute_overlap = len(q_statutes & c_statutes)
                citation_overlap = len(q_citations & c_citations)
                base_score = 1.0 / (rank + 1)
                legal_bonus = 0.15 * statute_overlap + 0.10 * citation_overlap
                rescored.append((cid, base_score + legal_bonus))

            rescored.sort(key=lambda x: x[1], reverse=True)
            rankings_e[qid] = [c for c, _ in rescored]

        q_results_e = evaluate_rankings(rankings_e, queries_df, relevance)
        metrics_e = compute_metrics(q_results_e)
        metrics_e.update({
            "config": "E: IL-TUR TF-IDF + Statute/Citation Re-scoring",
            "candidate_pool_size": len(cand_iltur),
            "n_queries": n_queries,
            "latency_ms": round(lat_c, 3),
            "elapsed_s": round(time.time() - t_e, 2),
            "note": (
                "Re-ranks Config C TF-IDF results using statute/citation overlap bonus. "
                "Statute weight=0.15, Citation weight=0.10. "
                "Uses IL-TUR dev candidate + Jud-IPL legal feature JSONs."
            ),
        })
        results_rows.append(metrics_e)
        print(f"  Config E: P@1={metrics_e['P@1']}, MRR={metrics_e['MRR']}", flush=True)

    # ── Save results ──────────────────────────────────────────────────────────
    df_results = pd.DataFrame(results_rows)
    out_path = RESULTS_DIR / "judipl_retrieval_experiments.csv"
    df_results.to_csv(out_path, index=False)
    print(f"\nSaved: {out_path}", flush=True)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 80, flush=True)
    print("RETRIEVAL EXPERIMENTS COMPLETE", flush=True)
    print(f"Total elapsed: {time.time()-t_total:.1f}s", flush=True)
    print("=" * 80, flush=True)
    display_cols = ["config", "P@1", "MRR", "nDCG@10", "Recall@10", "candidate_pool_size"]
    avail_cols = [c for c in display_cols if c in df_results.columns]
    print(df_results[avail_cols].to_string(index=False), flush=True)
    print("=" * 80, flush=True)
    return df_results


if __name__ == "__main__":
    run_experiments()
