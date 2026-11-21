import json
from pathlib import Path
import pyarrow.parquet as pq
import numpy as np

DATA_DIR = Path("data/external/IL-TUR-PCR")
RESULTS_DIR = Path("results/tables")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SPLITS = [
    "train_queries",
    "train_candidates",
    "dev_queries",
    "dev_candidates",
    "test_queries",
    "test_candidates"
]

def to_list(val):
    if val is None:
        return []
    if isinstance(val, (list, tuple)):
        return list(val)
    if isinstance(val, np.ndarray):
        return val.tolist()
    return []

def analyze_splits():
    split_data = {}
    stats = {}
    all_ids_by_split = {}
    
    # 1. Load each split and inspect basic structure
    for split in SPLITS:
        parquet_path = DATA_DIR / split / f"{split}.parquet"
        table = pq.read_table(parquet_path)
        df = table.to_pandas()
        split_data[split] = df
        
        # Collect IDs and check validity
        ids = [str(x) if x is not None else "" for x in df['id'].tolist()]
        all_ids_by_split[split] = set(ids)
        
        null_ids = int(df['id'].isnull().sum())
        empty_ids = int(sum(1 for x in ids if not x.strip()))
        duplicate_ids_within = int(len(ids) - len(set(ids)))
        
        # Document length metrics (sentence count, word count)
        sentence_counts = []
        word_counts = []
        for text_val in df['text']:
            sents = to_list(text_val)
            sentence_counts.append(len(sents))
            w_count = sum(len(str(s).split()) for s in sents)
            word_counts.append(w_count)
            
        sentence_counts = np.array(sentence_counts, dtype=np.int64)
        word_counts = np.array(word_counts, dtype=np.int64)
        
        split_stat = {
            "record_count": int(len(df)),
            "null_ids": null_ids,
            "empty_ids": empty_ids,
            "duplicate_ids_within_split": duplicate_ids_within,
            "unique_ids": int(len(set(ids))),
            "avg_sentence_count": round(float(np.mean(sentence_counts)), 2) if len(sentence_counts) > 0 else 0.0,
            "min_sentence_count": int(np.min(sentence_counts)) if len(sentence_counts) > 0 else 0,
            "max_sentence_count": int(np.max(sentence_counts)) if len(sentence_counts) > 0 else 0,
            "median_sentence_count": float(np.median(sentence_counts)) if len(sentence_counts) > 0 else 0.0,
            "avg_word_count": round(float(np.mean(word_counts)), 2) if len(word_counts) > 0 else 0.0,
            "min_word_count": int(np.min(word_counts)) if len(word_counts) > 0 else 0,
            "max_word_count": int(np.max(word_counts)) if len(word_counts) > 0 else 0,
            "median_word_count": float(np.median(word_counts)) if len(word_counts) > 0 else 0.0,
            "total_sentences": int(np.sum(sentence_counts)),
            "total_words": int(np.sum(word_counts))
        }
        
        # Check relevant_candidates field if present
        if 'relevant_candidates' in df.columns:
            rel_counts = [len(to_list(r)) for r in df['relevant_candidates']]
            rel_counts = np.array(rel_counts, dtype=np.int64)
            split_stat["total_relevant_judgments_listed"] = int(np.sum(rel_counts))
            split_stat["avg_relevant_candidates_listed_per_query"] = round(float(np.mean(rel_counts)), 2) if len(rel_counts) > 0 else 0.0
            split_stat["min_relevant_candidates_listed"] = int(np.min(rel_counts)) if len(rel_counts) > 0 else 0
            split_stat["max_relevant_candidates_listed"] = int(np.max(rel_counts)) if len(rel_counts) > 0 else 0
            split_stat["median_relevant_candidates_listed"] = float(np.median(rel_counts)) if len(rel_counts) > 0 else 0.0
            split_stat["queries_with_zero_relevant_listed"] = int(np.sum(rel_counts == 0))
            
        stats[split] = split_stat

    # 2. Check candidate pool verification for query splits
    pairs = [
        ("train_queries", "train_candidates"),
        ("dev_queries", "dev_candidates"),
        ("test_queries", "test_candidates")
    ]
    
    query_candidate_analysis = {}
    for q_split, c_split in pairs:
        q_df = split_data[q_split]
        c_ids = all_ids_by_split[c_split]
        
        total_queries = len(q_df)
        total_citations = 0
        total_resolvable_in_pool = 0
        total_missing_from_pool = 0
        queries_with_zero_in_pool = 0
        queries_all_in_pool = 0
        
        resolvable_counts = []
        
        for _, row in q_df.iterrows():
            rels = to_list(row['relevant_candidates'])
            total_citations += len(rels)
            
            in_pool = [str(cid) for cid in rels if str(cid) in c_ids]
            resolvable_counts.append(len(in_pool))
            total_resolvable_in_pool += len(in_pool)
            total_missing_from_pool += (len(rels) - len(in_pool))
            
            if len(in_pool) == 0:
                queries_with_zero_in_pool += 1
            if len(in_pool) == len(rels) and len(rels) > 0:
                queries_all_in_pool += 1
                
        resolvable_counts = np.array(resolvable_counts, dtype=np.int64)
        avg_resolvable = float(np.mean(resolvable_counts)) if len(resolvable_counts) > 0 else 0.0
        
        query_candidate_analysis[f"{q_split}_vs_{c_split}"] = {
            "query_count": total_queries,
            "candidate_pool_size": len(c_ids),
            "total_relevant_judgments_listed": total_citations,
            "total_relevant_judgments_in_pool": total_resolvable_in_pool,
            "total_relevant_judgments_missing_from_pool": total_missing_from_pool,
            "relevance_pool_coverage_pct": round(100.0 * total_resolvable_in_pool / total_citations, 2) if total_citations > 0 else 0.0,
            "queries_with_zero_relevant_in_candidate_pool": queries_with_zero_in_pool,
            "pct_queries_with_zero_in_pool": round(100.0 * queries_with_zero_in_pool / total_queries, 2) if total_queries > 0 else 0.0,
            "queries_fully_covered_in_pool": queries_all_in_pool,
            "avg_resolvable_candidates_per_query": round(avg_resolvable, 2),
            "min_resolvable_candidates": int(np.min(resolvable_counts)) if len(resolvable_counts) > 0 else 0,
            "max_resolvable_candidates": int(np.max(resolvable_counts)) if len(resolvable_counts) > 0 else 0,
            "median_resolvable_candidates": float(np.median(resolvable_counts)) if len(resolvable_counts) > 0 else 0.0
        }
        
        stats[q_split]["total_relevant_in_pool"] = total_resolvable_in_pool
        stats[q_split]["queries_with_zero_relevant_in_pool"] = queries_with_zero_in_pool
        stats[q_split]["avg_relevant_in_pool_per_query"] = round(avg_resolvable, 2)

    # 3. Check ID overlap across splits
    overlap_matrix = {}
    for s1 in SPLITS:
        overlap_matrix[s1] = {}
        for s2 in SPLITS:
            overlap_matrix[s1][s2] = len(all_ids_by_split[s1].intersection(all_ids_by_split[s2]))
            
    # Cross partition checks
    all_query_ids = all_ids_by_split["train_queries"] | all_ids_by_split["dev_queries"] | all_ids_by_split["test_queries"]
    all_candidate_ids = all_ids_by_split["train_candidates"] | all_ids_by_split["dev_candidates"] | all_ids_by_split["test_candidates"]
    query_candidate_overlap = len(all_query_ids.intersection(all_candidate_ids))
    
    train_dev_overlap = len(
        (all_ids_by_split["train_queries"] | all_ids_by_split["train_candidates"]).intersection(
            all_ids_by_split["dev_queries"] | all_ids_by_split["dev_candidates"]
        )
    )
    train_test_overlap = len(
        (all_ids_by_split["train_queries"] | all_ids_by_split["train_candidates"]).intersection(
            all_ids_by_split["test_queries"] | all_ids_by_split["test_candidates"]
        )
    )
    dev_test_overlap = len(
        (all_ids_by_split["dev_queries"] | all_ids_by_split["dev_candidates"]).intersection(
            all_ids_by_split["test_queries"] | all_ids_by_split["test_candidates"]
        )
    )

    final_report = {
        "dataset_name": "Exploration-Lab/IL-TUR",
        "configuration": "pcr",
        "storage_format": "Parquet",
        "provenance": {
            "source_repo": "https://huggingface.co/datasets/Exploration-Lab/IL-TUR",
            "benchmark_paper": "IL-TUR: Benchmark for Indian Legal Text Understanding and Reasoning (ACL 2024)",
            "local_storage_path": "data/external/IL-TUR-PCR",
            "access_tier": "Authenticated (Gated non-commercial research)",
            "file_format": "Apache Parquet (lossless typed schema)"
        },
        "split_statistics": stats,
        "query_candidate_alignment": query_candidate_analysis,
        "id_integrity": {
            "total_unique_case_ids_across_all_splits": len(all_query_ids | all_candidate_ids),
            "total_unique_query_ids": len(all_query_ids),
            "total_unique_candidate_ids": len(all_candidate_ids),
            "overlap_queries_and_candidates": query_candidate_overlap,
            "train_dev_partition_overlap": train_dev_overlap,
            "train_test_partition_overlap": train_test_overlap,
            "dev_test_partition_overlap": dev_test_overlap,
            "pairwise_split_overlap": overlap_matrix
        }
    }
    
    # Save to JSON
    json_path = RESULTS_DIR / "iltur_pcr_split_statistics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2)
    print(f"Validation statistics successfully written to {json_path}")
    
    # Print summary
    print("\n================== SPLIT SUMMARY STATISTICS ==================")
    for s in SPLITS:
        st = stats[s]
        print(f"Split: {s:<18} | Records: {st['record_count']:<5} | Avg Sents: {st['avg_sentence_count']:<6} | Avg Words: {st['avg_word_count']:<8} | Null IDs: {st['null_ids']}")
        
    print("\n============= QUERY-CANDIDATE POOL VERIFICATION =============")
    for k, v in query_candidate_analysis.items():
        print(f"Pair: {k}")
        print(f"  Queries: {v['query_count']} | Candidate Pool: {v['candidate_pool_size']}")
        print(f"  Total Relevant Citations Listed: {v['total_relevant_judgments_listed']}")
        print(f"  Total Relevant in Pool: {v['total_relevant_judgments_in_pool']} ({v['relevance_pool_coverage_pct']}%)")
        print(f"  Missing from Pool: {v['total_relevant_judgments_missing_from_pool']}")
        print(f"  Queries with 0 Relevant in Pool: {v['queries_with_zero_relevant_in_candidate_pool']} ({v['pct_queries_with_zero_in_pool']}%)")
        print(f"  Avg Relevant in Pool per Query: {v['avg_resolvable_candidates_per_query']}")
        print(f"  Min / Median / Max in Pool: {v['min_resolvable_candidates']} / {v['median_resolvable_candidates']} / {v['max_resolvable_candidates']}")

if __name__ == "__main__":
    analyze_splits()
