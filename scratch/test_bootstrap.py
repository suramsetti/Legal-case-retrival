import sys
sys.path.append('.')
import pandas as pd
import numpy as np
from src.evaluation.hybrid_evaluator import HybridEvaluator
from src.ranking.weighted_fusion import WeightedLinearFusion

dev_df = pd.read_parquet('data/processed/IL-TUR-PCR/dev/hybrid_features.parquet').sort_values('query_id')
features = ['bm25_score', 'tfidf_score', 'inlegalbert_sim', 'composite_legal_score']
wf = WeightedLinearFusion(feature_cols=features, use_query_min_max=True)
norm_dev = wf.normalize_features(dev_df)

# Systems:
# 1. TF-IDF
norm_dev['score_tfidf'] = norm_dev['tfidf_score'].values

# 2. Weighted Fusion / Full Hybrid: w = [0.0, 0.65, 0.25, 0.10]
norm_dev['score_weighted'] = (
    0.65 * norm_dev['_qnorm_tfidf_score'].values +
    0.25 * norm_dev['_qnorm_inlegalbert_sim'].values +
    0.10 * norm_dev['_qnorm_composite_legal_score'].values
)

# 3. TF-IDF + Legal: w = [0.0, 0.85, 0.0, 0.15]
norm_dev['score_tfidf_legal'] = (
    0.85 * norm_dev['_qnorm_tfidf_score'].values +
    0.15 * norm_dev['_qnorm_composite_legal_score'].values
)

# Compute per-query metrics
queries = []
qids = norm_dev['query_id'].unique()

for qid in qids:
    g = norm_dev[norm_dev['query_id'] == qid]
    lbls = g['label'].values.astype(int)
    pos = int(np.sum(lbls))
    if pos == 0:
        continue
    
    def q_metrics(score_col):
        s_g = g.sort_values(score_col, ascending=False)
        l = s_g['label'].values.astype(int)
        rel_idx = np.where(l == 1)[0]
        mrr = 1.0 / (rel_idx[0] + 1) if len(rel_idx) > 0 else 0.0
        
        # ndcg@10
        r_top10 = l[:10]
        dcg = np.sum(r_top10 / np.log2(np.arange(2, len(r_top10) + 2)))
        r_ideal = [1] * min(pos, 10) + [0] * max(0, 10 - pos)
        idcg = np.sum(r_ideal / np.log2(np.arange(2, len(r_ideal) + 2)))
        ndcg10 = dcg / idcg if idcg > 0 else 0.0
        return mrr, ndcg10

    mrr_t, ndcg_t = q_metrics('score_tfidf')
    mrr_w, ndcg_w = q_metrics('score_weighted')
    mrr_tl, ndcg_tl = q_metrics('score_tfidf_legal')
    
    queries.append({
        'qid': qid,
        'mrr_tfidf': mrr_t,
        'ndcg10_tfidf': ndcg_t,
        'mrr_weighted': mrr_w,
        'ndcg10_weighted': ndcg_w,
        'mrr_tfidf_legal': mrr_tl,
        'ndcg10_tfidf_legal': ndcg_tl,
    })

df_q = pd.DataFrame(queries)
print(f"Total evaluated queries: {len(df_q)}")

# Paired Bootstrap Function
def paired_bootstrap(diffs, n_boot=2000, seed=42):
    rng = np.random.RandomState(seed)
    n = len(diffs)
    boot_means = np.zeros(n_boot)
    for b in range(n_boot):
        sample = rng.choice(diffs, size=n, replace=True)
        boot_means[b] = np.mean(sample)
    ci_lower = np.percentile(boot_means, 2.5)
    ci_upper = np.percentile(boot_means, 97.5)
    orig_mean = np.mean(diffs)
    p_val = np.mean(boot_means <= 0.0) if orig_mean > 0 else np.mean(boot_means >= 0.0)
    return orig_mean, ci_lower, ci_upper, p_val

comparisons = [
    ("Weighted Fusion vs TF-IDF", df_q['mrr_weighted'] - df_q['mrr_tfidf'], df_q['ndcg10_weighted'] - df_q['ndcg10_tfidf']),
    ("TFIDF+Legal vs TF-IDF", df_q['mrr_tfidf_legal'] - df_q['mrr_tfidf'], df_q['ndcg10_tfidf_legal'] - df_q['ndcg10_tfidf']),
]

for name, diff_mrr, diff_ndcg in comparisons:
    m_mrr, l_mrr, u_mrr, p_mrr = paired_bootstrap(diff_mrr)
    m_ndcg, l_ndcg, u_ndcg, p_ndcg = paired_bootstrap(diff_ndcg)
    print(f"\n--- {name} ---")
    print(f"  MRR Diff:    mean={m_mrr:+.4f}, 95% CI=[{l_mrr:+.4f}, {u_mrr:+.4f}], p={p_mrr:.4f}")
    print(f"  nDCG@10 Diff: mean={m_ndcg:+.4f}, 95% CI=[{l_ndcg:+.4f}, {u_ndcg:+.4f}], p={p_ndcg:.4f}")

