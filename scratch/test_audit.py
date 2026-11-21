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

regimes_linear = {
    '1. TF-IDF Only': [0.0, 1.0, 0.0, 0.0],
    '2. Dense Only': [0.0, 0.0, 1.0, 0.0],
    '3. Legal Only': [0.0, 0.0, 0.0, 1.0],
    '4. BM25 Only': [1.0, 0.0, 0.0, 0.0],
    '5. TFIDF + Dense (0.75 / 0.25)': [0.0, 0.75, 0.25, 0.0],
    '6. TFIDF + Legal (0.85 / 0.15)': [0.0, 0.85, 0.0, 0.15],
    '7. TFIDF + Legal (0.90 / 0.10)': [0.0, 0.90, 0.0, 0.10],
    '8. Dense + Legal (0.70 / 0.30)': [0.0, 0.0, 0.70, 0.30],
    '9. Full Hybrid (0 / 0.65 / 0.25 / 0.10)': [0.0, 0.65, 0.25, 0.10]
}

print(f"{'Regime':<38} {'P@1':<8} {'MRR':<8} {'nDCG@5':<8} {'nDCG@10':<8} {'R@10':<8}")
print("-" * 78)
for name, w in regimes_linear.items():
    s = sum(w[i] * norm_dev[f'_qnorm_{features[i]}'].values for i in range(4))
    dev_df['score_tmp'] = s
    m = HybridEvaluator.evaluate_predictions(dev_df, score_col='score_tmp')
    p1 = m['P@1']
    mrr = m['MRR']
    ndcg5 = m['nDCG@5']
    ndcg10 = m['nDCG@10']
    r10 = m['R@10']
    print(f"{name:<38} {p1:<8.4f} {mrr:<8.4f} {ndcg5:<8.4f} {ndcg10:<8.4f} {r10:<8.4f}")

