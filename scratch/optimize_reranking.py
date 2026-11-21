import sys
sys.path.insert(0, ".")
import pandas as pd
import numpy as np
from src.evaluation.hybrid_evaluator import HybridEvaluator

data_dir = "data/processed/IL-TUR-PCR"
train_df = pd.read_parquet(f"{data_dir}/train/hybrid_features.parquet")
dev_df = pd.read_parquet(f"{data_dir}/dev/hybrid_features.parquet")
test_df = pd.read_parquet(f"{data_dir}/test/hybrid_features.parquet")

for df in [train_df, dev_df, test_df]:
    for col in ['tfidf_score', 'inlegalbert_sim', 'composite_legal_score', 'bm25_score', 'statute_containment_query']:
        s_min = df.groupby('query_id')[col].transform('min')
        s_max = df.groupby('query_id')[col].transform('max')
        diff = s_max - s_min
        df['qn_' + col] = (df[col] - s_min) / np.where(diff > 1e-9, diff, 1.0)

    # Base Stage 1 score
    df['stage1_score'] = 0.65 * df['qn_tfidf_score'] + 0.25 * df['qn_inlegalbert_sim'] + 0.10 * df['qn_composite_legal_score']
    df['s1_rank'] = df.groupby('query_id')['stage1_score'].rank(ascending=False, method='first')

# Let's inspect what happens if we search for optimal weights across ALL features:
# tfidf, dense, legal, bm25, statute_containment, rhetorical_cosine
best_dev_p1 = 0.0
best_weights = None

# Grid search on DEV split strictly
for w_ti in [0.45, 0.50, 0.55, 0.60, 0.65]:
    for w_de in [0.15, 0.20, 0.25, 0.30]:
        for w_bm in [0.00, 0.05, 0.10, 0.15]:
            for w_leg in [0.05, 0.10, 0.15, 0.20]:
                for w_rhet in [0.00, 0.05, 0.10]:
                    tot = w_ti + w_de + w_bm + w_leg + w_rhet
                    if abs(tot - 1.0) > 1e-5:
                        continue
                    
                    dev_df['test_score'] = (
                        w_ti * dev_df['qn_tfidf_score'] +
                        w_de * dev_df['qn_inlegalbert_sim'] +
                        w_bm * dev_df['qn_bm25_score'] +
                        w_leg * dev_df['qn_composite_legal_score'] +
                        w_rhet * dev_df['rhetorical_cosine']
                    )
                    
                    dev_df['r'] = dev_df.groupby('query_id')['test_score'].rank(ascending=False, method='first')
                    hits = (dev_df[dev_df['label'] == 1]['r'] == 1).sum()
                    p1 = hits / 118.0
                    if p1 > best_dev_p1:
                        best_dev_p1 = p1
                        best_weights = (w_ti, w_de, w_bm, w_leg, w_rhet)
                        print(f"New Best Dev P@1 = {best_dev_p1:.4f} ({hits}/118) with weights: {best_weights}")

print(f"\nFinal Best Dev P@1: {best_dev_p1:.4f} with weights: {best_weights}")

# Evaluate on test with the best dev weights
w_ti, w_de, w_bm, w_leg, w_rhet = best_weights
test_df['final_score'] = (
    w_ti * test_df['qn_tfidf_score'] +
    w_de * test_df['qn_inlegalbert_sim'] +
    w_bm * test_df['qn_bm25_score'] +
    w_leg * test_df['qn_composite_legal_score'] +
    w_rhet * test_df['rhetorical_cosine']
)
m_test = HybridEvaluator.evaluate_predictions(test_df, score_col='final_score')
print(f"Test with Best Dev Weights: P@1={m_test['P@1']:.4f}, MRR={m_test['MRR']:.4f}, nDCG@10={m_test['nDCG@10']:.4f}")

