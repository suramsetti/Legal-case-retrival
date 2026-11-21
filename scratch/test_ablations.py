import sys
sys.path.append('.')
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from src.evaluation.hybrid_evaluator import HybridEvaluator

train_df = pd.read_parquet('data/processed/IL-TUR-PCR/train/hybrid_features.parquet').sort_values('query_id')
dev_df = pd.read_parquet('data/processed/IL-TUR-PCR/dev/hybrid_features.parquet').sort_values('query_id')

lexical_cols = ['norm_bm25_score', 'norm_bm25_rank', 'norm_tfidf_score', 'norm_tfidf_rank']
dense_cols = ['norm_inlegalbert_sim', 'norm_inlegalbert_rank']
legal_cols = [c for c in train_df.columns if c.startswith('norm_') and c not in lexical_cols and c not in dense_cols]

regimes = {
    "1_Lexical_Only": lexical_cols,
    "2_Dense_Only": dense_cols,
    "3_Legal_Only": legal_cols,
    "4_Lexical_Dense": lexical_cols + dense_cols,
    "5_Lexical_Legal": lexical_cols + legal_cols,
    "6_Dense_Legal": dense_cols + legal_cols,
    "7_Full_Hybrid": lexical_cols + dense_cols + legal_cols,
}

y_train = train_df['label'].values.astype(int)

print(f"{'Regime':<20} {'P@1':<8} {'P@5':<8} {'P@10':<8} {'MRR':<8} {'nDCG@5':<8} {'nDCG@10':<8}")
print("-" * 68)

for name, cols in regimes.items():
    clf = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    clf.fit(train_df[cols], y_train)
    dev_df['score'] = clf.predict_proba(dev_df[cols])[:, 1]
    m = HybridEvaluator.evaluate_predictions(dev_df, score_col='score')
    print(f"{name:<20} {m['P@1']:<8.4f} {m['P@5']:<8.4f} {m['P@10']:<8.4f} {m['MRR']:<8.4f} {m['nDCG@5']:<8.4f} {m['nDCG@10']:<8.4f}")

