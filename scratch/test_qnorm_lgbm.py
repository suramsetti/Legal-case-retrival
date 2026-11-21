import sys
sys.path.append('.')
import pandas as pd
import numpy as np
import lightgbm as lgb
from src.evaluation.hybrid_evaluator import HybridEvaluator

train_df = pd.read_parquet('data/processed/IL-TUR-PCR/train/hybrid_features.parquet').sort_values('query_id')
dev_df = pd.read_parquet('data/processed/IL-TUR-PCR/dev/hybrid_features.parquet').sort_values('query_id')

def add_query_normalized_features(df):
    df = df.copy()
    feature_cols = [
        'bm25_score', 'tfidf_score', 'inlegalbert_sim',
        'statute_shared_count', 'citation_shared_count', 'entity_shared_count',
        'statute_jaccard', 'statute_containment_query', 'citation_jaccard',
        'entity_jaccard', 'judge_overlap', 'court_overlap', 'org_overlap',
        'rhetorical_cosine', 'rhetorical_js_sim', 'rhetorical_ratio_diff',
        'composite_legal_score'
    ]
    for col in feature_cols:
        def q_min_max(s):
            mn, mx = s.min(), s.max()
            return (s - mn) / (mx - mn) if mx > mn else np.zeros_like(s)
        df[f'qnorm_{col}'] = df.groupby('query_id')[col].transform(q_min_max)
    return df

train_norm = add_query_normalized_features(train_df)
dev_norm = add_query_normalized_features(dev_df)

q_cols = [c for c in train_norm.columns if c.startswith('qnorm_')]
print(f"Testing with {len(q_cols)} query-normalized features: {q_cols}")

X_train = train_norm[q_cols].values
y_train = train_norm['label'].values.astype(int)
train_groups = train_norm.groupby('query_id', sort=False).size().to_numpy()
X_dev = dev_norm[q_cols].values

for nl in [7, 15, 31]:
    for lr in [0.03, 0.05]:
        ranker = lgb.LGBMRanker(
            objective='lambdarank',
            n_estimators=100,
            learning_rate=lr,
            num_leaves=nl,
            random_state=42,
            verbose=-1
        )
        ranker.fit(X_train, y_train, group=train_groups)
        dev_norm['lgbm_score'] = ranker.predict(X_dev)
        m = HybridEvaluator.evaluate_predictions(dev_norm, score_col='lgbm_score')
        print(f"nl={nl}, lr={lr} -> nDCG@10={m['nDCG@10']:.4f}, MRR={m['MRR']:.4f}, P@1={m['P@1']:.4f}, R@10={m['R@10']:.4f}")

