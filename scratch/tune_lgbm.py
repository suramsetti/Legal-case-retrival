import sys
sys.path.append('.')
import pandas as pd
import numpy as np
import lightgbm as lgb
from src.evaluation.hybrid_evaluator import HybridEvaluator

train_df = pd.read_parquet('data/processed/IL-TUR-PCR/train/hybrid_features.parquet').sort_values('query_id')
dev_df = pd.read_parquet('data/processed/IL-TUR-PCR/dev/hybrid_features.parquet').sort_values('query_id')

feature_cols = [c for c in train_df.columns if c.startswith('norm_') and 'rank' not in c]

X_train = train_df[feature_cols].values
y_train = train_df['label'].values.astype(int)
train_groups = train_df.groupby('query_id', sort=False).size().to_numpy()
X_dev = dev_df[feature_cols].values

print("Testing LightGBM configurations:")
for nl in [4, 7, 15]:
    for lr in [0.03, 0.05, 0.1]:
        ranker = lgb.LGBMRanker(
            objective='lambdarank',
            n_estimators=100,
            learning_rate=lr,
            num_leaves=nl,
            min_child_samples=50,
            random_state=42,
            verbose=-1
        )
        ranker.fit(X_train, y_train, group=train_groups)
        dev_df['lgbm_score'] = ranker.predict(X_dev)
        m = HybridEvaluator.evaluate_predictions(dev_df, score_col='lgbm_score')
        print(f"nl={nl}, lr={lr} -> nDCG@10={m['nDCG@10']:.4f}, MRR={m['MRR']:.4f}, P@1={m['P@1']:.4f}")
