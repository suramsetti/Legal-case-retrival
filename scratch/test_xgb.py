import sys
sys.path.append('.')
import pandas as pd
import numpy as np
import xgboost as xgb
from src.evaluation.hybrid_evaluator import HybridEvaluator

train_df = pd.read_parquet('data/processed/IL-TUR-PCR/train/hybrid_features.parquet').sort_values('query_id')
dev_df = pd.read_parquet('data/processed/IL-TUR-PCR/dev/hybrid_features.parquet').sort_values('query_id')

feature_cols = [c for c in train_df.columns if c.startswith('norm_')]

X_train = train_df[feature_cols].values
y_train = train_df['label'].values.astype(int)
train_groups = train_df.groupby('query_id', sort=False).size().to_numpy()

X_dev = dev_df[feature_cols].values
dev_groups = dev_df.groupby('query_id', sort=False).size().to_numpy()

for obj in ['rank:ndcg', 'rank:pairwise']:
    for lr in [0.05, 0.1]:
        ranker = xgb.XGBRanker(
            objective=obj,
            eval_metric='ndcg@10',
            n_estimators=100,
            learning_rate=lr,
            max_depth=4,
            random_state=42
        )
        ranker.fit(
            X_train, y_train,
            group=train_groups,
            eval_set=[(X_dev, dev_df['label'].values.astype(int))],
            eval_group=[dev_groups],
            verbose=False
        )
        dev_df['xgb_score'] = ranker.predict(X_dev)
        m = HybridEvaluator.evaluate_predictions(dev_df, score_col='xgb_score')
        print(f"XGB ({obj}, lr={lr}): nDCG@10={m['nDCG@10']:.4f}, MRR={m['MRR']:.4f}, P@1={m['P@1']:.4f}, R@10={m['R@10']:.4f}")

