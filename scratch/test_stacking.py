import sys
sys.path.append('.')
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression, RidgeClassifier, Ridge
from src.evaluation.hybrid_evaluator import HybridEvaluator

train_df = pd.read_parquet('data/processed/IL-TUR-PCR/train/hybrid_features.parquet').sort_values('query_id')
dev_df = pd.read_parquet('data/processed/IL-TUR-PCR/dev/hybrid_features.parquet').sort_values('query_id')

lexical_cols = ['norm_bm25_score', 'norm_bm25_rank', 'norm_tfidf_score', 'norm_tfidf_rank']
dense_cols = ['norm_inlegalbert_sim', 'norm_inlegalbert_rank']
legal_cols = [c for c in train_df.columns if c.startswith('norm_') and c not in lexical_cols and c not in dense_cols]

print(f"Lexical cols ({len(lexical_cols)}): {lexical_cols}")
print(f"Dense cols ({len(dense_cols)}): {dense_cols}")
print(f"Legal cols ({len(legal_cols)}): {legal_cols}")

# Set up 5-Fold GroupKFold
gkf = GroupKFold(n_splits=5)
groups = train_df['query_id'].values

oof_preds = np.zeros((len(train_df), 3)) # col 0: lex, col 1: dense, col 2: legal
dev_preds = np.zeros((len(dev_df), 3))

y_train = train_df['label'].values.astype(int)

for fold, (trn_idx, val_idx) in enumerate(gkf.split(train_df, y_train, groups=groups)):
    # Lexical base model
    m_lex = LogisticRegression(class_weight='balanced', max_iter=1000)
    m_lex.fit(train_df.iloc[trn_idx][lexical_cols], y_train[trn_idx])
    oof_preds[val_idx, 0] = m_lex.predict_proba(train_df.iloc[val_idx][lexical_cols])[:, 1]
    dev_preds[:, 0] += m_lex.predict_proba(dev_df[lexical_cols])[:, 1] / 5.0

    # Dense base model
    m_dense = LogisticRegression(class_weight='balanced', max_iter=1000)
    m_dense.fit(train_df.iloc[trn_idx][dense_cols], y_train[trn_idx])
    oof_preds[val_idx, 1] = m_dense.predict_proba(train_df.iloc[val_idx][dense_cols])[:, 1]
    dev_preds[:, 1] += m_dense.predict_proba(dev_df[dense_cols])[:, 1] / 5.0

    # Legal base model
    m_legal = LogisticRegression(class_weight='balanced', max_iter=1000)
    m_legal.fit(train_df.iloc[trn_idx][legal_cols], y_train[trn_idx])
    oof_preds[val_idx, 2] = m_legal.predict_proba(train_df.iloc[val_idx][legal_cols])[:, 1]
    dev_preds[:, 2] += m_legal.predict_proba(dev_df[legal_cols])[:, 1] / 5.0

# Fit Meta-Learner on OOF predictions
meta_model = LogisticRegression(penalty=None, max_iter=1000)
meta_model.fit(oof_preds, y_train)

print(f"Meta-model coefficients: {meta_model.coef_[0]}, intercept: {meta_model.intercept_[0]}")

dev_df['stacking_score'] = meta_model.predict_proba(dev_preds)[:, 1]
metrics = HybridEvaluator.evaluate_predictions(dev_df, score_col='stacking_score')
print("Stacking Dev Metrics:", metrics)

