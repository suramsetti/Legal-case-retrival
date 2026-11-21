import sys
from pathlib import Path
PROJECT_ROOT = Path("c:/Users/mohan/OneDrive/Desktop/Projects/legal-case-retrieval")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import lightgbm as lgb
from src.evaluation.hybrid_evaluator import HybridEvaluator

data_dir = PROJECT_ROOT / "data/processed/IL-TUR-PCR"
train_df = pd.read_parquet(data_dir / "train/hybrid_features.parquet")
dev_df = pd.read_parquet(data_dir / "dev/hybrid_features.parquet")
test_df = pd.read_parquet(data_dir / "test/hybrid_features.parquet")

norm_cols = [
    'tfidf_score', 'inlegalbert_sim', 'composite_legal_score', 'bm25_score', 
    'statute_shared_count', 'citation_shared_count', 'entity_shared_count',
    'statute_containment_query', 'rhetorical_cosine', 'judge_overlap', 'court_overlap'
]

for df in [train_df, dev_df, test_df]:
    for col in norm_cols:
        s_min = df.groupby('query_id')[col].transform('min')
        s_max = df.groupby('query_id')[col].transform('max')
        diff = s_max - s_min
        df['qn_' + col] = (df[col] - s_min) / np.where(diff > 1e-9, diff, 1.0)
    
    df['stage1_score'] = 0.65 * df['qn_tfidf_score'] + 0.25 * df['qn_inlegalbert_sim'] + 0.10 * df['qn_composite_legal_score']
    df['s1_rank'] = df.groupby('query_id')['stage1_score'].rank(ascending=False, method='first')
    
    # Interaction terms
    df['lex_dense_prod'] = df['qn_tfidf_score'] * df['qn_inlegalbert_sim']
    df['lex_legal_prod'] = df['qn_tfidf_score'] * df['qn_composite_legal_score']
    df['statute_density'] = df['qn_statute_shared_count'] * (df['qn_tfidf_score'] + 0.1)
    df['authority_score'] = df['qn_statute_shared_count'] * 2.0 + df['qn_citation_shared_count'] * 3.0
    
    # Differential from query top 1
    top1 = df.groupby('query_id')['stage1_score'].transform('max')
    df['diff_from_top1'] = top1 - df['stage1_score']

features = [
    'stage1_score',
    'qn_tfidf_score', 'qn_inlegalbert_sim', 'qn_composite_legal_score', 'qn_bm25_score',
    'qn_statute_shared_count', 'qn_citation_shared_count', 'qn_entity_shared_count',
    'qn_statute_containment_query', 'qn_rhetorical_cosine', 'qn_judge_overlap', 'qn_court_overlap',
    'lex_dense_prod', 'lex_legal_prod', 'statute_density', 'authority_score', 'diff_from_top1'
]

# Filter train to Top-K per query plus positives
K = 25
train_top = train_df[(train_df['s1_rank'] <= K) | (train_df['label'] == 1)].copy()
train_top = train_top.sort_values('query_id')
train_groups = train_top.groupby('query_id', sort=False).size().values

dev_top = dev_df[(dev_df['s1_rank'] <= K) | (dev_df['label'] == 1)].copy()
dev_top = dev_top.sort_values('query_id')
dev_groups = dev_top.groupby('query_id', sort=False).size().values

print(f"Train rows: {len(train_top)}, Dev rows: {len(dev_top)}")

ranker = lgb.LGBMRanker(
    objective="lambdarank",
    metric="ndcg",
    eval_at=[1, 5],
    n_estimators=100,
    learning_rate=0.03,
    num_leaves=15,
    min_child_samples=10,
    random_state=42
)

ranker.fit(
    train_top[features],
    train_top['label'],
    group=train_groups,
    eval_set=[(dev_top[features], dev_top['label'])],
    eval_group=[dev_groups],
    callbacks=[lgb.early_stopping(stopping_rounds=15, verbose=False)]
)

# Test evaluation on FULL candidate pool
for name, df in [("Dev", dev_df), ("Test", test_df)]:
    df = df.copy()
    preds = ranker.predict(df[features])
    # Blend with stage 1 score for items outside top K
    df['lgb_pred'] = preds
    # Re-rank only top K
    df['final_score'] = np.where(df['s1_rank'] <= K, df['stage1_score'] * 0.4 + preds * 0.6, df['stage1_score'] - 1000.0)
    m = HybridEvaluator.evaluate_predictions(df, score_col='final_score')
    print(f"{name} with LambdaMART Re-ranking: P@1={m['P@1']:.4f}, MRR={m['MRR']:.4f}, nDCG@10={m['nDCG@10']:.4f}")
