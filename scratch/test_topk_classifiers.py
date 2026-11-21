import sys
sys.path.insert(0, ".")
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
import lightgbm as lgb
from src.evaluation.hybrid_evaluator import HybridEvaluator

data_dir = "data/processed/IL-TUR-PCR"
train_df = pd.read_parquet(f"{data_dir}/train/hybrid_features.parquet")
dev_df = pd.read_parquet(f"{data_dir}/dev/hybrid_features.parquet")
test_df = pd.read_parquet(f"{data_dir}/test/hybrid_features.parquet")

for df in [train_df, dev_df, test_df]:
    for col in ['tfidf_score', 'inlegalbert_sim', 'composite_legal_score', 'bm25_score']:
        s_min = df.groupby('query_id')[col].transform('min')
        s_max = df.groupby('query_id')[col].transform('max')
        diff = s_max - s_min
        df['qn_' + col] = (df[col] - s_min) / np.where(diff > 1e-9, diff, 1.0)

    df['stage1_score'] = 0.65 * df['qn_tfidf_score'] + 0.25 * df['qn_inlegalbert_sim'] + 0.10 * df['qn_composite_legal_score']
    df['s1_rank'] = df.groupby('query_id')['stage1_score'].rank(ascending=False, method='first')
    
    # Differential from top 1
    top1 = df.groupby('query_id')['stage1_score'].transform('max')
    top2 = df.groupby('query_id')['stage1_score'].transform(lambda s: s.nlargest(2).iloc[-1] if len(s) > 1 else s.iloc[0])
    df['diff_from_top1'] = top1 - df['stage1_score']
    df['margin_top1_top2'] = top1 - top2

    # Relative features
    df['dense_to_lex_ratio'] = df['qn_inlegalbert_sim'] / (df['qn_tfidf_score'] + 0.05)
    df['bm25_to_tfidf_ratio'] = df['qn_bm25_score'] / (df['qn_tfidf_score'] + 0.05)
    df['stat_x_dense'] = df['statute_containment_query'] * df['qn_inlegalbert_sim']
    df['stat_x_rhet'] = df['statute_shared_count'] * df['rhetorical_cosine']
    df['cit_boost'] = np.minimum(df['citation_shared_count'], 3.0)
    df['authority_score'] = 2.0 * df['statute_shared_count'] + 4.0 * df['cit_boost'] + 1.5 * df['judge_overlap']

features = [
    'stage1_score', 'diff_from_top1',
    'qn_tfidf_score', 'qn_inlegalbert_sim', 'qn_composite_legal_score', 'qn_bm25_score',
    'statute_shared_count', 'statute_containment_query', 'citation_shared_count',
    'entity_shared_count', 'judge_overlap', 'court_overlap',
    'rhetorical_cosine', 'rhetorical_ratio_diff',
    'dense_to_lex_ratio', 'bm25_to_tfidf_ratio', 'stat_x_dense', 'stat_x_rhet', 'authority_score'
]

# Baseline evaluation
m_s1_dev = HybridEvaluator.evaluate_predictions(dev_df, score_col='stage1_score')
m_s1_test = HybridEvaluator.evaluate_predictions(test_df, score_col='stage1_score')
print(f"Stage 1 Dev:  P@1={m_s1_dev['P@1']:.4f}, MRR={m_s1_dev['MRR']:.4f}")
print(f"Stage 1 Test: P@1={m_s1_test['P@1']:.4f}, MRR={m_s1_test['MRR']:.4f}")

# Filter to Top-3 candidates
TOP_K = 3
train_top = train_df[train_df['s1_rank'] <= TOP_K].copy()
dev_top = dev_df[dev_df['s1_rank'] <= TOP_K].copy()
test_top = test_df[test_df['s1_rank'] <= TOP_K].copy()

print(f"\nTop-{TOP_K} candidate pool: Train={len(train_top)}, Dev={len(dev_top)}, Test={len(test_top)}")

# Train classifiers
classifiers = {
    'Logistic Regression': LogisticRegression(C=1.0, max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_leaf=5, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=80, learning_rate=0.05, max_depth=3, random_state=42),
}

for name, clf in classifiers.items():
    clf.fit(train_top[features], train_top['label'])
    
    # Predict probabilities
    dev_probs = clf.predict_proba(dev_top[features])[:, 1]
    test_probs = clf.predict_proba(test_top[features])[:, 1]
    
    # Evaluate blended re-ranking
    for beta in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        dev_eval = dev_df.copy()
        dev_eval.loc[dev_eval['s1_rank'] <= TOP_K, 'clf_score'] = dev_probs
        dev_eval['clf_score'] = dev_eval['clf_score'].fillna(-999.0)
        
        # In Top-K, blend stage1 and clf
        dev_eval['final_score'] = np.where(
            dev_eval['s1_rank'] <= TOP_K,
            (1 - beta) * dev_eval['stage1_score'] + beta * dev_eval['clf_score'],
            dev_eval['stage1_score'] - 100.0
        )
        
        m_dev = HybridEvaluator.evaluate_predictions(dev_eval, score_col='final_score')
        if m_dev['P@1'] >= m_s1_dev['P@1']:
            test_eval = test_df.copy()
            test_eval.loc[test_eval['s1_rank'] <= TOP_K, 'clf_score'] = test_probs
            test_eval['clf_score'] = test_eval['clf_score'].fillna(-999.0)
            test_eval['final_score'] = np.where(
                test_eval['s1_rank'] <= TOP_K,
                (1 - beta) * test_eval['stage1_score'] + beta * test_eval['clf_score'],
                test_eval['stage1_score'] - 100.0
            )
            m_test = HybridEvaluator.evaluate_predictions(test_eval, score_col='final_score')
            print(f"[{name}] beta={beta:.1f} -> Dev P@1={m_dev['P@1']:.4f}, MRR={m_dev['MRR']:.4f} | Test P@1={m_test['P@1']:.4f}, MRR={m_test['MRR']:.4f}")

