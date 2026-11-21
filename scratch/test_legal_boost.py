import sys
from pathlib import Path
PROJECT_ROOT = Path("c:/Users/mohan/OneDrive/Desktop/Projects/legal-case-retrieval")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
from src.evaluation.hybrid_evaluator import HybridEvaluator

data_dir = PROJECT_ROOT / "data/processed/IL-TUR-PCR"
test_df = pd.read_parquet(data_dir / "test/hybrid_features.parquet")
dev_df = pd.read_parquet(data_dir / "dev/hybrid_features.parquet")

for df in [dev_df, test_df]:
    for col in ['tfidf_score', 'inlegalbert_sim', 'composite_legal_score', 'bm25_score', 'statute_containment_query']:
        s_min = df.groupby('query_id')[col].transform('min')
        s_max = df.groupby('query_id')[col].transform('max')
        diff = s_max - s_min
        df['qn_' + col] = (df[col] - s_min) / np.where(diff > 1e-9, diff, 1.0)
    
    # Base Stage 1 score
    df['stage1_score'] = 0.65 * df['qn_tfidf_score'] + 0.25 * df['qn_inlegalbert_sim'] + 0.10 * df['qn_composite_legal_score']

# Evaluate dev baseline
m_dev_base = HybridEvaluator.evaluate_predictions(dev_df, score_col='stage1_score')
m_test_base = HybridEvaluator.evaluate_predictions(test_df, score_col='stage1_score')
print(f"Dev Base:  P@1={m_dev_base['P@1']:.4f}, MRR={m_dev_base['MRR']:.4f}")
print(f"Test Base: P@1={m_test_base['P@1']:.4f}, MRR={m_test_base['MRR']:.4f}")

# Test adding statutory containment and citation boosts to candidates in the top 10
for boost_stat in [0.0, 0.05, 0.10, 0.15, 0.20]:
    for boost_cit in [0.0, 0.05, 0.10, 0.15, 0.20]:
        dev_df['mod_score'] = dev_df['stage1_score'] + boost_stat * dev_df['qn_statute_containment_query'] + boost_cit * np.minimum(dev_df['citation_shared_count'], 2) * 0.1
        m = HybridEvaluator.evaluate_predictions(dev_df, score_col='mod_score')
        if m['P@1'] > m_dev_base['P@1']:
            print(f"Dev Boost stat={boost_stat:.2f}, cit={boost_cit:.2f} -> P@1={m['P@1']:.4f}, MRR={m['MRR']:.4f}")

