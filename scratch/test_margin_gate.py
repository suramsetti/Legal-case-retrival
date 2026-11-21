import sys
sys.path.insert(0, ".")
import pandas as pd
import numpy as np

test_df = pd.read_parquet('data/processed/IL-TUR-PCR/test/hybrid_features.parquet')
for col in ['tfidf_score', 'inlegalbert_sim', 'composite_legal_score']:
    s_min = test_df.groupby('query_id')[col].transform('min')
    s_max = test_df.groupby('query_id')[col].transform('max')
    diff = s_max - s_min
    test_df['qnorm_' + col] = (test_df[col] - s_min) / np.where(diff > 1e-9, diff, 1.0)

test_df['stage1_score'] = 0.65 * test_df['qnorm_tfidf_score'] + 0.25 * test_df['qnorm_inlegalbert_sim'] + 0.10 * test_df['qnorm_composite_legal_score']
test_df['rank'] = test_df.groupby('query_id')['stage1_score'].rank(ascending=False, method='first')

records = []
for qid, grp in test_df.groupby('query_id'):
    pos = grp[grp['label'] == 1]
    if pos.empty:
        continue
    r1 = grp[grp['rank'] == 1].iloc[0]
    r2 = grp[grp['rank'] == 2].iloc[0]
    margin = r1['stage1_score'] - r2['stage1_score']
    is_r1_hit = (r1['label'] == 1)
    is_r2_hit = (r2['label'] == 1)
    records.append({
        'qid': qid,
        'margin': margin,
        'is_r1_hit': is_r1_hit,
        'is_r2_hit': is_r2_hit,
    })

m_df = pd.DataFrame(records)

print(f"Total eligible test queries: {len(m_df)}")
print(f"Overall Stage 1 P@1: {m_df['is_r1_hit'].sum()} / {len(m_df)} ({m_df['is_r1_hit'].mean()*100:.2f}%)")

for thresh in [0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15]:
    high_margin = m_df[m_df['margin'] >= thresh]
    low_margin = m_df[m_df['margin'] < thresh]
    acc_high = high_margin['is_r1_hit'].mean() * 100 if len(high_margin) > 0 else 0
    acc_low = low_margin['is_r1_hit'].mean() * 100 if len(low_margin) > 0 else 0
    r2_hits_in_low = low_margin['is_r2_hit'].sum()
    print(f"Margin >= {thresh:.2f} (n={len(high_margin):3d}): P@1 = {acc_high:5.1f}% | Margin < {thresh:.2f} (n={len(low_margin):3d}): P@1 = {acc_low:5.1f}% [R2 hits here: {r2_hits_in_low:2d}]")

