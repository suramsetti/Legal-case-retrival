import sys
sys.path.insert(0, ".")
import pandas as pd
import numpy as np

test_df = pd.read_parquet('data/processed/IL-TUR-PCR/test/hybrid_features.parquet')
for col in ['tfidf_score', 'inlegalbert_sim', 'composite_legal_score', 'bm25_score']:
    s_min = test_df.groupby('query_id')[col].transform('min')
    s_max = test_df.groupby('query_id')[col].transform('max')
    diff = s_max - s_min
    test_df['qnorm_' + col] = (test_df[col] - s_min) / np.where(diff > 1e-9, diff, 1.0)

test_df['stage1_score'] = 0.65 * test_df['qnorm_tfidf_score'] + 0.25 * test_df['qnorm_inlegalbert_sim'] + 0.10 * test_df['qnorm_composite_legal_score']
test_df['rank'] = test_df.groupby('query_id')['stage1_score'].rank(ascending=False, method='first')

target_queries = []
for qid, grp in test_df.groupby('query_id'):
    r1 = grp[grp['rank'] == 1].iloc[0]
    r2 = grp[grp['rank'] == 2].iloc[0]
    margin = r1['stage1_score'] - r2['stage1_score']
    if margin < 0.04 and r2['label'] == 1 and r1['label'] == 0:
        target_queries.append(qid)

print(f"Low margin queries where R1 is FALSE and R2 is TRUE: {len(target_queries)}")

# Let's inspect features comparing R2 (True) vs R1 (False) in these queries
cols = [
    'tfidf_score', 'bm25_score', 'inlegalbert_sim', 'composite_legal_score',
    'statute_shared_count', 'statute_containment_query', 'citation_shared_count',
    'entity_shared_count', 'judge_overlap', 'court_overlap', 'rhetorical_cosine'
]

diff_df = []
for qid in target_queries:
    grp = test_df[test_df['query_id'] == qid]
    r1 = grp[grp['rank'] == 1].iloc[0]
    r2 = grp[grp['rank'] == 2].iloc[0]
    row = {'qid': qid}
    for c in cols:
        row[c + '_diff'] = r2[c] - r1[c]
    diff_df.append(row)

diff_df = pd.DataFrame(diff_df)
print("\nMean difference (R2_True - R1_False):")
for c in cols:
    mean_d = diff_df[c + '_diff'].mean()
    pos_c = (diff_df[c + '_diff'] > 0).sum()
    neg_c = (diff_df[c + '_diff'] < 0).sum()
    eq_c = (diff_df[c + '_diff'] == 0).sum()
    print(f"  {c:30s}: mean={mean_d:+.4f} (R2 > R1 in {pos_c:2d}, R2 < R1 in {neg_c:2d}, equal in {eq_c:2d})")

