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

c_docs = pd.read_parquet('data/processed/IL-TUR-PCR/test_candidates/documents.parquet').set_index('id')

r1_lengths = []
r2_lengths = []

for qid, grp in test_df.groupby('query_id'):
    pos = grp[grp['label'] == 1]
    if not pos.empty and pos['rank'].min() == 2:
        r1_id = grp[grp['rank'] == 1]['candidate_id'].iloc[0]
        r2_id = pos[pos['rank'] == 2]['candidate_id'].iloc[0]
        r1_lengths.append(c_docs.loc[r1_id, 'word_count'])
        r2_lengths.append(c_docs.loc[r2_id, 'word_count'])

s1 = pd.Series(r1_lengths)
s2 = pd.Series(r2_lengths)

print("Rank 1 (False Alarm) word counts:")
print(s1.describe())
print("\nRank 2 (True Precedent) word counts:")
print(s2.describe())
print(f"\nIn how many queries is Rank 1 longer than Rank 2? {(s1 > s2).sum()}/33 ({(s1 > s2).sum()/33*100:.1f}%)")

