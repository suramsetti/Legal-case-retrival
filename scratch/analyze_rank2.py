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

target_queries = []
for qid, grp in test_df.groupby('query_id'):
    pos = grp[grp['label'] == 1]
    if not pos.empty and pos['rank'].min() == 2:
        target_queries.append(qid)

print(f"Total Rank 2 queries: {len(target_queries)}")

# Analyze all 33 queries:
# In how many does True Precedent have higher:
# - inlegalbert_sim?
# - bm25_score?
# - statute_containment_query?
# - statute_shared_count?
# - citation_shared_count?
# - rhetorical_cosine?
# - entity_shared_count?
# - court_overlap?
# - judge_overlap?

counts = {
    'inlegalbert_higher': 0,
    'bm25_higher': 0,
    'statute_higher': 0,
    'citation_higher': 0,
    'entity_higher': 0,
    'rhetorical_higher': 0,
    'court_higher': 0,
    'judge_higher': 0,
    'equal_statute': 0,
    'equal_citation': 0,
}

for qid in target_queries:
    grp = test_df[test_df['query_id'] == qid]
    r1 = grp[grp['rank'] == 1].iloc[0]
    r2 = grp[grp['label'] == 1].sort_values('rank').iloc[0]
    
    if r2['inlegalbert_sim'] > r1['inlegalbert_sim']:
        counts['inlegalbert_higher'] += 1
    if r2['bm25_score'] > r1['bm25_score']:
        counts['bm25_higher'] += 1
    if r2['statute_shared_count'] > r1['statute_shared_count']:
        counts['statute_higher'] += 1
    elif r2['statute_shared_count'] == r1['statute_shared_count']:
        counts['equal_statute'] += 1
    if r2['citation_shared_count'] > r1['citation_shared_count']:
        counts['citation_higher'] += 1
    elif r2['citation_shared_count'] == r1['citation_shared_count']:
        counts['equal_citation'] += 1
    if r2['entity_shared_count'] > r1['entity_shared_count']:
        counts['entity_higher'] += 1
    if r2['rhetorical_cosine'] > r1['rhetorical_cosine']:
        counts['rhetorical_higher'] += 1
    if r2['court_overlap'] > r1['court_overlap']:
        counts['court_higher'] += 1
    if r2['judge_overlap'] > r1['judge_overlap']:
        counts['judge_higher'] += 1

print("\nFor the 33 Rank-2 queries, True Precedent beats Rank-1 in:")
for k, v in counts.items():
    print(f"  {k:25s}: {v}/33 ({v/33*100:.1f}%)")

