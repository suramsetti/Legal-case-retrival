import pandas as pd

test_df = pd.read_parquet('data/processed/IL-TUR-PCR/test/hybrid_features.parquet')
pos = test_df[test_df['label'] == 1]
neg = test_df[test_df['label'] == 0]

cols = [
    'tfidf_score', 'bm25_score', 'inlegalbert_sim', 'composite_legal_score',
    'statute_shared_count', 'statute_jaccard', 'statute_containment_query',
    'citation_shared_count', 'citation_jaccard',
    'entity_shared_count', 'entity_jaccard',
    'judge_overlap', 'court_overlap', 'org_overlap',
    'rhetorical_cosine', 'rhetorical_ratio_diff'
]

print(f"{'Feature':30s} | {'Pos Mean':>10s} | {'Neg Mean':>10s} | {'Pos Median':>10s} | {'Neg Median':>10s}")
print('-' * 76)
for c in cols:
    print(f"{c:30s} | {pos[c].mean():10.4f} | {neg[c].mean():10.4f} | {pos[c].median():10.4f} | {neg[c].median():10.4f}")

