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

count = 0
for qid, grp in test_df.groupby('query_id'):
    pos = grp[grp['label'] == 1]
    if pos.empty:
        continue
    best_rank = pos['rank'].min()
    if best_rank == 2:
        r1 = grp[grp['rank'] == 1].iloc[0]
        r2 = pos[pos['rank'] == 2].iloc[0]
        cid1 = r1['candidate_id']
        cid2 = r2['candidate_id']
        s1 = r1['stage1_score']
        s2 = r2['stage1_score']
        print(f"=== Query {qid} ===")
        print(f"R1 (False) [{cid1}] vs R2 (True) [{cid2}]")
        print(f"  Stage 1 Score:  {s1:.4f} vs {s2:.4f} (diff = {s1 - s2:.4f})")
        print(f"  TF-IDF Score:   {r1['tfidf_score']:.4f} vs {r2['tfidf_score']:.4f}")
        print(f"  BM25 Score:     {r1['bm25_score']:.1f} vs {r2['bm25_score']:.1f}")
        print(f"  InLegalBERT:    {r1['inlegalbert_sim']:.4f} vs {r2['inlegalbert_sim']:.4f}")
        print(f"  Composite Leg:  {r1['composite_legal_score']:.4f} vs {r2['composite_legal_score']:.4f}")
        print(f"  Statute Count:  {r1['statute_shared_count']} vs {r2['statute_shared_count']}")
        print(f"  Citation Count: {r1['citation_shared_count']} vs {r2['citation_shared_count']}")
        print(f"  Entity Count:   {r1['entity_shared_count']} vs {r2['entity_shared_count']}")
        print(f"  Rhetorical Cos: {r1['rhetorical_cosine']:.4f} vs {r2['rhetorical_cosine']:.4f}")
        print(f"  Court Overlap:  {r1['court_overlap']} vs {r2['court_overlap']}")
        count += 1
        if count >= 4:
            break

