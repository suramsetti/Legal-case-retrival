import sys
sys.path.insert(0, ".")
import pandas as pd
import numpy as np

for split in ['train', 'dev', 'test']:
    df = pd.read_parquet(f'data/processed/IL-TUR-PCR/{split}/hybrid_features.parquet')
    for col in ['tfidf_score', 'inlegalbert_sim', 'composite_legal_score']:
        s_min = df.groupby('query_id')[col].transform('min')
        s_max = df.groupby('query_id')[col].transform('max')
        diff = s_max - s_min
        df['qnorm_' + col] = (df[col] - s_min) / np.where(diff > 1e-9, diff, 1.0)

    df['stage1_score'] = 0.65 * df['qnorm_tfidf_score'] + 0.25 * df['qnorm_inlegalbert_sim'] + 0.10 * df['qnorm_composite_legal_score']
    df['rank'] = df.groupby('query_id')['stage1_score'].rank(ascending=False, method='first')

    # Count how many queries have margin < 0.05
    margins = []
    r1_correct = 0
    r2_correct = 0
    both_correct = 0
    neither_correct = 0
    
    total = 0
    for qid, grp in df.groupby('query_id'):
        pos = grp[grp['label'] == 1]
        if pos.empty:
            continue
        total += 1
        r1 = grp[grp['rank'] == 1].iloc[0]
        r2 = grp[grp['rank'] == 2].iloc[0]
        m = r1['stage1_score'] - r2['stage1_score']
        margins.append(m)
        
        c1 = (r1['label'] == 1)
        c2 = (r2['label'] == 1)
        if c1 and c2:
            both_correct += 1
        elif c1:
            r1_correct += 1
        elif c2:
            r2_correct += 1
        else:
            neither_correct += 1
            
    print(f"=== Split: {split.upper()} ({total} queries) ===")
    print(f"R1 correct: {r1_correct} ({r1_correct/total*100:.1f}%) | R2 correct (R1 wrong): {r2_correct} ({r2_correct/total*100:.1f}%) | Both: {both_correct} | Neither: {neither_correct}")
    
    # In queries with margin < 0.05:
    m_series = pd.Series(margins)
    print(f"Queries with margin < 0.05: {(m_series < 0.05).sum()}/ {total}")

