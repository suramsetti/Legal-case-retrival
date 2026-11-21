import sys
from pathlib import Path
PROJECT_ROOT = Path("c:/Users/mohan/OneDrive/Desktop/Projects/legal-case-retrieval")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

# Load all-MiniLM-L6-v2
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2', local_files_only=True)
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2', local_files_only=True)
model.eval()

def encode_texts(texts):
    encoded = tokenizer(texts, padding=True, truncation=True, max_length=256, return_tensors='pt')
    with torch.no_grad():
        out = model(**encoded)
        # Mean pooling with attention mask
        attn = encoded['attention_mask'].unsqueeze(-1)
        emb = (out.last_hidden_state * attn).sum(dim=1) / torch.clamp(attn.sum(dim=1), min=1e-9)
        emb = torch.nn.functional.normalize(emb, p=2, dim=1)
    return emb.numpy()

# Load test data
test_df = pd.read_parquet(data_dir := PROJECT_ROOT / "data/processed/IL-TUR-PCR/test/hybrid_features.parquet")
for col in ['tfidf_score', 'inlegalbert_sim', 'composite_legal_score']:
    s_min = test_df.groupby('query_id')[col].transform('min')
    s_max = test_df.groupby('query_id')[col].transform('max')
    diff = s_max - s_min
    test_df['qnorm_' + col] = (test_df[col] - s_min) / np.where(diff > 1e-9, diff, 1.0)

test_df['stage1_score'] = 0.65 * test_df['qnorm_tfidf_score'] + 0.25 * test_df['qnorm_inlegalbert_sim'] + 0.10 * test_df['qnorm_composite_legal_score']
test_df['rank'] = test_df.groupby('query_id')['stage1_score'].rank(ascending=False, method='first')

q_docs = pd.read_parquet(PROJECT_ROOT / 'data/processed/IL-TUR-PCR/test_queries/documents.parquet').set_index('id')
c_docs = pd.read_parquet(PROJECT_ROOT / 'data/processed/IL-TUR-PCR/test_candidates/documents.parquet').set_index('id')

# Test on 10 Rank-2 queries
rank2_qids = []
for qid, grp in test_df.groupby('query_id'):
    pos = grp[grp['label'] == 1]
    if not pos.empty and pos['rank'].min() == 2:
        rank2_qids.append(qid)

print(f"Testing MiniLM on first 10 Rank-2 queries...")
true_beat_false = 0
for qid in rank2_qids[:10]:
    grp = test_df[test_df['query_id'] == qid]
    r1_id = grp[grp['rank'] == 1]['candidate_id'].iloc[0]
    r2_id = grp[grp['label'] == 1].sort_values('rank')['candidate_id'].iloc[0]
    
    q_text = q_docs.loc[qid, 'full_text'][:1000]
    c1_text = c_docs.loc[r1_id, 'full_text'][:1000]
    c2_text = c_docs.loc[r2_id, 'full_text'][:1000]
    
    embs = encode_texts([q_text, c1_text, c2_text])
    sim_r1 = float(np.dot(embs[0], embs[1]))
    sim_r2 = float(np.dot(embs[0], embs[2]))
    
    winner = "TRUE (R2)" if sim_r2 > sim_r1 else "FALSE (R1)"
    if sim_r2 > sim_r1:
        true_beat_false += 1
    print(f"Query {qid}: sim(Q, R1_false)={sim_r1:.4f}, sim(Q, R2_true)={sim_r2:.4f} -> Winner: {winner}")

print(f"\nResult: True precedent beat False alarm in {true_beat_false}/10 cases!")

