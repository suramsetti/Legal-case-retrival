import sys
sys.path.insert(0, ".")
import time
import numpy as np
import torch
from src.ranking.legal_cross_encoder import InLegalBERTCrossEncoder

ce = InLegalBERTCrossEncoder(model_name_or_path="law-ai/InLegalBERT", device="cpu")
q_list = ["Civil Appeal against High Court order regarding property dispute and injunction."] * 32
c_list = ["Trial Court dismissed suit for declaration of title. Held: possession must follow title."] * 32

t0 = time.time()
scores = ce.score_pairs(q_list, c_list, batch_size=32, max_length=128)
t1 = time.time()
print(f"Scored 32 pairs in {t1 - t0:.2f}s ({(t1 - t0)/32:.3f}s per pair = {32/(t1 - t0):.1f} pairs/sec)")
