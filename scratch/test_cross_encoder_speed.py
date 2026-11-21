import time
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

t0 = time.time()
tok = AutoTokenizer.from_pretrained('law-ai/InLegalBERT', local_files_only=True)
base = AutoModel.from_pretrained('law-ai/InLegalBERT', local_files_only=True)

# Freeze lower 10 layers
for name, param in base.named_parameters():
    if "encoder.layer.10" not in name and "encoder.layer.11" not in name and "pooler" not in name:
        param.requires_grad = False

head = nn.Linear(768, 1)

text_a = "Civil Appeal against High Court order under Section 302 IPC regarding dowry death."
text_b = "Held: conviction under Section 304B IPC upheld. Evidence proves presumption under Section 113B Evidence Act."

inputs = tok(text_a, text_b, max_length=128, padding='max_length', truncation=True, return_tensors='pt')
optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, list(base.parameters()) + list(head.parameters())), lr=2e-5)

t1 = time.time()
for step in range(5):
    optimizer.zero_grad()
    out = base(**inputs)
    cls_rep = out.last_hidden_state[:, 0, :]
    score = head(cls_rep)
    loss = score.sum()
    loss.backward()
    optimizer.step()

t2 = time.time()
print(f"5 forward-backward steps took {t2 - t1:.2f}s ({(t2 - t1)/5:.3f}s per step)")

