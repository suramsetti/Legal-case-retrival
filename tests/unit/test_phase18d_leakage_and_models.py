"""
Phase 18D Unit Tests: Leakage Remediation & Transformer Model Architecture.

Tests:
1. Leakage remediation functions (exact masking, normalized masking, concluding truncation).
2. Preservation of legal facts and reasoning (ensures body is not over-masked).
3. All 4 Transformer architectures (instantiation, forward pass, backward pass, shapes).
4. Attention pooling mathematical correctness (weights sum to 1.0).
5. Baseline model serialization and inference.
6. Clean partition verification (ensures masked parquet files have <= 0.1% leakage).
"""

import sys
from pathlib import Path
import pytest
import torch
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.phase18d_leakage_remediation import remediate_single_text, AUDIT_PATTERN
from src.models.phase18d_transformer_models import (
    InLegalBERTSequenceClassifier,
    InLegalBERTBiLSTM,
    InLegalBERTAttentionPooling,
    InLegalBERTBiLSTMAttention,
    AttentionPooling
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "judipl" / "phase18c"


class TestLeakageRemediation:

    def test_masks_explicit_ruling_phrase(self):
        text = "The appellant challenged the conviction under Section 302. The appeal is allowed."
        proof = ""
        remediated = remediate_single_text(text, proof)
        assert "[MASKED_RULING]" in remediated or "[MASKED_CONCLUDING_DISPOSITION]" in remediated
        assert not AUDIT_PATTERN.search(remediated), f"Audit pattern still matched: {remediated}"

    def test_masks_dismissal_phrase(self):
        text = "Having heard the counsel, we find no error in the High Court order. The appeal is dismissed with costs."
        remediated = remediate_single_text(text, "")
        assert not AUDIT_PATTERN.search(remediated)

    def test_masks_proof_sentence_exact(self):
        proof = "34. we are, therefore, of the view that this appeal must be allowed and the judgment of the high court is set aside."
        text = f"The matter was argued before the bench. {proof} There will be no order as to costs."
        remediated = remediate_single_text(text, proof)
        assert "[MASKED_HOLDING]" in remediated
        assert not AUDIT_PATTERN.search(remediated)

    def test_masks_proof_sentence_normalized(self):
        proof = "34.  we are of the view that this appeal must be allowed."
        # Slightly different whitespace in text
        text = "The matter was argued. 34. we are of the view that this appeal must be allowed. Ordered accordingly."
        remediated = remediate_single_text(text, proof)
        assert "[MASKED_HOLDING]" in remediated or "[MASKED_RULING]" in remediated
        assert not AUDIT_PATTERN.search(remediated)

    def test_concluding_disposition_truncation(self):
        body = "This is the factual background. The appellant was appointed in 1985. " * 30
        conclusion = "In the result, the appeals are allowed and the order of the tribunal is quashed. No costs."
        full = body + conclusion
        remediated = remediate_single_text(full, "")
        assert "[MASKED_CONCLUDING_DISPOSITION]" in remediated or "[MASKED_RULING]" in remediated
        assert not AUDIT_PATTERN.search(remediated)

    def test_preserves_facts_and_procedural_history(self):
        facts = "The appellant filed a claim petition under Section 166 of the Motor Vehicles Act, 1988 claiming compensation."
        ruling = "In the result, the appeal is allowed."
        full = (facts + " ") * 20 + ruling
        remediated = remediate_single_text(full, "")
        # Facts must be preserved
        assert "Motor Vehicles Act" in remediated
        assert "Section 166" in remediated


class TestTransformerArchitectures:

    def test_attention_pooling_weights_sum_to_one(self):
        pool = AttentionPooling(input_dim=64, hidden_dim=32)
        x = torch.randn(2, 5, 64)
        pooled, weights = pool(x)
        assert pooled.shape == (2, 64)
        assert weights.shape == (2, 5, 1)
        # Weights along seq_len must sum to 1.0
        sums = weights.sum(dim=1).squeeze(-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    def test_sequence_classifier_forward(self):
        model = InLegalBERTSequenceClassifier(num_classes=2, model_name_or_path=None, hidden_dim=64)
        x_ids = torch.randint(100, 1000, (2, 3, 32))
        x_mask = torch.ones((2, 3, 32), dtype=torch.long)
        logits = model(x_ids, x_mask)
        assert logits.shape == (2, 2)

    def test_bilstm_forward(self):
        model = InLegalBERTBiLSTM(num_classes=2, model_name_or_path=None, hidden_dim=64, lstm_hidden=32)
        x_ids = torch.randint(100, 1000, (2, 3, 32))
        x_mask = torch.ones((2, 3, 32), dtype=torch.long)
        logits = model(x_ids, x_mask)
        assert logits.shape == (2, 2)

    def test_attention_pooling_forward(self):
        model = InLegalBERTAttentionPooling(num_classes=2, model_name_or_path=None, hidden_dim=64)
        x_ids = torch.randint(100, 1000, (2, 3, 32))
        x_mask = torch.ones((2, 3, 32), dtype=torch.long)
        logits = model(x_ids, x_mask)
        assert logits.shape == (2, 2)

    def test_bilstm_attention_forward(self):
        model = InLegalBERTBiLSTMAttention(num_classes=2, model_name_or_path=None, hidden_dim=64, lstm_hidden=32)
        x_ids = torch.randint(100, 1000, (2, 3, 32))
        x_mask = torch.ones((2, 3, 32), dtype=torch.long)
        logits = model(x_ids, x_mask)
        assert logits.shape == (2, 2)

    def test_gradient_backward_pass(self):
        model = InLegalBERTBiLSTMAttention(num_classes=2, model_name_or_path=None, hidden_dim=64, lstm_hidden=32)
        x_ids = torch.randint(100, 1000, (2, 3, 32))
        x_mask = torch.ones((2, 3, 32), dtype=torch.long)
        targets = torch.tensor([0, 1], dtype=torch.long)
        logits = model(x_ids, x_mask)
        loss = torch.nn.CrossEntropyLoss()(logits, targets)
        loss.backward()
        for p in model.parameters():
            if p.requires_grad and p.grad is not None:
                assert not torch.isnan(p.grad).any()
                return
        pytest.fail("No valid gradients found")


class TestPartitionLeakageVerification:

    def test_binary_train_masked_leakage_rate(self):
        path = OUTPUT_DIR / "binary" / "train_masked.parquet"
        if not path.exists():
            pytest.skip("train_masked.parquet not yet written")
        df = pd.read_parquet(path)
        has_ruling = df["masked_text"].str.contains(AUDIT_PATTERN, regex=True, na=False)
        ruling_pct = has_ruling.mean() * 100
        # Leakage must be <= 0.1%
        assert ruling_pct <= 0.1, f"Expected <=0.1% ruling leakage, got {ruling_pct:.2f}%"

    def test_binary_val_masked_leakage_rate(self):
        path = OUTPUT_DIR / "binary" / "val_masked.parquet"
        if not path.exists():
            pytest.skip("val_masked.parquet not yet written")
        df = pd.read_parquet(path)
        has_ruling = df["masked_text"].str.contains(AUDIT_PATTERN, regex=True, na=False)
        ruling_pct = has_ruling.mean() * 100
        assert ruling_pct <= 0.1, f"Expected <=0.1% ruling leakage in val, got {ruling_pct:.2f}%"

