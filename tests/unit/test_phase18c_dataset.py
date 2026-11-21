"""
Unit tests for Phase 18C dataset preparation and leakage validation.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "judipl" / "phase18c"


class TestDatasetPartitions:
    """Test dataset partition integrity."""

    def test_binary_partitions_exist(self):
        """Test that binary task partitions exist."""
        assert (DATA_DIR / "binary" / "train.parquet").exists()
        assert (DATA_DIR / "binary" / "val.parquet").exists()
        assert (DATA_DIR / "binary" / "test.parquet").exists()

    def test_three_class_partitions_exist(self):
        """Test that three-class task partitions exist."""
        assert (DATA_DIR / "three_class" / "train.parquet").exists()
        assert (DATA_DIR / "three_class" / "val.parquet").exists()
        assert (DATA_DIR / "three_class" / "test.parquet").exists()

    def test_holding_extraction_partitions_exist(self):
        """Test that holding extraction partitions exist."""
        assert (DATA_DIR / "holding_extraction" / "train.parquet").exists()
        assert (DATA_DIR / "holding_extraction" / "val.parquet").exists()
        assert (DATA_DIR / "holding_extraction" / "test.parquet").exists()

    def test_binary_partition_sizes(self):
        """Test binary partition sizes are reasonable."""
        train = pd.read_parquet(DATA_DIR / "binary" / "train.parquet")
        val = pd.read_parquet(DATA_DIR / "binary" / "val.parquet")
        test = pd.read_parquet(DATA_DIR / "binary" / "test.parquet")

        total = len(train) + len(val) + len(test)

        # Check total matches expected
        assert total == 35520  # Expected binary-labeled count

        # Check partition ratios are approximately 70/15/15
        assert 0.65 <= len(train) / total <= 0.75
        assert 0.10 <= len(val) / total <= 0.20
        assert 0.10 <= len(test) / total <= 0.20

    def test_three_class_partition_sizes(self):
        """Test three-class partition sizes are reasonable."""
        train = pd.read_parquet(DATA_DIR / "three_class" / "train.parquet")
        val = pd.read_parquet(DATA_DIR / "three_class" / "val.parquet")
        test = pd.read_parquet(DATA_DIR / "three_class" / "test.parquet")

        total = len(train) + len(val) + len(test)

        # Check total matches expected (Accepted + Rejected + Other)
        assert total == 42005  # Excluding 1 unlabeled

        # Check partition ratios are approximately 70/15/15
        assert 0.65 <= len(train) / total <= 0.75
        assert 0.10 <= len(val) / total <= 0.20
        assert 0.10 <= len(test) / total <= 0.20

    def test_binary_labels(self):
        """Test binary task only has Accepted and Rejected labels."""
        train = pd.read_parquet(DATA_DIR / "binary" / "train.parquet")
        val = pd.read_parquet(DATA_DIR / "binary" / "val.parquet")
        test = pd.read_parquet(DATA_DIR / "binary" / "test.parquet")

        for df in [train, val, test]:
            assert set(df['label'].unique()) <= {'Accepted', 'Rejected'}

    def test_three_class_labels(self):
        """Test three-class task has Accepted, Rejected, and Other labels."""
        train = pd.read_parquet(DATA_DIR / "three_class" / "train.parquet")
        val = pd.read_parquet(DATA_DIR / "three_class" / "val.parquet")
        test = pd.read_parquet(DATA_DIR / "three_class" / "test.parquet")

        for df in [train, val, test]:
            assert set(df['label'].unique()) <= {'Accepted', 'Rejected', 'Other'}


class TestLeakagePrevention:
    """Test leakage prevention mechanisms."""

    def test_no_duplicate_group_leakage_binary(self):
        """Test no duplicate groups cross partitions in binary task."""
        train = pd.read_parquet(DATA_DIR / "binary" / "train.parquet")
        val = pd.read_parquet(DATA_DIR / "binary" / "val.parquet")
        test = pd.read_parquet(DATA_DIR / "binary" / "test.parquet")

        train_groups = set(train['duplicate_group'].unique())
        val_groups = set(val['duplicate_group'].unique())
        test_groups = set(test['duplicate_group'].unique())

        # No overlaps
        assert len(train_groups & val_groups) == 0
        assert len(train_groups & test_groups) == 0
        assert len(val_groups & test_groups) == 0

    def test_no_duplicate_group_leakage_three_class(self):
        """Test no duplicate groups cross partitions in three-class task."""
        train = pd.read_parquet(DATA_DIR / "three_class" / "train.parquet")
        val = pd.read_parquet(DATA_DIR / "three_class" / "val.parquet")
        test = pd.read_parquet(DATA_DIR / "three_class" / "test.parquet")

        train_groups = set(train['duplicate_group'].unique())
        val_groups = set(val['duplicate_group'].unique())
        test_groups = set(test['duplicate_group'].unique())

        # No overlaps
        assert len(train_groups & val_groups) == 0
        assert len(train_groups & test_groups) == 0
        assert len(val_groups & test_groups) == 0

    def test_no_exact_text_leakage_binary(self):
        """Test no exact text duplicates cross partitions in binary task."""
        train = pd.read_parquet(DATA_DIR / "binary" / "train.parquet")
        val = pd.read_parquet(DATA_DIR / "binary" / "val.parquet")
        test = pd.read_parquet(DATA_DIR / "binary" / "test.parquet")

        train_texts = set(train['full_text'].unique())
        val_texts = set(val['full_text'].unique())
        test_texts = set(test['full_text'].unique())

        # No overlaps
        assert len(train_texts & val_texts) == 0
        assert len(train_texts & test_texts) == 0
        assert len(val_texts & test_texts) == 0

    def test_no_exact_text_leakage_three_class(self):
        """Test no exact text duplicates cross partitions in three-class task."""
        train = pd.read_parquet(DATA_DIR / "three_class" / "train.parquet")
        val = pd.read_parquet(DATA_DIR / "three_class" / "val.parquet")
        test = pd.read_parquet(DATA_DIR / "three_class" / "test.parquet")

        train_texts = set(train['full_text'].unique())
        val_texts = set(val['full_text'].unique())
        test_texts = set(test['full_text'].unique())

        # No overlaps
        assert len(train_texts & val_texts) == 0
        assert len(train_texts & test_texts) == 0
        assert len(val_texts & test_texts) == 0


class TestDatasetColumns:
    """Test dataset schema and columns."""

    def test_binary_columns(self):
        """Test binary dataset has required columns."""
        train = pd.read_parquet(DATA_DIR / "binary" / "train.parquet")

        required_columns = [
            'id', 'raw_id', 'raw_name', 'case_category', 'case_type', 'label',
            'petitioner', 'respondent', 'date_of_judgment', 'bench',
            'citation_meta', 'proof_sentence', 'char_length', 'word_length',
            'sentence_count', 'statute_count', 'citation_count', 'full_text',
            'source', 'duplicate_group', 'partition'
        ]

        for col in required_columns:
            assert col in train.columns

    def test_three_class_columns(self):
        """Test three-class dataset has required columns."""
        train = pd.read_parquet(DATA_DIR / "three_class" / "train.parquet")

        required_columns = [
            'id', 'raw_id', 'raw_name', 'case_category', 'case_type', 'label',
            'petitioner', 'respondent', 'date_of_judgment', 'bench',
            'citation_meta', 'proof_sentence', 'char_length', 'word_length',
            'sentence_count', 'statute_count', 'citation_count', 'full_text',
            'source', 'duplicate_group', 'partition'
        ]

        for col in required_columns:
            assert col in train.columns

    def test_holding_extraction_columns(self):
        """Test holding extraction dataset has required columns."""
        train = pd.read_parquet(DATA_DIR / "holding_extraction" / "train.parquet")

        required_columns = [
            'id', 'raw_id', 'label', 'partition', 'duplicate_group',
            'input_text', 'target_sentence', 'full_text', 'proof_sentence'
        ]

        for col in required_columns:
            assert col in train.columns


class TestLabelDistribution:
    """Test label distribution is reasonable."""

    def test_binary_label_balance(self):
        """Test binary labels are reasonably balanced."""
        train = pd.read_parquet(DATA_DIR / "binary" / "train.parquet")
        val = pd.read_parquet(DATA_DIR / "binary" / "val.parquet")
        test = pd.read_parquet(DATA_DIR / "binary" / "test.parquet")

        for df in [train, val, test]:
            accepted = (df['label'] == 'Accepted').sum()
            rejected = (df['label'] == 'Rejected').sum()
            total = len(df)

            # Both classes should be present
            assert accepted > 0
            assert rejected > 0

            # Neither class should be too dominant (> 80%)
            assert accepted / total < 0.8
            assert rejected / total < 0.8

    def test_three_class_label_balance(self):
        """Test three-class labels are reasonably distributed."""
        train = pd.read_parquet(DATA_DIR / "three_class" / "train.parquet")
        val = pd.read_parquet(DATA_DIR / "three_class" / "val.parquet")
        test = pd.read_parquet(DATA_DIR / "three_class" / "test.parquet")

        for df in [train, val, test]:
            accepted = (df['label'] == 'Accepted').sum()
            rejected = (df['label'] == 'Rejected').sum()
            other = (df['label'] == 'Other').sum()
            total = len(df)

            # All classes should be present
            assert accepted > 0
            assert rejected > 0
            assert other > 0

            # No class should be completely absent


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
