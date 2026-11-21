"""
Negative Sampler for Learning-to-Rank Training Dataset Construction.

Implements deterministic hard negative mining and random negative sampling for
Prior Case Retrieval (PCR) queries:
- Preserves 100% of available positive query-candidate pairs.
- Mines hard negatives from top BM25 candidate ranks (lexical distractors).
- Samples uniform random negatives for broader semantic space regularization.
- Strictly enforces query self-exclusion (query_id != candidate_id).
- Uses deterministic random seeds for full reproducibility.
"""

from typing import Dict, List, Set, Sequence, Tuple, Any, Optional
import numpy as np


class NegativeSampler:
    """Samples hard negatives and uniform random negatives per query for ranking models."""

    def __init__(
        self,
        n_hard_negatives: int = 15,
        n_random_negatives: int = 5,
        random_seed: int = 42,
    ):
        """
        Args:
            n_hard_negatives: Number of top non-relevant candidates by lexical score to retain.
            n_random_negatives: Number of random non-relevant candidates to sample.
            random_seed: Deterministic random seed for numpy RNG.
        """
        self.n_hard_negatives = n_hard_negatives
        self.n_random_negatives = n_random_negatives
        self.random_seed = random_seed
        self.rng = np.random.RandomState(random_seed)

    def sample_candidates_for_query(
        self,
        query_id: str,
        positive_candidate_ids: Set[str],
        candidate_pool_ids: List[str],
        candidate_scores: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Samples candidates for a single query.
        
        Args:
            query_id: ID of the query case.
            positive_candidate_ids: Set of relevant candidate IDs for this query.
            candidate_pool_ids: Complete list of candidate IDs in the candidate pool.
            candidate_scores: Lexical scores (e.g. BM25) for each candidate in candidate_pool_ids.
            
        Returns:
            Dictionary containing:
            - 'selected_candidate_ids': Ordered list of candidate IDs for this query.
            - 'labels': Binary list of labels (1 for positive, 0 for negative).
            - 'negative_types': List of negative types ('positive', 'hard_negative', 'random_negative').
            - 'positive_count': Number of positive pairs retained.
            - 'hard_negative_count': Number of hard negatives sampled.
            - 'random_negative_count': Number of random negatives sampled.
        """
        n_cands = len(candidate_pool_ids)
        if len(candidate_scores) != n_cands:
            raise ValueError(f"Score count ({len(candidate_scores)}) does not match candidate pool ({n_cands})")

        # 1. Positives: All valid relevant candidates in the pool, excluding self-match
        valid_positives = [
            cid for cid in candidate_pool_ids
            if cid in positive_candidate_ids and cid != query_id
        ]

        # 2. Hard Negatives: Sort candidates by score descending
        sorted_indices = np.argsort(candidate_scores)[::-1]
        
        hard_negatives: List[str] = []
        positive_set = set(valid_positives)

        for idx in sorted_indices:
            cid = candidate_pool_ids[idx]
            # Exclude self-matches and positives
            if cid != query_id and cid not in positive_set:
                hard_negatives.append(cid)
                if len(hard_negatives) >= self.n_hard_negatives:
                    break

        # 3. Random Negatives: Sample uniformly from the remainder
        hard_neg_set = set(hard_negatives)
        remaining_negatives = [
            cid for cid in candidate_pool_ids
            if cid != query_id and cid not in positive_set and cid not in hard_neg_set
        ]

        if len(remaining_negatives) <= self.n_random_negatives:
            random_negatives = list(remaining_negatives)
        else:
            # Deterministic selection per query
            q_seed = (self.random_seed + abs(hash(query_id))) % (2**31 - 1)
            q_rng = np.random.RandomState(q_seed)
            sampled_indices = q_rng.choice(
                len(remaining_negatives), size=self.n_random_negatives, replace=False
            )
            random_negatives = [remaining_negatives[i] for i in sampled_indices]

        # Combine
        selected_ids: List[str] = []
        labels: List[int] = []
        neg_types: List[str] = []

        for cid in valid_positives:
            selected_ids.append(cid)
            labels.append(1)
            neg_types.append("positive")

        for cid in hard_negatives:
            selected_ids.append(cid)
            labels.append(0)
            neg_types.append("hard_negative")

        for cid in random_negatives:
            selected_ids.append(cid)
            labels.append(0)
            neg_types.append("random_negative")

        return {
            "selected_candidate_ids": selected_ids,
            "labels": labels,
            "negative_types": neg_types,
            "positive_count": len(valid_positives),
            "hard_negative_count": len(hard_negatives),
            "random_negative_count": len(random_negatives),
        }

