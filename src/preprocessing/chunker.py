"""Sentence-aware text chunking module for legal documents.

Chunks long judicial opinions while strictly preserving sentence boundaries,
document identifiers, sentence index spans, chunk order, and token constraints.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from tokenizers import Tokenizer


class SentenceAwareChunker:
    """Chunks documents into sentence-aligned spans suitable for transformer models."""

    def __init__(
        self,
        tokenizer: Optional[Union[Tokenizer, str, Path]] = None,
        max_tokens: int = 512,
        overlap_tokens: int = 64,
        special_tokens_reserve: int = 2,  # Reserves space for [CLS] and [SEP]
    ):
        """Initialize chunker.

        Args:
            tokenizer: Pre-initialized Tokenizer instance or path to tokenizer.json.
            max_tokens: Maximum allowed tokens per chunk (including special tokens).
            overlap_tokens: Target token overlap between consecutive chunks.
            special_tokens_reserve: Tokens reserved for model special tokens (default 2 for BERT).
        """
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.special_tokens_reserve = special_tokens_reserve
        self.effective_max_tokens = max(10, max_tokens - special_tokens_reserve)

        # Initialize tokenizer
        if isinstance(tokenizer, Tokenizer):
            self.tokenizer = tokenizer
        elif tokenizer is not None and Path(tokenizer).exists():
            self.tokenizer = Tokenizer.from_file(str(tokenizer))
        else:
            default_path = Path("models/bert/tokenizer.json")
            if default_path.exists():
                self.tokenizer = Tokenizer.from_file(str(default_path))
            else:
                self.tokenizer = None

    def count_tokens(self, text: str) -> int:
        """Count tokens using tokenizer or robust whitespace heuristic if tokenizer unavailable."""
        if self.tokenizer is not None:
            # Tokenizer count without special tokens
            return len(self.tokenizer.encode(text, add_special_tokens=False).ids)
        # Fallback whitespace approximation with 1.3 subword factor
        words = text.split()
        return max(1, int(len(words) * 1.3))

    def count_tokens_batch(self, texts: List[str]) -> List[int]:
        """Count tokens for a batch of strings using fast batch tokenization."""
        if not texts:
            return []
        if self.tokenizer is not None:
            encodings = self.tokenizer.encode_batch(texts, add_special_tokens=False)
            return [len(e.ids) for e in encodings]
        return [max(1, int(len(t.split()) * 1.3)) for t in texts]

    def chunk_document(
        self,
        doc_id: str,
        sentences: List[str],
    ) -> List[Dict[str, Any]]:
        """Chunk a document given its list of sentences.

        Args:
            doc_id: Case document ID.
            sentences: Ordered list of clean sentence strings.

        Returns:
            List of chunk dictionaries with metadata.
        """
        if not sentences:
            return []

        # Fast batch token length computation for all sentences in the document
        sentence_token_counts = self.count_tokens_batch(sentences)

        chunks: List[Dict[str, Any]] = []
        n_sentences = len(sentences)
        current_idx = 0
        chunk_idx = 0

        while current_idx < n_sentences:
            chunk_sentences: List[str] = []
            chunk_token_count = 0
            start_sentence_idx = current_idx
            idx = current_idx

            # Build chunk by adding full sentences until reaching effective_max_tokens
            while idx < n_sentences:
                s_tokens = sentence_token_counts[idx]

                # Case A: A single sentence exceeds effective_max_tokens on its own
                if s_tokens > self.effective_max_tokens and len(chunk_sentences) == 0:
                    # Must split this unusually long sentence into sub-chunks
                    sub_chunks = self._split_oversized_sentence(
                        doc_id=doc_id,
                        sentence=sentences[idx],
                        sentence_idx=idx,
                        start_chunk_idx=chunk_idx,
                    )
                    chunks.extend(sub_chunks)
                    chunk_idx += len(sub_chunks)
                    current_idx = idx + 1
                    idx = current_idx
                    break

                # Case B: Sentence fits in current chunk
                if chunk_token_count + s_tokens <= self.effective_max_tokens:
                    chunk_sentences.append(sentences[idx])
                    chunk_token_count += s_tokens
                    idx += 1
                else:
                    # Current chunk is full
                    break

            if idx == current_idx:
                # Handled oversized sentence branch
                continue

            end_sentence_idx = idx - 1
            chunk_text = " ".join(chunk_sentences)

            chunks.append(
                {
                    "doc_id": str(doc_id),
                    "chunk_id": f"{doc_id}_chk_{chunk_idx:04d}",
                    "chunk_idx": chunk_idx,
                    "text": chunk_text,
                    "token_count": chunk_token_count,
                    "sentence_count": len(chunk_sentences),
                    "start_sentence_idx": start_sentence_idx,
                    "end_sentence_idx": end_sentence_idx,
                }
            )
            chunk_idx += 1

            if idx >= n_sentences:
                # Reached end of document
                break

            # Step 2: Calculate step forward using overlap_tokens
            overlap_accum = 0
            back_idx = idx - 1
            while back_idx > current_idx and overlap_accum < self.overlap_tokens:
                overlap_accum += sentence_token_counts[back_idx]
                back_idx -= 1

            # Advance current_idx to the sentence after overlap start
            next_idx = back_idx + 1
            if next_idx <= current_idx:
                next_idx = current_idx + 1

            current_idx = next_idx

        return chunks

    def _split_oversized_sentence(
        self,
        doc_id: str,
        sentence: str,
        sentence_idx: int,
        start_chunk_idx: int,
    ) -> List[Dict[str, Any]]:
        """Splits an oversized single sentence by word windows to maintain natural text."""
        words = sentence.split()
        if not words:
            return []

        # Target ~380 words per window (~490 tokens) with ~50 words overlap
        target_words = max(10, int(self.effective_max_tokens / 1.35))
        overlap_w = max(2, int(self.overlap_tokens / 1.35))
        step = max(1, target_words - overlap_w)

        sub_chunks: List[Dict[str, Any]] = []
        c_idx = start_chunk_idx

        for i in range(0, len(words), step):
            window_words = words[i : i + target_words]
            chunk_text = " ".join(window_words)
            tok_count = self.count_tokens(chunk_text)

            sub_chunks.append(
                {
                    "doc_id": str(doc_id),
                    "chunk_id": f"{doc_id}_chk_{c_idx:04d}",
                    "chunk_idx": c_idx,
                    "text": chunk_text,
                    "token_count": tok_count,
                    "sentence_count": 1,
                    "start_sentence_idx": sentence_idx,
                    "end_sentence_idx": sentence_idx,
                }
            )
            c_idx += 1
            if i + target_words >= len(words):
                break

        return sub_chunks
