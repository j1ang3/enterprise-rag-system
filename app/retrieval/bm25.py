import math
import re
from collections import Counter, defaultdict
from copy import deepcopy
from typing import Any, Callable, Dict, List, Mapping, Sequence


BM25Tokenizer = Callable[[str], List[str]]
TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9_]+|[\u4e00-\u9fff]")
STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "does",
    "for",
    "how",
    "is",
    "it",
    "of",
    "the",
    "this",
    "to",
    "what",
    "with",
}


def tokenize_bm25(text: str) -> List[str]:
    """Tokenize lexical text without adding semantic synonyms or expansions."""
    return [
        token.lower()
        for token in TOKEN_PATTERN.findall(text)
        if token.lower() not in STOP_WORDS
    ]


class BM25Index:
    """In-memory BM25 index built from the repository's structured chunks."""

    def __init__(
        self,
        chunks: Sequence[Mapping[str, Any]],
        *,
        tokenizer: BM25Tokenizer = tokenize_bm25,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        if k1 <= 0:
            raise ValueError("k1 must be greater than 0")
        if not 0 <= b <= 1:
            raise ValueError("b must be between 0 and 1")

        self.k1 = k1
        self.b = b
        self._tokenizer = tokenizer
        self._chunks: List[Dict[str, Any]] = []
        self._document_lengths: List[int] = []
        self._postings: Dict[str, Dict[int, int]] = defaultdict(dict)
        self._document_frequencies: Counter[str] = Counter()

        for document_index, source_chunk in enumerate(chunks):
            content = source_chunk.get("content")
            if not isinstance(content, str):
                raise ValueError("each chunk must contain string content")

            # The index owns its metadata copy so retrieval scores never mutate
            # the canonical chunk dictionaries supplied by the caller.
            self._chunks.append(deepcopy(dict(source_chunk)))
            term_frequencies = Counter(tokenizer(content))
            self._document_lengths.append(sum(term_frequencies.values()))

            for term, frequency in term_frequencies.items():
                self._postings[term][document_index] = frequency
                self._document_frequencies[term] += 1

        self.average_document_length = (
            sum(self._document_lengths) / len(self._document_lengths)
            if self._document_lengths
            else 0.0
        )

    def __len__(self) -> int:
        return len(self._chunks)

    def tokenize(self, text: str) -> List[str]:
        return self._tokenizer(text)

    def document_frequency(self, term: str) -> int:
        return self._document_frequencies.get(term, 0)

    def postings(self, term: str) -> Mapping[int, int]:
        return self._postings.get(term, {})

    def document_length(self, document_index: int) -> int:
        return self._document_lengths[document_index]

    def chunk(self, document_index: int) -> Dict[str, Any]:
        return deepcopy(self._chunks[document_index])

    def inverse_document_frequency(self, term: str) -> float:
        document_frequency = self.document_frequency(term)
        if document_frequency == 0:
            return 0.0

        document_count = len(self)
        return math.log(
            1.0
            + (document_count - document_frequency + 0.5)
            / (document_frequency + 0.5)
        )


class BM25Retriever:
    """Rank chunks from a BM25 index through a small retriever interface."""

    source = "bm25"

    def __init__(self, index: BM25Index) -> None:
        self.index = index

    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")
        if not query.strip() or len(self.index) == 0:
            return []

        scores: Dict[int, float] = defaultdict(float)
        for term in set(self.index.tokenize(query)):
            inverse_document_frequency = self.index.inverse_document_frequency(term)
            for document_index, term_frequency in self.index.postings(term).items():
                document_length = self.index.document_length(document_index)
                length_ratio = (
                    document_length / self.index.average_document_length
                    if self.index.average_document_length > 0
                    else 0.0
                )
                denominator = term_frequency + self.index.k1 * (
                    1.0 - self.index.b + self.index.b * length_ratio
                )
                scores[document_index] += inverse_document_frequency * (
                    term_frequency * (self.index.k1 + 1.0) / denominator
                )

        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        results = []
        for document_index, score in ranked[:top_k]:
            if score <= 0:
                continue
            results.append(
                {
                    **self.index.chunk(document_index),
                    "score": round(score, 6),
                    "bm25_score": round(score, 6),
                    "retrieval_mode": "bm25",
                }
            )
        return results
