from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

import numpy as np


TOKEN_RE = re.compile(r"[A-Za-z]+|[@]|\d+(?:\.\d+)?|[^\sA-Za-z0-9]")


def simple_tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(str(text).lower())


@dataclass
class NeuralTokenizer:
    max_vocab_size: int = 40000
    min_freq: int = 2
    pad_token: str = "<PAD>"
    unk_token: str = "<UNK>"
    token_to_id: dict[str, int] = field(default_factory=dict)
    id_to_token: list[str] = field(default_factory=list)

    def fit(self, texts) -> "NeuralTokenizer":
        counter: Counter[str] = Counter()
        for text in texts:
            counter.update(simple_tokenize(text))
        vocab = [self.pad_token, self.unk_token]
        for token, freq in counter.most_common(max(0, self.max_vocab_size - 2)):
            if freq < self.min_freq:
                continue
            vocab.append(token)
        self.id_to_token = vocab
        self.token_to_id = {tok: idx for idx, tok in enumerate(vocab)}
        return self

    def encode(self, text: str, max_len: int) -> list[int]:
        unk = self.token_to_id[self.unk_token]
        ids = [self.token_to_id.get(tok, unk) for tok in simple_tokenize(text)[:max_len]]
        return ids

    def transform(self, texts, max_len: int) -> np.ndarray:
        arr = np.zeros((len(texts), max_len), dtype=np.int64)
        for i, text in enumerate(texts):
            ids = self.encode(text, max_len=max_len)
            if ids:
                arr[i, : len(ids)] = ids
        return arr

    @property
    def vocab_size(self) -> int:
        return len(self.id_to_token)

