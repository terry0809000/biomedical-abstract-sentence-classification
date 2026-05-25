from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion


def build_word_tfidf(max_features: int = 80000) -> TfidfVectorizer:
    return TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.98,
        max_features=max_features,
        sublinear_tf=True,
    )


def build_word_char_tfidf(max_word_features: int = 70000, max_char_features: int = 40000) -> FeatureUnion:
    word = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.98,
        max_features=max_word_features,
        sublinear_tf=True,
    )
    char = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=3,
        max_features=max_char_features,
        sublinear_tf=True,
    )
    return FeatureUnion([("word", word), ("char", char)], n_jobs=1)

