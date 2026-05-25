from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

import pandas as pd


PICO_LABELS = {
    "A": "Aim",
    "P": "Participants",
    "I": "Intervention",
    "O": "Outcome",
    "M": "Method",
    "R": "Results",
    "C": "Conclusion",
}
PICO_POSITIVE = {"R", "C"}
RCT_POSITIVE = {"RESULTS", "CONCLUSIONS"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_pubmed_pico(path: Path, split: str) -> pd.DataFrame:
    rows = []
    abstract_id = None
    sentence_index = 0
    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("###"):
                abstract_id = line[3:].rstrip(":")
                sentence_index = 0
                continue
            parts = line.split("|", 2)
            if len(parts) != 3:
                continue
            section_heading, label, text = parts
            rows.append(
                {
                    "dataset": "PubMed-PICO",
                    "split": split,
                    "abstract_id": abstract_id,
                    "sentence_index": sentence_index,
                    "section_heading": section_heading,
                    "source_label": label,
                    "source_label_name": PICO_LABELS.get(label, label),
                    "text": text.strip(),
                    "label": int(label in PICO_POSITIVE),
                    "label_name": "findings_conclusion" if label in PICO_POSITIVE else "other",
                }
            )
            sentence_index += 1
    return pd.DataFrame(rows)


def parse_pubmed_rct(path: Path, split: str = "external_test") -> pd.DataFrame:
    rows = []
    abstract_id = None
    sentence_index = 0
    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("###"):
                abstract_id = line[3:].rstrip(":")
                sentence_index = 0
                continue
            parts = line.split("\t", 1)
            if len(parts) != 2:
                continue
            label, text = parts
            rows.append(
                {
                    "dataset": "PubMed-20k-RCT",
                    "split": split,
                    "abstract_id": abstract_id,
                    "sentence_index": sentence_index,
                    "section_heading": label,
                    "source_label": label,
                    "source_label_name": label.title(),
                    "text": text.strip(),
                    "label": int(label in RCT_POSITIVE),
                    "label_name": "findings_conclusion" if label in RCT_POSITIVE else "other",
                }
            )
            sentence_index += 1
    return pd.DataFrame(rows)


def add_text_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["char_length"] = out["text"].astype(str).str.len()
    out["word_count"] = out["text"].astype(str).str.split().str.len()
    return out


def class_distribution(df: pd.DataFrame, group_cols: Iterable[str]) -> pd.DataFrame:
    group_cols = list(group_cols)
    counts = df.groupby(group_cols + ["label_name"], dropna=False).size().reset_index(name="n")
    totals = counts.groupby(group_cols)["n"].transform("sum")
    counts["proportion"] = counts["n"] / totals
    return counts

