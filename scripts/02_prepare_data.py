from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.parsers import PICO_POSITIVE, RCT_POSITIVE, add_text_features, parse_pubmed_pico, parse_pubmed_rct, sha256_file
from src.utils.paths import OUTPUTS_DIR, PROCESSED_DIR, RAW_DIR, ensure_project_dirs


def save(df: pd.DataFrame, filename: str) -> Path:
    path = PROCESSED_DIR / filename
    df.to_csv(path, index=False)
    return path


def main() -> None:
    ensure_project_dirs()
    inputs = {
        "train": RAW_DIR / "pubmed_pico_train.txt",
        "validation": RAW_DIR / "pubmed_pico_dev.txt",
        "test": RAW_DIR / "pubmed_pico_test.txt",
    }
    missing = [str(p) for p in inputs.values() if not p.exists()]
    external_path = RAW_DIR / "pubmed_20k_rct_test.txt"
    if not external_path.exists():
        missing.append(str(external_path))
    if missing:
        raise FileNotFoundError(f"Missing raw files. Run scripts/01_download_data.py first: {missing}")

    primary_frames = []
    output_files = {}
    for split, path in inputs.items():
        df = add_text_features(parse_pubmed_pico(path, split))
        out = save(df, f"primary_{split}.csv")
        output_files[f"primary_{split}"] = str(out)
        primary_frames.append(df)
    primary_all = pd.concat(primary_frames, ignore_index=True)
    output_files["primary_all"] = str(save(primary_all, "primary_all.csv"))

    external = add_text_features(parse_pubmed_rct(external_path, "external_test"))
    output_files["external_pubmed_rct20k_test"] = str(save(external, "external_pubmed_rct20k_test.csv"))

    label_mapping = {
        "task": "binary findings-or-conclusion sentence detection",
        "positive_class": "findings_conclusion",
        "negative_class": "other",
        "primary_pubmed_pico_positive_source_labels": sorted(PICO_POSITIVE),
        "external_pubmed_rct20k_positive_source_labels": sorted(RCT_POSITIVE),
        "label_note": "The task is sentence-level detection of result/conclusion rhetorical content. PubMed-PICO labels R and C are mapped to positive; PubMed RCT labels RESULTS and CONCLUSIONS are mapped to positive for external validation.",
    }
    (PROCESSED_DIR / "label_mapping.json").write_text(json.dumps(label_mapping, indent=2), encoding="utf-8")

    raw_manifest = {}
    download_manifest_path = RAW_DIR / "download_manifest.json"
    if download_manifest_path.exists():
        raw_manifest = json.loads(download_manifest_path.read_text(encoding="utf-8"))
    data_manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "selected_dataset": "PubMed-PICO Detection",
        "selected_task": "Binary detection of findings/conclusion sentences in biomedical abstracts",
        "external_validation_dataset": "PubMed 20k RCT official test split",
        "primary_rows": int(len(primary_all)),
        "external_rows": int(len(external)),
        "primary_splits": {split: int((primary_all["split"] == split).sum()) for split in ["train", "validation", "test"]},
        "label_mapping": label_mapping,
        "raw_download_manifest": raw_manifest,
        "processed_files": output_files,
        "processed_sha256": {k: sha256_file(Path(v)) for k, v in output_files.items()},
    }
    out = OUTPUTS_DIR / "02_data" / "data_manifest.json"
    out.write_text(json.dumps(data_manifest, indent=2), encoding="utf-8")
    print(json.dumps(data_manifest, indent=2))


if __name__ == "__main__":
    main()

