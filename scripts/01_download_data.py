from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.parsers import sha256_file
from src.utils.paths import RAW_DIR, ensure_project_dirs


SOURCES = {
    "pico_train": {
        "url": "https://raw.githubusercontent.com/jind11/PubMed-PICO-Detection/master/splitted/PICO_train.txt",
        "path": RAW_DIR / "pubmed_pico_train.txt",
    },
    "pico_dev": {
        "url": "https://raw.githubusercontent.com/jind11/PubMed-PICO-Detection/master/splitted/PICO_dev.txt",
        "path": RAW_DIR / "pubmed_pico_dev.txt",
    },
    "pico_test": {
        "url": "https://raw.githubusercontent.com/jind11/PubMed-PICO-Detection/master/splitted/PICO_test.txt",
        "path": RAW_DIR / "pubmed_pico_test.txt",
    },
    "pubmed_rct20k_test": {
        "url": "https://raw.githubusercontent.com/Franck-Dernoncourt/pubmed-rct/master/PubMed_20k_RCT/test.txt",
        "path": RAW_DIR / "pubmed_20k_rct_test.txt",
    },
}


def download(url: str, path: Path, retries: int = 4) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        return {"status": "exists", "bytes": path.stat().st_size}
    last_error = None
    for attempt in range(1, retries + 1):
        try:
            with requests.get(url, stream=True, timeout=60) as response:
                response.raise_for_status()
                with path.open("wb") as f:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
            return {"status": "downloaded", "bytes": path.stat().st_size, "attempt": attempt}
        except Exception as exc:
            last_error = repr(exc)
            time.sleep(2 * attempt)
    return {"status": "failed", "error": last_error}


def main() -> None:
    ensure_project_dirs()
    manifest = {}
    for name, spec in SOURCES.items():
        result = download(spec["url"], spec["path"])
        result["url"] = spec["url"]
        result["path"] = str(spec["path"])
        if spec["path"].exists() and spec["path"].stat().st_size > 0:
            result["sha256"] = sha256_file(spec["path"])
        manifest[name] = result
    out = RAW_DIR / "download_manifest.json"
    out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

