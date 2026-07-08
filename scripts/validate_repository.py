from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REQUIRED = [
    "README.md",
    "CITATION.cff",
    "DATA_ACCESS.md",
    "docs/CLINICAL_ALGORITHMICIAN_ANNOTATION.md",
    "docs/DEPLOYMENT_READINESS.md",
    ".zenodo.json",
    "Dockerfile",
]
FORBIDDEN_SUFFIXES = {
    ".zip", ".7z", ".tar", ".gz", ".bz2", ".xz",
    ".parquet", ".feather", ".pkl", ".pickle", ".joblib",
    ".pt", ".pth", ".ckpt", ".onnx", ".duckdb", ".sqlite", ".db",
}
FORBIDDEN_DIR_NAMES = {
    ".git", ".venv", "venv", "__pycache__", ".mypy_cache",
    ".pytest_cache", "node_modules",
}
PERSONAL_PATH_PATTERNS = [
    re.compile(r"C:\\Users\\" + "Terry", re.IGNORECASE),
    re.compile(r"C:/Users/" + "Terry", re.IGNORECASE),
    re.compile(r"Desktop\\Articles", re.IGNORECASE),
    re.compile(r"Documents\\Codex", re.IGNORECASE),
]
TEXT_SUFFIXES = {
    ".md", ".py", ".txt", ".yml", ".yaml", ".json", ".toml",
    ".cff", ".csv", ".tex", ".bib", ".rst", ".html",
}


def iter_files() -> list[Path]:
    files: list[Path] = []
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        if any(part in FORBIDDEN_DIR_NAMES for part in path.relative_to(ROOT).parts):
            continue
        files.append(path)
    return files


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def check_required(errors: list[str]) -> None:
    for item in REQUIRED:
        if not (ROOT / item).exists():
            errors.append(f"missing required public-release artifact: {item}")


def check_forbidden_files(errors: list[str]) -> None:
    for path in iter_files():
        suffixes = "".join(path.suffixes).lower()
        if any(suffixes.endswith(s) for s in FORBIDDEN_SUFFIXES):
            errors.append(f"forbidden raw/binary/model artifact tracked: {rel(path)}")


def check_text_hygiene(errors: list[str]) -> None:
    for path in iter_files():
        if rel(path) == "scripts/validate_repository.py":
            continue
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = path.read_text(encoding="utf-8", errors="ignore")
        for pattern in PERSONAL_PATH_PATTERNS:
            if pattern.search(text):
                errors.append(f"personal workstation path remains in {rel(path)}")
                break


def check_python_syntax(errors: list[str]) -> None:
    for path in iter_files():
        if path.suffix.lower() != ".py":
            continue
        try:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError as exc:
            errors.append(f"python syntax error in {rel(path)}:{exc.lineno}: {exc.msg}")


def check_notebook_json(errors: list[str]) -> None:
    for path in iter_files():
        if path.suffix.lower() != ".ipynb":
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            errors.append(f"invalid notebook JSON in {rel(path)}: {exc}")
            continue
        if payload.get("nbformat") != 4 or "cells" not in payload:
            errors.append(f"unexpected notebook structure in {rel(path)}")


def check_zenodo(errors: list[str]) -> None:
    path = ROOT / ".zenodo.json"
    if not path.exists():
        return
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        errors.append(f".zenodo.json is not valid JSON: {exc}")
        return
    for key in ["title", "upload_type", "description", "creators", "access_right"]:
        if key not in payload:
            errors.append(f".zenodo.json missing {key}")


def main() -> int:
    errors: list[str] = []
    check_required(errors)
    check_forbidden_files(errors)
    check_text_hygiene(errors)
    check_python_syntax(errors)
    check_notebook_json(errors)
    check_zenodo(errors)
    if errors:
        print("Repository validation failed:")
        for error in errors:
            print(f"- {error}")
        return 1
    print("Repository validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
