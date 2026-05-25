from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.calibration.temperature import TemperatureScaler
from src.evaluation.metrics import binary_metrics, optimize_threshold
from src.features.neural_tokenizer import NeuralTokenizer
from src.models.neural import NeuralTextClassifier, TextDataset
from src.utils.paths import OUTPUTS_DIR, PROCESSED_DIR, RUNS_DIR, ARTIFACTS_DIR, ensure_project_dirs
from src.utils.seed import set_seed


SEED = 42
MAX_LEN = 64
MAX_NEURAL_TRAIN = 120000


def stratified_cap(df: pd.DataFrame, max_n: int, seed: int) -> pd.DataFrame:
    if len(df) <= max_n:
        return df
    frac = max_n / len(df)
    return (
        df.groupby("label", group_keys=False)
        .apply(lambda x: x.sample(max(1, int(round(len(x) * frac))), random_state=seed), include_groups=True)
        .sample(frac=1.0, random_state=seed)
        .reset_index(drop=True)
    )


def predict_logits(model, x, batch_size: int = 1024) -> np.ndarray:
    model.eval()
    loader = DataLoader(TextDataset(x), batch_size=batch_size, shuffle=False)
    logits = []
    with torch.no_grad():
        for xb in loader:
            logits.append(model(xb).detach().cpu().numpy())
    return np.concatenate(logits)


def run_model(model_name: str, model_type: str, dropout: float, tokenizer, arrays, frames, train_y, val_y) -> None:
    set_seed(SEED)
    run_dir = RUNS_DIR / model_name
    run_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "model": model_name,
        "model_type": model_type,
        "seed": SEED,
        "max_sequence_length": MAX_LEN,
        "max_neural_train_samples": MAX_NEURAL_TRAIN,
        "vocab_size": tokenizer.vocab_size,
        "embedding_dim": 64,
        "hidden_dim": 64,
        "dropout": dropout,
        "batch_size": 512,
        "max_epochs": 5,
        "early_stopping_patience": 2,
        "calibration": "Temperature scaling fitted on validation logits",
        "threshold": "Optimized for validation F1 only",
        "device": "cpu",
    }
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    (run_dir / "seed.txt").write_text(str(SEED), encoding="utf-8")

    model = NeuralTextClassifier(
        vocab_size=tokenizer.vocab_size,
        embedding_dim=64,
        hidden_dim=64,
        model_type=model_type,
        dropout=dropout,
    )
    pos = float(train_y.sum())
    neg = float(len(train_y) - train_y.sum())
    pos_weight = torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loader = DataLoader(TextDataset(arrays["train"], train_y), batch_size=512, shuffle=True)

    best_auc = -1.0
    best_state = None
    stale = 0
    log_lines = []
    for epoch in range(1, config["max_epochs"] + 1):
        model.train()
        losses = []
        for xb, yb in loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
        val_logits = predict_logits(model, arrays["validation"])
        val_prob = 1.0 / (1.0 + np.exp(-np.clip(val_logits, -60, 60)))
        val_auc = roc_auc_score(val_y, val_prob)
        log_line = f"epoch={epoch} train_loss={np.mean(losses):.6f} validation_auc={val_auc:.6f}"
        print(model_name, log_line)
        log_lines.append(log_line)
        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= config["early_stopping_patience"]:
            log_lines.append("early_stopping_triggered=true")
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), run_dir / "model.pt")
    joblib.dump(tokenizer, ARTIFACTS_DIR / f"{model_name}_tokenizer.joblib")

    val_logits = predict_logits(model, arrays["validation"])
    calibrator = TemperatureScaler().fit(val_logits, val_y)
    joblib.dump(calibrator, run_dir / "calibration_artifact.joblib")
    logits_by_split = {split: predict_logits(model, arr) for split, arr in arrays.items() if split != "train"}
    probs = {split: calibrator.predict_proba(logits) for split, logits in logits_by_split.items()}
    threshold = optimize_threshold(val_y, probs["validation"], metric="f1")
    log_lines.append(f"best_validation_auc={best_auc:.6f}")
    log_lines.append(f"temperature={calibrator.temperature_:.6f}")
    log_lines.append(f"threshold={threshold:.6f}")

    for split in ["validation", "test", "external"]:
        df = frames[split]
        pred = df[["dataset", "split", "abstract_id", "sentence_index", "source_label", "text", "label"]].copy()
        pred["raw_score"] = logits_by_split[split]
        pred["probability"] = probs[split]
        pred["prediction"] = (pred["probability"] >= threshold).astype(int)
        pred.to_csv(run_dir / f"predictions_{split}.csv", index=False)
        if split in {"validation", "test"}:
            pred.to_csv(OUTPUTS_DIR / "07_predictions" / f"{model_name}_{split}.csv", index=False)
        metrics = binary_metrics(pred["label"].to_numpy(), pred["probability"].to_numpy(), threshold)
        (run_dir / f"metrics_{split}.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        log_lines.append(f"{split}: roc_auc={metrics.get('roc_auc')} brier={metrics.get('brier_score')} ece={metrics.get('ece_10bin')}")
    (run_dir / "training_log.txt").write_text("\n".join(log_lines), encoding="utf-8")


def main() -> None:
    ensure_project_dirs()
    set_seed(SEED)
    train_full = pd.read_csv(PROCESSED_DIR / "primary_train.csv")
    train = stratified_cap(train_full, MAX_NEURAL_TRAIN, SEED)
    val = pd.read_csv(PROCESSED_DIR / "primary_validation.csv")
    test = pd.read_csv(PROCESSED_DIR / "primary_test.csv")
    external = pd.read_csv(PROCESSED_DIR / "external_pubmed_rct20k_test.csv")
    tokenizer = NeuralTokenizer(max_vocab_size=40000, min_freq=2).fit(train["text"].tolist())
    arrays = {
        "train": tokenizer.transform(train["text"].tolist(), MAX_LEN),
        "validation": tokenizer.transform(val["text"].tolist(), MAX_LEN),
        "test": tokenizer.transform(test["text"].tolist(), MAX_LEN),
        "external": tokenizer.transform(external["text"].tolist(), MAX_LEN),
    }
    frames = {"validation": val, "test": test, "external": external}
    train_y = train["label"].to_numpy().astype("float32")
    val_y = val["label"].to_numpy().astype("int64")
    model_specs = [
        ("ffnn_avg_embeddings", "ffnn_avg", 0.2),
        ("lstm", "lstm", 0.0),
        ("bilstm", "bilstm", 0.0),
        ("bilstm_dropout", "bilstm", 0.4),
    ]
    for model_name, model_type, dropout in model_specs:
        run_model(model_name, model_type, dropout, tokenizer, arrays, frames, train_y, val_y)


if __name__ == "__main__":
    main()

