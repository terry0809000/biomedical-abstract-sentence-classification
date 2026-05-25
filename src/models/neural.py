from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import Dataset


class TextDataset(Dataset):
    def __init__(self, x, y=None):
        self.x = torch.tensor(x, dtype=torch.long)
        self.y = None if y is None else torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        if self.y is None:
            return self.x[idx]
        return self.x[idx], self.y[idx]


class NeuralTextClassifier(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 64,
        hidden_dim: int = 64,
        model_type: str = "lstm",
        dropout: float = 0.0,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.model_type = model_type
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.dropout = nn.Dropout(dropout)
        if model_type == "ffnn_avg":
            self.encoder = nn.Sequential(
                nn.Linear(embedding_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )
            out_dim = hidden_dim
        elif model_type in {"lstm", "bilstm"}:
            self.rnn = nn.LSTM(
                embedding_dim,
                hidden_dim,
                batch_first=True,
                bidirectional=(model_type == "bilstm"),
            )
            out_dim = hidden_dim * (2 if model_type == "bilstm" else 1)
        elif model_type == "cnn":
            self.convs = nn.ModuleList([nn.Conv1d(embedding_dim, hidden_dim, k) for k in (3, 4, 5)])
            out_dim = hidden_dim * 3
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
        self.classifier = nn.Linear(out_dim, 1)

    def forward(self, x):
        mask = (x != 0).float()
        emb = self.dropout(self.embedding(x))
        if self.model_type == "ffnn_avg":
            denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
            pooled = (emb * mask.unsqueeze(-1)).sum(dim=1) / denom
            features = self.encoder(pooled)
        elif self.model_type in {"lstm", "bilstm"}:
            _, (h_n, _) = self.rnn(emb)
            if self.model_type == "bilstm":
                features = torch.cat([h_n[-2], h_n[-1]], dim=1)
            else:
                features = h_n[-1]
            features = self.dropout(features)
        else:
            z = emb.transpose(1, 2)
            convs = [torch.relu(conv(z)).max(dim=2).values for conv in self.convs]
            features = self.dropout(torch.cat(convs, dim=1))
        return self.classifier(features).squeeze(1)

