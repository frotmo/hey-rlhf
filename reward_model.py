import torch
import torch.nn as nn


class RewardModel(nn.Module):
    """Score a complete prompt-response sequence with one scalar."""

    def __init__(self, vocab, embed=64, hidden=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab, embed, padding_idx=0)
        self.lstm = nn.LSTM(embed, hidden, batch_first=True)
        self.score = nn.Linear(hidden, 1)

    def forward(self, input_ids, lengths=None):
        hidden, _ = self.lstm(self.embedding(input_ids))
        if lengths is None:
            lengths = (input_ids != 0).sum(dim=1)
        lengths = lengths.clamp(min=1)
        last = hidden[torch.arange(hidden.size(0), device=hidden.device), lengths - 1]
        return self.score(last)
