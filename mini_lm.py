import torch
import torch.nn as nn


class MiniLM(nn.Module):
    """A small causal actor-critic language model.

    The LSTM is unidirectional, so logits at position t only depend on tokens
    up to t. The shared trunk feeds both the token policy and the scalar critic.
    """

    def __init__(self, vocab, embed=64, hidden=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab, embed, padding_idx=0)
        self.lstm = nn.LSTM(embed, hidden, batch_first=True)
        self.lm_head = nn.Linear(hidden, vocab)
        self.value_head = nn.Linear(hidden, 1)

    def forward(self, input_ids):
        hidden = self.embedding(input_ids)
        hidden, _ = self.lstm(hidden)
        logits = self.lm_head(hidden)
        values = self.value_head(hidden).squeeze(-1)
        return logits, values
