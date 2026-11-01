import collections
import re

import torch


PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"


def _words(text):
    """A tiny word tokenizer that keeps code-like tokens readable."""
    return re.findall(r"[a-z0-9]+(?:'[a-z]+)?", text.lower())


def _iter_texts(items):
    for item in items:
        if isinstance(item, str):
            yield item
        elif isinstance(item, tuple):
            yield from item
        elif isinstance(item, dict):
            yield from (item["prompt"], item["chosen"], item["rejected"])


class Tokenizer:
    def __init__(self, vocab=None):
        self.vocab = dict(vocab) if vocab is not None else {
            PAD_TOKEN: 0,
            UNK_TOKEN: 1,
            BOS_TOKEN: 2,
            EOS_TOKEN: 3,
        }
        self.inverse_vocab = {idx: token for token, idx in self.vocab.items()}

    @property
    def vocab_size(self):
        return len(self.vocab)

    @property
    def pad_id(self):
        return self.vocab[PAD_TOKEN]

    @property
    def eos_id(self):
        return self.vocab[EOS_TOKEN]

    def build_vocab(self, items):
        counts = collections.Counter()
        for text in _iter_texts(items):
            counts.update(_words(text))

        for token in (PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN):
            self.vocab.pop(token, None)
        self.vocab = {
            PAD_TOKEN: 0,
            UNK_TOKEN: 1,
            BOS_TOKEN: 2,
            EOS_TOKEN: 3,
        }
        for token, _ in counts.most_common():
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab)

        self.inverse_vocab = {idx: token for token, idx in self.vocab.items()}
        return self

    def encode(self, text, add_bos=False, add_eos=False):
        ids = [self.vocab.get(word, self.vocab[UNK_TOKEN]) for word in _words(text)]
        if add_bos:
            ids.insert(0, self.vocab[BOS_TOKEN])
        if add_eos:
            ids.append(self.vocab[EOS_TOKEN])
        return ids

    def decode(self, ids, skip_special=True):
        special = {self.vocab[token] for token in (PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN)}
        words = [
            self.inverse_vocab.get(int(idx), UNK_TOKEN)
            for idx in ids
            if not skip_special or int(idx) not in special
        ]
        return " ".join(words)


def build_vocab(items):
    """Backward-compatible helper used by the original project."""
    return Tokenizer().build_vocab(items).vocab


def tokenize_text(text, vocab):
    """Return a [1, sequence_length] tensor for one text."""
    tokenizer = Tokenizer(vocab)
    ids = tokenizer.encode(text)
    if not ids:
        ids = [tokenizer.vocab[UNK_TOKEN]]
    return torch.tensor([ids], dtype=torch.long)


def pad_sequences(sequences, pad_id=0):
    """Right-pad a list of token lists and return ids plus a validity mask."""
    max_len = max(len(sequence) for sequence in sequences)
    ids = torch.full((len(sequences), max_len), pad_id, dtype=torch.long)
    mask = torch.zeros((len(sequences), max_len), dtype=torch.bool)
    for row, sequence in enumerate(sequences):
        ids[row, :len(sequence)] = torch.tensor(sequence, dtype=torch.long)
        mask[row, :len(sequence)] = True
    return ids, mask
