import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from mini_lm import MiniLM
from tokenizer import Tokenizer, pad_sequences
from toy_dataset import data, preferences


def set_seed(seed):
    torch.manual_seed(seed)


def make_examples(tokenizer, pairs):
    inputs = []
    labels = []
    for prompt, response in pairs:
        context = tokenizer.encode(prompt, add_bos=True)
        answer = tokenizer.encode(response, add_eos=True)
        inputs.append(context + answer[:-1])
        # The last prompt token predicts the first answer token.
        labels.append([-100] * (len(context) - 1) + answer)
    return inputs, labels


def train_sft(epochs=300, learning_rate=3e-3, seed=7, output="artifacts/sft_policy.pt"):
    set_seed(seed)
    tokenizer = Tokenizer().build_vocab(data + preferences)
    input_lists, label_lists = make_examples(tokenizer, data)
    input_ids, _ = pad_sequences(input_lists, tokenizer.pad_id)
    padded_labels, _ = pad_sequences(label_lists, -100)

    config = {"embed": 64, "hidden": 128}
    model = MiniLM(tokenizer.vocab_size, **config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for epoch in range(1, epochs + 1):
        logits, _ = model(input_ids)
        loss = F.cross_entropy(
            logits.reshape(-1, tokenizer.vocab_size),
            padded_labels.reshape(-1),
            ignore_index=-100,
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch == 1 or epoch % 50 == 0:
            print(f"SFT epoch {epoch:03d} loss {loss.item():.4f}")

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state_dict": model.state_dict(), "vocab": tokenizer.vocab, "config": config},
        output_path,
    )
    print(f"Saved SFT policy to {output_path}")
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", default="artifacts/sft_policy.pt")
    args = parser.parse_args()
    train_sft(args.epochs, args.learning_rate, args.seed, args.output)


if __name__ == "__main__":
    main()
