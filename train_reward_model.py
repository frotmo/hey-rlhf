import argparse
import copy
from pathlib import Path

import torch

from preference_loss import preference_loss
from reward_model import RewardModel
from tokenizer import Tokenizer, pad_sequences
from toy_dataset import data, preferences


def set_seed(seed):
    torch.manual_seed(seed)


def sequence_ids(tokenizer, prompt, response):
    return (
        tokenizer.encode(prompt, add_bos=True)
        + tokenizer.encode(response, add_eos=True)
    )


def score_sequences(model, sequences, pad_id):
    input_ids, _ = pad_sequences(sequences, pad_id)
    lengths = torch.tensor([len(sequence) for sequence in sequences])
    return model(input_ids, lengths).squeeze(-1)


def pairwise_accuracy(model, chosen, rejected, pad_id):
    with torch.no_grad():
        chosen_scores = score_sequences(model, chosen, pad_id)
        rejected_scores = score_sequences(model, rejected, pad_id)
        return (chosen_scores > rejected_scores).float().mean().item()


def train_reward_model(
    epochs=250,
    learning_rate=5e-3,
    seed=11,
    output="artifacts/reward_model.pt",
):
    set_seed(seed)
    tokenizer = Tokenizer().build_vocab(data + preferences)
    config = {"embed": 64, "hidden": 128}
    model = RewardModel(tokenizer.vocab_size, **config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    chosen = [
        sequence_ids(tokenizer, item["prompt"], item["chosen"])
        for item in preferences
    ]
    rejected = [
        sequence_ids(tokenizer, item["prompt"], item["rejected"])
        for item in preferences
    ]

    best_accuracy = -1.0
    best_state = None
    for epoch in range(1, epochs + 1):
        chosen_scores = score_sequences(model, chosen, tokenizer.pad_id)
        rejected_scores = score_sequences(model, rejected, tokenizer.pad_id)
        loss = preference_loss(chosen_scores, rejected_scores)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch == 1 or epoch % 25 == 0:
            accuracy = pairwise_accuracy(model, chosen, rejected, tokenizer.pad_id)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_state = copy.deepcopy(model.state_dict())
            print(
                f"Reward epoch {epoch:03d} loss {loss.item():.4f} "
                f"pairwise accuracy {accuracy:.3f}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)
    final_accuracy = pairwise_accuracy(model, chosen, rejected, tokenizer.pad_id)
    chosen_scores = score_sequences(model, chosen, tokenizer.pad_id).detach()
    rejected_scores = score_sequences(model, rejected, tokenizer.pad_id).detach()
    final_margin = (chosen_scores - rejected_scores).mean().item()

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state_dict": model.state_dict(), "vocab": tokenizer.vocab, "config": config},
        output_path,
    )
    print(
        f"Final pairwise accuracy {final_accuracy:.3f}, "
        f"margin {final_margin:.3f}, saved to {output_path}"
    )
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--learning-rate", type=float, default=5e-3)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--output", default="artifacts/reward_model.pt")
    args = parser.parse_args()
    train_reward_model(args.epochs, args.learning_rate, args.seed, args.output)


if __name__ == "__main__":
    main()
