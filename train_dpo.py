import argparse
from pathlib import Path

import torch

from checkpoints import load_policy
from dpo_step import dpo_loss, response_sequence
from ppo_step import make_reference
from toy_dataset import preferences


def train_dpo(
    sft_path="artifacts/sft_policy.pt",
    output="artifacts/dpo_policy.pt",
    epochs=100,
    learning_rate=1e-4,
    beta=0.1,
    seed=29,
):
    torch.manual_seed(seed)
    policy, tokenizer, checkpoint = load_policy(sft_path)
    reference = make_reference(policy)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=learning_rate)

    chosen = [
        response_sequence(tokenizer, item["prompt"], item["chosen"])
        for item in preferences
    ]
    rejected = [
        response_sequence(tokenizer, item["prompt"], item["rejected"])
        for item in preferences
    ]
    chosen_sequences = [sequence for sequence, _ in chosen]
    rejected_sequences = [sequence for sequence, _ in rejected]
    chosen_prompt_lengths = [length for _, length in chosen]
    rejected_prompt_lengths = [length for _, length in rejected]

    for epoch in range(1, epochs + 1):
        policy.train()
        loss, _, _ = dpo_loss(
            policy,
            reference,
            chosen_sequences,
            rejected_sequences,
            chosen_prompt_lengths,
            rejected_prompt_lengths,
            tokenizer.pad_id,
            beta=beta,
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch == 1 or epoch % 10 == 0:
            policy.eval()
            with torch.no_grad():
                eval_loss, logits, accuracy = dpo_loss(
                    policy,
                    reference,
                    chosen_sequences,
                    rejected_sequences,
                    chosen_prompt_lengths,
                    rejected_prompt_lengths,
                    tokenizer.pad_id,
                    beta=beta,
                )
            print(
                f"DPO epoch {epoch:03d} loss {eval_loss.item():.4f} "
                f"accuracy {accuracy.item():.3f} margin {logits.mean().item():.4f}"
            )

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": policy.state_dict(),
            "vocab": tokenizer.vocab,
            "config": checkpoint["config"],
        },
        output_path,
    )
    print(f"Saved DPO policy to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft-checkpoint", default="artifacts/sft_policy.pt")
    parser.add_argument("--output", default="artifacts/dpo_policy.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=29)
    args = parser.parse_args()
    train_dpo(
        args.sft_checkpoint,
        args.output,
        args.epochs,
        args.learning_rate,
        args.beta,
        args.seed,
    )


if __name__ == "__main__":
    main()
