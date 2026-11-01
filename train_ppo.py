import argparse
from pathlib import Path

import torch

from checkpoints import load_policy, load_reward
from ppo_step import collect_rollouts, compare_to_reference, make_reference, ppo_update
from toy_dataset import prompts


def build_models(sft_path, reward_path):
    policy, tokenizer, _ = load_policy(sft_path)
    reward_model = load_reward(reward_path, tokenizer)
    return policy, reward_model, tokenizer


def train_ppo(
    sft_path="artifacts/sft_policy.pt",
    reward_path="artifacts/reward_model.pt",
    output="artifacts/ppo_policy.pt",
    iterations=30,
    learning_rate=3e-4,
    beta=0.02,
    max_new_tokens=16,
    seed=17,
):
    torch.manual_seed(seed)
    policy, reward_model, tokenizer = build_models(sft_path, reward_path)
    reference = make_reference(policy)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=learning_rate)

    before = compare_to_reference(
        policy,
        reference,
        reward_model,
        tokenizer,
        prompts,
        max_new_tokens=max_new_tokens,
        beta=beta,
    )
    print(
        "Before PPO: "
        f"policy={before['policy_reward']:.4f} "
        f"reference={before['reference_reward']:.4f} "
        f"KL={before['mean_policy_kl']:.4f}"
    )

    for iteration in range(1, iterations + 1):
        rollouts = collect_rollouts(
            policy,
            reference,
            reward_model,
            tokenizer,
            prompts,
            max_new_tokens=max_new_tokens,
            beta=beta,
        )
        stats = ppo_update(policy, rollouts, optimizer, tokenizer)
        mean_reward = sum(rollout.task_reward for rollout in rollouts) / len(rollouts)
        mean_kl = sum(sum(rollout.token_kl) for rollout in rollouts) / len(rollouts)
        print(
            f"PPO {iteration:03d} reward {mean_reward:.4f} KL {mean_kl:.4f} "
            f"loss {stats['loss']:.4f} entropy {stats['entropy']:.4f}"
        )

    after = compare_to_reference(
        policy,
        reference,
        reward_model,
        tokenizer,
        prompts,
        max_new_tokens=max_new_tokens,
        beta=beta,
    )
    print(
        "After PPO: "
        f"policy={after['policy_reward']:.4f} "
        f"reference={after['reference_reward']:.4f} "
        f"win_rate={after['policy_win_rate']:.3f} "
        f"KL={after['mean_policy_kl']:.4f}"
    )

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": policy.state_dict(),
            "vocab": tokenizer.vocab,
            "config": {
                "embed": policy.embedding.embedding_dim,
                "hidden": policy.lstm.hidden_size,
            },
        },
        output_path,
    )
    print(f"Saved PPO policy to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft-checkpoint", default="artifacts/sft_policy.pt")
    parser.add_argument("--reward-checkpoint", default="artifacts/reward_model.pt")
    parser.add_argument("--output", default="artifacts/ppo_policy.pt")
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--beta", type=float, default=0.02)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    train_ppo(
        args.sft_checkpoint,
        args.reward_checkpoint,
        args.output,
        args.iterations,
        args.learning_rate,
        args.beta,
        args.max_new_tokens,
        args.seed,
    )


if __name__ == "__main__":
    main()
