import argparse
from pathlib import Path

import torch

from checkpoints import load_policy, load_reward
from grpo_step import collect_grpo_groups, grpo_update
from ppo_step import compare_to_reference, make_reference
from toy_dataset import prompts


def train_grpo(
    sft_path="artifacts/sft_policy.pt",
    reward_path="artifacts/reward_model.pt",
    output="artifacts/grpo_policy.pt",
    iterations=20,
    group_size=4,
    learning_rate=3e-4,
    beta=0.02,
    max_new_tokens=16,
    seed=31,
):
    torch.manual_seed(seed)
    policy, tokenizer, checkpoint = load_policy(sft_path)
    reward_model = load_reward(reward_path, tokenizer)
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
        "Before GRPO: "
        f"policy={before['policy_reward']:.4f} "
        f"reference={before['reference_reward']:.4f}"
    )

    for iteration in range(1, iterations + 1):
        groups = collect_grpo_groups(
            policy,
            reference,
            reward_model,
            tokenizer,
            prompts,
            group_size=group_size,
            max_new_tokens=max_new_tokens,
        )
        stats = grpo_update(
            policy,
            groups,
            optimizer,
            tokenizer,
            beta=beta,
        )
        rollouts = [rollout for group in groups for rollout in group]
        mean_reward = sum(item.task_reward for item in rollouts) / len(rollouts)
        print(
            f"GRPO {iteration:03d} reward {mean_reward:.4f} "
            f"loss {stats['loss']:.4f} KL {stats['kl']:.4f}"
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
        "After GRPO: "
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
            "config": checkpoint["config"],
        },
        output_path,
    )
    print(f"Saved GRPO policy to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sft-checkpoint", default="artifacts/sft_policy.pt")
    parser.add_argument("--reward-checkpoint", default="artifacts/reward_model.pt")
    parser.add_argument("--output", default="artifacts/grpo_policy.pt")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--beta", type=float, default=0.02)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--seed", type=int, default=31)
    args = parser.parse_args()
    train_grpo(
        args.sft_checkpoint,
        args.reward_checkpoint,
        args.output,
        args.iterations,
        args.group_size,
        args.learning_rate,
        args.beta,
        args.max_new_tokens,
        args.seed,
    )


if __name__ == "__main__":
    main()
