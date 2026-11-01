import copy
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F


@dataclass
class Rollout:
    prompt: str
    prompt_ids: list[int]
    response_ids: list[int]
    old_logprobs: list[float]
    reference_logprobs: list[float]
    values: list[float]
    advantages: list[float] = field(default_factory=list)
    returns: list[float] = field(default_factory=list)
    task_reward: float = 0.0
    token_kl: list[float] = field(default_factory=list)


def make_reference(policy):
    """Freeze a copy of the current policy as the KL reference."""
    reference = copy.deepcopy(policy)
    reference.eval()
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    return reference


def compute_gae(rewards, values, gamma=1.0, lam=0.95):
    """Generalized advantage estimation for one variable-length trajectory."""
    advantages = [0.0] * len(rewards)
    next_advantage = 0.0
    for step in reversed(range(len(rewards))):
        next_value = values[step + 1] if step + 1 < len(values) else 0.0
        delta = rewards[step] + gamma * next_value - values[step]
        next_advantage = delta + gamma * lam * next_advantage
        advantages[step] = next_advantage
    returns = [advantage + value for advantage, value in zip(advantages, values)]
    return advantages, returns


@torch.no_grad()
def collect_rollouts(
    policy,
    reference,
    reward_model,
    tokenizer,
    prompts,
    max_new_tokens=24,
    beta=0.02,
    greedy=False,
):
    """Generate responses and record the data required by PPO."""
    was_training = policy.training
    policy.eval()
    reference.eval()
    reward_model.eval()
    rollouts = []

    for prompt in prompts:
        prompt_ids = tokenizer.encode(prompt, add_bos=True)
        sequence = list(prompt_ids)
        response_ids = []
        old_logprobs = []
        reference_logprobs = []
        values = []

        for _ in range(max_new_tokens):
            input_ids = torch.tensor([sequence], dtype=torch.long)
            logits, value_estimates = policy(input_ids)
            reference_logits, _ = reference(input_ids)
            policy_logits = logits[0, -1]
            reference_logits = reference_logits[0, -1]

            distribution = torch.distributions.Categorical(logits=policy_logits)
            action = (
                torch.argmax(policy_logits)
                if greedy
                else distribution.sample()
            )
            old_logprobs.append(distribution.log_prob(action).item())
            reference_logprobs.append(
                F.log_softmax(reference_logits, dim=-1)[action].item()
            )
            values.append(value_estimates[0, -1].item())
            response_ids.append(action.item())
            sequence.append(action.item())
            if action.item() == tokenizer.eos_id:
                break

        reward_ids = prompt_ids + response_ids
        if reward_ids[-1] != tokenizer.eos_id:
            reward_ids.append(tokenizer.eos_id)
        task_reward = reward_model(torch.tensor([reward_ids], dtype=torch.long))[0, 0].item()

        token_kl = [
            policy_logprob - reference_logprob
            for policy_logprob, reference_logprob in zip(
                old_logprobs, reference_logprobs
            )
        ]
        token_rewards = [-beta * kl for kl in token_kl]
        token_rewards[-1] += task_reward
        advantages, returns = compute_gae(token_rewards, values)
        rollouts.append(
            Rollout(
                prompt=prompt,
                prompt_ids=prompt_ids,
                response_ids=response_ids,
                old_logprobs=old_logprobs,
                reference_logprobs=reference_logprobs,
                values=values,
                advantages=advantages,
                returns=returns,
                task_reward=task_reward,
                token_kl=token_kl,
            )
        )

    if was_training:
        policy.train()
    return rollouts


def build_ppo_batch(rollouts, pad_id):
    batch_size = len(rollouts)
    max_response = max(len(rollout.response_ids) for rollout in rollouts)
    max_sequence = max(
        len(rollout.prompt_ids) + len(rollout.response_ids)
        for rollout in rollouts
    )
    input_ids = torch.full((batch_size, max_sequence), pad_id, dtype=torch.long)
    actions = torch.full((batch_size, max_response), pad_id, dtype=torch.long)
    positions = torch.zeros((batch_size, max_response), dtype=torch.long)
    action_mask = torch.zeros((batch_size, max_response), dtype=torch.bool)
    old_logprobs = torch.zeros((batch_size, max_response))
    reference_logprobs = torch.zeros((batch_size, max_response))
    advantages = torch.zeros((batch_size, max_response))
    returns = torch.zeros((batch_size, max_response))

    for row, rollout in enumerate(rollouts):
        sequence = rollout.prompt_ids + rollout.response_ids
        length = len(rollout.response_ids)
        input_ids[row, :len(sequence)] = torch.tensor(sequence)
        actions[row, :length] = torch.tensor(rollout.response_ids)
        positions[row, :length] = torch.arange(
            len(rollout.prompt_ids) - 1,
            len(rollout.prompt_ids) - 1 + length,
        )
        action_mask[row, :length] = True
        old_logprobs[row, :length] = torch.tensor(rollout.old_logprobs)
        reference_logprobs[row, :length] = torch.tensor(
            rollout.reference_logprobs
        )
        advantages[row, :length] = torch.tensor(rollout.advantages)
        returns[row, :length] = torch.tensor(rollout.returns)

    return (
        input_ids,
        actions,
        positions,
        action_mask,
        old_logprobs,
        reference_logprobs,
        advantages,
        returns,
    )


def ppo_update(
    policy,
    rollouts,
    optimizer,
    tokenizer,
    epochs=4,
    clip_ratio=0.2,
    value_coef=0.5,
    entropy_coef=0.01,
    max_grad_norm=1.0,
):
    """Run several PPO epochs on one on-policy rollout batch."""
    (
        input_ids,
        actions,
        positions,
        action_mask,
        old_logprobs,
        _reference_logprobs,
        advantages,
        returns,
    ) = build_ppo_batch(rollouts, tokenizer.pad_id)

    active_advantages = advantages[action_mask]
    advantages = (advantages - active_advantages.mean()) / (
        active_advantages.std(unbiased=False) + 1e-8
    )
    policy.train()
    stats = {}

    for _ in range(epochs):
        logits, values = policy(input_ids)
        vocab_size = logits.size(-1)
        action_logits = logits.gather(
            1,
            positions.unsqueeze(-1).expand(-1, -1, vocab_size),
        )
        log_probs = F.log_softmax(action_logits, dim=-1)
        new_logprobs = log_probs.gather(
            2, actions.unsqueeze(-1)
        ).squeeze(-1)
        action_values = values.gather(1, positions)

        ratio = torch.exp(new_logprobs - old_logprobs)
        surrogate = torch.minimum(
            ratio * advantages,
            ratio.clamp(1.0 - clip_ratio, 1.0 + clip_ratio) * advantages,
        )
        mask_count = action_mask.sum()
        policy_loss = -(surrogate * action_mask).sum() / mask_count
        value_loss = 0.5 * (
            (action_values - returns).pow(2) * action_mask
        ).sum() / mask_count
        entropy = -(log_probs.exp() * log_probs).sum(dim=-1)
        entropy_bonus = (entropy * action_mask).sum() / mask_count
        loss = (
            policy_loss
            + value_coef * value_loss
            - entropy_coef * entropy_bonus
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
        optimizer.step()
        stats = {
            "loss": loss.item(),
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy_bonus.item(),
        }

    return stats


@torch.no_grad()
def compare_to_reference(
    policy,
    reference,
    reward_model,
    tokenizer,
    prompts,
    max_new_tokens=24,
    beta=0.02,
):
    """Compare deterministic policy and reference responses with the reward model."""
    policy_rollouts = collect_rollouts(
        policy,
        reference,
        reward_model,
        tokenizer,
        prompts,
        max_new_tokens=max_new_tokens,
        beta=beta,
        greedy=True,
    )
    reference_rollouts = collect_rollouts(
        reference,
        reference,
        reward_model,
        tokenizer,
        prompts,
        max_new_tokens=max_new_tokens,
        beta=beta,
        greedy=True,
    )
    policy_rewards = torch.tensor([rollout.task_reward for rollout in policy_rollouts])
    reference_rewards = torch.tensor(
        [rollout.task_reward for rollout in reference_rollouts]
    )
    return {
        "policy_reward": policy_rewards.mean().item(),
        "reference_reward": reference_rewards.mean().item(),
        "policy_win_rate": (policy_rewards > reference_rewards).float().mean().item(),
        "mean_policy_kl": torch.tensor(
            [sum(rollout.token_kl) for rollout in policy_rollouts]
        ).mean().item(),
        "policy_responses": [
            tokenizer.decode(rollout.response_ids) for rollout in policy_rollouts
        ],
        "reference_responses": [
            tokenizer.decode(rollout.response_ids) for rollout in reference_rollouts
        ],
    }
