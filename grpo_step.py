import torch
import torch.nn.functional as F

from ppo_step import build_ppo_batch, collect_rollouts


def collect_grpo_groups(
    policy,
    reference,
    reward_model,
    tokenizer,
    prompts,
    group_size=4,
    max_new_tokens=16,
):
    groups = []
    for prompt in prompts:
        groups.append(
            collect_rollouts(
                policy,
                reference,
                reward_model,
                tokenizer,
                [prompt] * group_size,
                max_new_tokens=max_new_tokens,
                beta=0.0,
            )
        )
    return groups


def group_advantages(groups):
    advantages = []
    for group in groups:
        rewards = torch.tensor([rollout.task_reward for rollout in group])
        normalized = (rewards - rewards.mean()) / (rewards.std(unbiased=False) + 1e-8)
        advantages.append(normalized)
    return advantages


def grpo_update(
    policy,
    groups,
    optimizer,
    tokenizer,
    epochs=4,
    beta=0.02,
    clip_ratio=0.2,
    entropy_coef=0.01,
    max_grad_norm=1.0,
):
    rollouts = [rollout for group in groups for rollout in group]
    (
        input_ids,
        actions,
        positions,
        action_mask,
        old_logprobs,
        reference_logprobs,
        _advantages,
        _returns,
    ) = build_ppo_batch(rollouts, tokenizer.pad_id)

    advantages = torch.zeros_like(action_mask, dtype=torch.float32)
    normalized_groups = group_advantages(groups)
    row = 0
    for group, group_advantage in zip(groups, normalized_groups):
        for group_row in range(len(group)):
            length = len(group[group_row].response_ids)
            advantages[row, :length] = group_advantage[group_row]
            row += 1

    policy.train()
    stats = {}
    mask_count = action_mask.sum()
    for _ in range(epochs):
        logits, _values = policy(input_ids)
        action_logits = logits.gather(
            1,
            positions.unsqueeze(-1).expand(-1, -1, logits.size(-1)),
        )
        log_probs = F.log_softmax(action_logits, dim=-1)
        new_logprobs = log_probs.gather(
            2, actions.unsqueeze(-1)
        ).squeeze(-1)
        ratio = torch.exp(new_logprobs - old_logprobs)
        surrogate = torch.minimum(
            ratio * advantages,
            ratio.clamp(1.0 - clip_ratio, 1.0 + clip_ratio) * advantages,
        )
        policy_loss = -(surrogate * action_mask).sum() / mask_count
        token_kl = new_logprobs - reference_logprobs
        kl_loss = beta * (token_kl * action_mask).sum() / mask_count
        entropy = -(log_probs.exp() * log_probs).sum(dim=-1)
        entropy_bonus = (entropy * action_mask).sum() / mask_count
        loss = policy_loss + kl_loss - entropy_coef * entropy_bonus

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
        optimizer.step()
        stats = {
            "loss": loss.item(),
            "policy_loss": policy_loss.item(),
            "kl": kl_loss.item(),
            "entropy": entropy_bonus.item(),
        }
    return stats
