"""GPU-scale PPO extension. This entry point is intentionally not tested locally.

Default footprint: SmolLM2-135M policy + frozen reference + a DeBERTa reward
model. It targets a high-memory GPU profile, but only a real run can confirm the
tokenizer, reward checkpoint, and memory behavior for a chosen model revision.
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification
from transformers import AutoTokenizer

from ppo_step import Rollout, compute_gae, make_reference
from toy_dataset import prompts as toy_prompts


class ActorCritic(torch.nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        base_parameter = next(base_model.parameters())
        self.value_head = torch.nn.Linear(
            base_model.config.hidden_size,
            1,
            device=base_parameter.device,
            dtype=base_parameter.dtype,
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )
        hidden = outputs.hidden_states[-1]
        return outputs.logits, self.value_head(hidden).squeeze(-1)


def load_prompts(path):
    if path is None:
        return toy_prompts
    with open(path, "r", encoding="utf-8") as handle:
        return [
            json.loads(line)["prompt"]
            for line in handle
            if line.strip()
        ]


def format_prompt(tokenizer, prompt):
    if tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return prompt


def left_pad_sequences(sequences, pad_id, device):
    max_length = max(len(sequence) for sequence in sequences)
    input_ids = torch.full(
        (len(sequences), max_length),
        pad_id,
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.zeros_like(input_ids)
    for row, sequence in enumerate(sequences):
        input_ids[row, -len(sequence):] = torch.tensor(
            sequence, dtype=torch.long, device=device
        )
        attention_mask[row, -len(sequence):] = 1
    return input_ids, attention_mask


@torch.no_grad()
def collect_gpu_rollouts(
    policy,
    reference,
    reward_model,
    actor_tokenizer,
    reward_tokenizer,
    prompts,
    device,
    max_new_tokens=48,
    beta=0.02,
):
    policy.eval()
    reference.eval()
    reward_model.eval()
    formatted_prompts = [format_prompt(actor_tokenizer, prompt) for prompt in prompts]
    sequences = [actor_tokenizer.encode(text) for text in formatted_prompts]
    prompt_lengths = [len(sequence) for sequence in sequences]
    response_ids = [[] for _ in prompts]
    old_logprobs = [[] for _ in prompts]
    reference_logprobs = [[] for _ in prompts]
    values = [[] for _ in prompts]
    active = [True] * len(prompts)

    for _ in range(max_new_tokens):
        if not any(active):
            break
        input_ids, attention_mask = left_pad_sequences(
            sequences, actor_tokenizer.pad_token_id, device
        )
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits, value_estimates = policy(input_ids, attention_mask)
            reference_logits, _ = reference(input_ids, attention_mask)
        policy_logits = logits[:, -1].float()
        reference_logits = reference_logits[:, -1].float()
        distributions = torch.distributions.Categorical(logits=policy_logits)
        actions = distributions.sample()
        policy_logprobs = distributions.log_prob(actions)
        reference_logprobs_step = F.log_softmax(reference_logits, dim=-1).gather(
            1, actions.unsqueeze(-1)
        ).squeeze(-1)

        for row, is_active in enumerate(active):
            if not is_active:
                continue
            action = actions[row].item()
            response_ids[row].append(action)
            old_logprobs[row].append(policy_logprobs[row].item())
            reference_logprobs[row].append(reference_logprobs_step[row].item())
            values[row].append(value_estimates[row, -1].float().item())
            sequences[row].append(action)
            if action == actor_tokenizer.eos_token_id:
                active[row] = False

    reward_texts = []
    for prompt, response in zip(formatted_prompts, response_ids):
        response_text = actor_tokenizer.decode(response, skip_special_tokens=True)
        reward_texts.append(f"{prompt}\n\n{response_text}")
    reward_inputs = reward_tokenizer(
        reward_texts,
        padding=True,
        truncation=True,
        max_length=2048,
        return_tensors="pt",
    ).to(device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        task_rewards = reward_model(**reward_inputs).logits[:, 0].float()

    rollouts = []
    for row in range(len(prompts)):
        token_kl = [
            policy_logprob - reference_logprob
            for policy_logprob, reference_logprob in zip(
                old_logprobs[row], reference_logprobs[row]
            )
        ]
        token_rewards = [-beta * kl for kl in token_kl]
        token_rewards[-1] += task_rewards[row].item()
        advantages, returns = compute_gae(token_rewards, values[row])
        rollouts.append(
            Rollout(
                prompt=prompts[row],
                prompt_ids=sequences[row][:prompt_lengths[row]],
                response_ids=response_ids[row],
                old_logprobs=old_logprobs[row],
                reference_logprobs=reference_logprobs[row],
                values=values[row],
                advantages=advantages,
                returns=returns,
                task_reward=task_rewards[row].item(),
                token_kl=token_kl,
            )
        )
    return rollouts


def build_update_batch(rollouts, pad_id, device):
    batch_size = len(rollouts)
    max_response = max(len(rollout.response_ids) for rollout in rollouts)
    max_sequence = max(
        len(rollout.prompt_ids) + len(rollout.response_ids)
        for rollout in rollouts
    )
    input_ids = torch.full(
        (batch_size, max_sequence), pad_id, dtype=torch.long, device=device
    )
    attention_mask = torch.zeros_like(input_ids)
    actions = torch.zeros((batch_size, max_response), dtype=torch.long, device=device)
    positions = torch.zeros((batch_size, max_response), dtype=torch.long, device=device)
    action_mask = torch.zeros((batch_size, max_response), dtype=torch.bool, device=device)
    old_logprobs = torch.zeros((batch_size, max_response), device=device)
    advantages = torch.zeros((batch_size, max_response), device=device)
    returns = torch.zeros((batch_size, max_response), device=device)

    for row, rollout in enumerate(rollouts):
        sequence = rollout.prompt_ids + rollout.response_ids
        length = len(rollout.response_ids)
        input_ids[row, :len(sequence)] = torch.tensor(sequence, device=device)
        attention_mask[row, :len(sequence)] = 1
        actions[row, :length] = torch.tensor(rollout.response_ids, device=device)
        positions[row, :length] = torch.arange(
            len(rollout.prompt_ids) - 1,
            len(rollout.prompt_ids) - 1 + length,
            device=device,
        )
        action_mask[row, :length] = True
        old_logprobs[row, :length] = torch.tensor(
            rollout.old_logprobs, device=device
        )
        advantages[row, :length] = torch.tensor(
            rollout.advantages, device=device
        )
        returns[row, :length] = torch.tensor(rollout.returns, device=device)

    return (
        input_ids,
        attention_mask,
        actions,
        positions,
        action_mask,
        old_logprobs,
        advantages,
        returns,
    )


def ppo_update(
    policy,
    rollouts,
    optimizer,
    pad_id,
    device,
    epochs=4,
    clip_ratio=0.2,
    value_coef=0.5,
    entropy_coef=0.01,
):
    (
        input_ids,
        attention_mask,
        actions,
        positions,
        action_mask,
        old_logprobs,
        advantages,
        returns,
    ) = build_update_batch(rollouts, pad_id, device)
    active_advantages = advantages[action_mask]
    advantages = (advantages - active_advantages.mean()) / (
        active_advantages.std(unbiased=False) + 1e-8
    )
    mask_count = action_mask.sum()
    policy.train()

    for _ in range(epochs):
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits, values = policy(input_ids, attention_mask)
            action_logits = logits.gather(
                1,
                positions.unsqueeze(-1).expand(-1, -1, logits.size(-1)),
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
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()

    return {
        "loss": loss.item(),
        "policy_loss": policy_loss.item(),
        "value_loss": value_loss.item(),
        "entropy": entropy_bonus.item(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--policy-model",
        default="HuggingFaceTB/SmolLM2-135M-Instruct",
    )
    parser.add_argument(
        "--reward-model",
        default="OpenAssistant/reward-model-deberta-v3-base",
    )
    parser.add_argument("--reward-tokenizer", default=None)
    parser.add_argument("--prompts-file", default=None)
    parser.add_argument("--output-dir", default="artifacts/scaleup_ppo")
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--beta", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=23)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("scaleup_ppo.py requires CUDA; use train_ppo.py for CPU")
    torch.manual_seed(args.seed)
    device = torch.device("cuda")

    actor_tokenizer = AutoTokenizer.from_pretrained(args.policy_model)
    if actor_tokenizer.pad_token_id is None:
        actor_tokenizer.pad_token = actor_tokenizer.eos_token
    reward_tokenizer = AutoTokenizer.from_pretrained(
        args.reward_tokenizer or args.reward_model
    )

    base_policy = AutoModelForCausalLM.from_pretrained(
        args.policy_model,
        dtype=torch.bfloat16,
    ).to(device)
    policy = ActorCritic(base_policy)
    if hasattr(policy.base_model, "gradient_checkpointing_enable"):
        policy.base_model.gradient_checkpointing_enable()
    reference = make_reference(policy)
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        args.reward_model,
        num_labels=1,
        dtype=torch.bfloat16,
    ).to(device)
    reward_model.eval()
    for parameter in reward_model.parameters():
        parameter.requires_grad_(False)

    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.learning_rate)
    prompts = load_prompts(args.prompts_file)
    for iteration in range(1, args.iterations + 1):
        batch = prompts[(iteration - 1) * args.batch_size:iteration * args.batch_size]
        if len(batch) < args.batch_size:
            batch = prompts[:args.batch_size]
        rollouts = collect_gpu_rollouts(
            policy,
            reference,
            reward_model,
            actor_tokenizer,
            reward_tokenizer,
            batch,
            device,
            max_new_tokens=args.max_new_tokens,
            beta=args.beta,
        )
        stats = ppo_update(
            policy,
            rollouts,
            optimizer,
            actor_tokenizer.pad_token_id,
            device,
        )
        mean_reward = sum(item.task_reward for item in rollouts) / len(rollouts)
        mean_kl = sum(sum(item.token_kl) for item in rollouts) / len(rollouts)
        print(
            f"PPO {iteration:03d} reward {mean_reward:.4f} KL {mean_kl:.4f} "
            f"loss {stats['loss']:.4f}"
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    policy.base_model.save_pretrained(output_dir)
    actor_tokenizer.save_pretrained(output_dir)
    torch.save(policy.value_head.state_dict(), output_dir / "value_head.pt")
    print(f"Saved scaled PPO policy to {output_dir}")


if __name__ == "__main__":
    main()
