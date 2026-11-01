"""Compare a policy against a frozen reference on a fixed prompt set."""

import argparse
import hashlib
import json
import statistics
from pathlib import Path

import torch


def load_prompts(path, limit=None):
    prompts = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            prompts.append(json.loads(line)["prompt"])
    return prompts[:limit] if limit else prompts


def format_prompt(tokenizer, prompt):
    if tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return prompt


def load_tokenizer(model_name, padding_side=None):
    from transformers import AutoTokenizer

    model_path = Path(model_name)
    if model_path.exists() and not (
        (model_path / "config.json").exists()
        or (model_path / "adapter_config.json").exists()
    ):
        raise FileNotFoundError(
            f"Local model directory is incomplete: {model_path}. "
            "Expected config.json or adapter_config.json."
        )
    if not model_path.exists() and (
        model_name.startswith(("/", "./", "../", "artifacts/"))
    ):
        raise FileNotFoundError(
            f"Local model path does not exist: {model_path}. "
            "Check the DPO --output-dir and the directory used by --policy-model."
        )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if padding_side:
        tokenizer.padding_side = padding_side
    return tokenizer


def load_causal_model(model_name, device):
    from transformers import AutoModelForCausalLM

    model_path = Path(model_name)
    if model_path.is_dir() and (model_path / "adapter_config.json").exists():
        from peft import AutoPeftModelForCausalLM

        model = AutoPeftModelForCausalLM.from_pretrained(
            str(model_path),
            dtype=torch.bfloat16,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
        )
    return model.to(device).eval()


def load_reward_model(model_name, device):
    from transformers import AutoModelForSequenceClassification

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=1,
        dtype=torch.bfloat16,
    ).to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def generate_responses(
    model,
    tokenizer,
    prompts,
    device,
    max_new_tokens,
    max_prompt_length,
):
    inputs = tokenizer(
        prompts,
        padding=True,
        truncation=True,
        max_length=max_prompt_length,
        return_tensors="pt",
    ).to(device)
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    input_width = inputs["input_ids"].size(1)
    new_tokens = generated[:, input_width:]
    new_attention = torch.ones_like(new_tokens)
    stop_ids = {
        token_id
        for token_id in (tokenizer.eos_token_id, tokenizer.pad_token_id)
        if token_id is not None
    }
    response_lengths = []
    responses = []
    for row in range(new_tokens.size(0)):
        length = new_tokens.size(1)
        for column, token_id in enumerate(new_tokens[row].tolist()):
            if token_id in stop_ids:
                length = column + 1
                break
        response_lengths.append(length)
        if length < new_tokens.size(1):
            new_attention[row, length:] = 0
        responses.append(
            tokenizer.decode(
                new_tokens[row, :length],
                skip_special_tokens=True,
            )
        )

    attention_mask = torch.cat([inputs["attention_mask"], new_attention], dim=1)
    return responses, generated, attention_mask, input_width, response_lengths


def score_texts(
    reward_model,
    tokenizer,
    texts,
    device,
    batch_size,
    max_length,
):
    scores = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)
        with torch.inference_mode():
            logits = reward_model(**inputs).logits[:, 0].float()
        scores.extend(logits.cpu().tolist())
    return scores


def mean_response_logprobs(
    model,
    input_ids,
    attention_mask,
    response_start,
    response_lengths,
):
    with torch.inference_mode():
        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        ).logits[:, :-1].float()
    log_probs = torch.log_softmax(logits, dim=-1)
    targets = input_ids[:, 1:]
    token_log_probs = log_probs.gather(
        2,
        targets.unsqueeze(-1),
    ).squeeze(-1)

    positions = torch.arange(
        token_log_probs.size(1),
        device=input_ids.device,
    ).unsqueeze(0)
    lengths = torch.tensor(
        response_lengths,
        device=input_ids.device,
    ).unsqueeze(1)
    response_mask = (
        (positions >= response_start - 1)
        & (positions < response_start - 1 + lengths)
        & attention_mask[:, 1:].bool()
    )
    token_counts = response_mask.sum(dim=1).clamp_min(1)
    return (
        (token_log_probs * response_mask).sum(dim=1) / token_counts
    ).cpu().tolist()


def summarize_records(records, metadata=None):
    def metric(name):
        return [float(record[name]) for record in records]

    def stats(values):
        if not values:
            return {"mean": None, "median": None, "std": None}
        return {
            "mean": statistics.fmean(values),
            "median": statistics.median(values),
            "std": statistics.pstdev(values),
        }

    policy_rewards = metric("policy_reward")
    reference_rewards = metric("reference_reward")
    reward_deltas = metric("reward_delta")
    sampled_kl = metric("sampled_kl")
    policy_lengths = metric("policy_token_length")
    reference_lengths = metric("reference_token_length")
    tolerance = 1e-6
    wins = sum(delta > tolerance for delta in reward_deltas)
    ties = sum(abs(delta) <= tolerance for delta in reward_deltas)
    losses = len(reward_deltas) - wins - ties

    summary = {
        "prompt_count": len(records),
        "policy": {
            "reward": stats(policy_rewards),
            "token_length": stats(policy_lengths),
        },
        "reference": {
            "reward": stats(reference_rewards),
            "token_length": stats(reference_lengths),
        },
        "comparison": {
            "reward_delta": stats(reward_deltas),
            "policy_win_rate": wins / len(records) if records else None,
            "tie_rate": ties / len(records) if records else None,
            "policy_loss_rate": losses / len(records) if records else None,
            "sampled_kl": stats(sampled_kl),
            "sampled_kl_abs_mean": (
                statistics.fmean(abs(value) for value in sampled_kl)
                if sampled_kl
                else None
            ),
        },
    }
    if metadata:
        summary["metadata"] = metadata
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-model", default="artifacts/scaleup_ppo")
    parser.add_argument(
        "--reference-model",
        default="HuggingFaceTB/SmolLM2-135M-Instruct",
    )
    parser.add_argument(
        "--reward-model",
        default="OpenAssistant/reward-model-deberta-v3-base",
    )
    parser.add_argument(
        "--prompts-file",
        default="data/low_memory/eval_prompts.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/evaluations/scaleup_ppo",
    )
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-prompt-length", type=int, default=512)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--reward-max-length", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for evaluation")

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    prompts = load_prompts(args.prompts_file, args.limit)
    if not prompts:
        raise ValueError(f"No prompts found in {args.prompts_file}")

    tokenizer = load_tokenizer(args.policy_model, padding_side="left")
    reward_tokenizer = load_tokenizer(args.reward_model)
    policy = load_causal_model(args.policy_model, device)
    reference = load_causal_model(args.reference_model, device)
    reward_model = load_reward_model(args.reward_model, device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / "records.jsonl"
    summary_path = output_dir / "summary.json"
    records = []

    with records_path.open("w", encoding="utf-8") as handle:
        for start in range(0, len(prompts), args.batch_size):
            prompt_batch = prompts[start:start + args.batch_size]
            formatted = [
                format_prompt(tokenizer, prompt)
                for prompt in prompt_batch
            ]
            (
                policy_responses,
                generated_ids,
                generated_attention,
                response_start,
                policy_token_lengths,
            ) = generate_responses(
                policy,
                tokenizer,
                formatted,
                device,
                args.max_new_tokens,
                args.max_prompt_length,
            )
            (
                reference_responses,
                _reference_ids,
                _reference_attention,
                _reference_start,
                reference_token_lengths,
            ) = generate_responses(
                reference,
                tokenizer,
                formatted,
                device,
                args.max_new_tokens,
                args.max_prompt_length,
            )

            policy_logprobs = mean_response_logprobs(
                policy,
                generated_ids,
                generated_attention,
                response_start,
                policy_token_lengths,
            )
            reference_logprobs = mean_response_logprobs(
                reference,
                generated_ids,
                generated_attention,
                response_start,
                policy_token_lengths,
            )
            reward_texts = []
            for formatted_prompt, reference_response, policy_response in zip(
                formatted,
                reference_responses,
                policy_responses,
            ):
                reward_texts.append(
                    f"{formatted_prompt}\n\n{reference_response}"
                )
                reward_texts.append(
                    f"{formatted_prompt}\n\n{policy_response}"
                )
            reward_scores = score_texts(
                reward_model,
                reward_tokenizer,
                reward_texts,
                device,
                args.batch_size * 2,
                args.reward_max_length,
            )
            reference_rewards = reward_scores[0::2]
            policy_rewards = reward_scores[1::2]

            for row, prompt in enumerate(prompt_batch):
                record = {
                    "index": start + row,
                    "prompt": prompt,
                    "formatted_prompt": formatted[row],
                    "reference_response": reference_responses[row],
                    "policy_response": policy_responses[row],
                    "reference_reward": reference_rewards[row],
                    "policy_reward": policy_rewards[row],
                    "reward_delta": (
                        policy_rewards[row] - reference_rewards[row]
                    ),
                    "sampled_kl": (
                        policy_logprobs[row] - reference_logprobs[row]
                    ),
                    "reference_token_length": reference_token_lengths[row],
                    "policy_token_length": policy_token_lengths[row],
                    "reference_char_length": len(reference_responses[row]),
                    "policy_char_length": len(policy_responses[row]),
                }
                records.append(record)
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    metadata = {
        "policy_model": args.policy_model,
        "reference_model": args.reference_model,
        "reward_model": args.reward_model,
        "prompts_file": args.prompts_file,
        "prompts_sha256": hashlib.sha256(
            Path(args.prompts_file).read_bytes()
        ).hexdigest(),
        "decoding": {
            "method": "greedy",
            "batch_size": args.batch_size,
            "max_new_tokens": args.max_new_tokens,
            "max_prompt_length": args.max_prompt_length,
        },
        "seed": args.seed,
    }
    summary = summarize_records(records, metadata)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Wrote records to {records_path}")
    print(f"Wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
