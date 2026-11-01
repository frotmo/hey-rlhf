"""Evaluate DPO implicit rewards on held-out preference pairs."""

import argparse
import json
import math
import statistics
from pathlib import Path

import torch
import torch.nn.functional as F

from evaluate_hf_policy import load_causal_model, load_tokenizer


def load_preferences(path, limit=None):
    records = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if not all(
                isinstance(record.get(field), str)
                and record[field].strip()
                for field in ("prompt", "chosen", "rejected")
            ):
                raise ValueError(f"Invalid preference record in {path}")
            records.append(record)
    return records[:limit] if limit else records


def fallback_assistant_mask(tokenizer, prompt, response, max_length):
    prompt_text = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    full_text = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ],
        tokenize=False,
        add_generation_prompt=False,
    )
    prompt_ids = tokenizer(
        prompt_text,
        add_special_tokens=False,
    )["input_ids"]
    encoded = tokenizer(
        full_text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_length,
    )
    input_ids = encoded["input_ids"]
    common = 0
    while (
        common < min(len(prompt_ids), len(input_ids))
        and prompt_ids[common] == input_ids[common]
    ):
        common += 1
    assistant_mask = [False] * common + [True] * (len(input_ids) - common)
    truncated = len(input_ids) >= max_length
    return input_ids, assistant_mask, truncated


def encode_preference(tokenizer, prompt, response, max_length):
    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response},
    ]
    try:
        encoded = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
            return_assistant_tokens_mask=True,
            truncation=True,
            max_length=max_length,
        )
    except (TypeError, ValueError):
        return fallback_assistant_mask(
            tokenizer,
            prompt,
            response,
            max_length,
        )

    input_ids = encoded["input_ids"]
    assistant_mask = encoded.get("assistant_masks")
    if assistant_mask is None:
        assistant_mask = encoded.get("assistant_mask")
    if assistant_mask is None or not any(assistant_mask):
        return fallback_assistant_mask(
            tokenizer,
            prompt,
            response,
            max_length,
        )
    return input_ids, [bool(value) for value in assistant_mask], (
        len(input_ids) >= max_length
    )


def response_logprob(model, input_ids, assistant_mask, device):
    ids = torch.tensor([input_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(ids)
    mask = torch.tensor([assistant_mask], dtype=torch.bool, device=device)
    with torch.inference_mode():
        logits = model(
            input_ids=ids,
            attention_mask=attention_mask,
            use_cache=False,
        ).logits[:, :-1].float()
    log_probs = torch.log_softmax(logits, dim=-1)
    targets = ids[:, 1:]
    token_log_probs = log_probs.gather(
        2,
        targets.unsqueeze(-1),
    ).squeeze(-1)
    response_mask = mask[:, 1:]
    token_count = response_mask.sum().clamp_min(1)
    total = (token_log_probs * response_mask).sum()
    return {
        "sum_logprob": total.item(),
        "mean_logprob": (total / token_count).item(),
        "token_count": int(token_count.item()),
    }


def summarize_preference_records(records, metadata=None):
    def values(name):
        return [float(record[name]) for record in records]

    def stats(items):
        if not items:
            return {"mean": None, "median": None, "std": None}
        return {
            "mean": statistics.fmean(items),
            "median": statistics.median(items),
            "std": statistics.pstdev(items),
        }

    margins = values("margin")
    correct = sum(
        record["margin"] > 1e-9 for record in records
    )
    tied = sum(
        abs(record["margin"]) <= 1e-9 for record in records
    )
    incorrect = len(records) - correct - tied
    non_tied_count = correct + incorrect
    margin_mean = statistics.fmean(margins) if margins else None
    margin_ci95 = None
    if margins:
        standard_error = statistics.pstdev(margins) / math.sqrt(len(margins))
        margin_ci95 = [
            margin_mean - 1.96 * standard_error,
            margin_mean + 1.96 * standard_error,
        ]

    summary = {
        "pair_count": len(records),
        "pairwise_accuracy": correct / len(records) if records else None,
        "pairwise_accuracy_with_ties_half": (
            (correct + 0.5 * tied) / len(records) if records else None
        ),
        "pairwise_accuracy_non_tied": (
            correct / non_tied_count if non_tied_count else None
        ),
        "tie_rate": tied / len(records) if records else None,
        "zero_margin_count": sum(
            abs(record["margin"]) <= 1e-9 for record in records
        ),
        "margin": {
            **stats(margins),
            "ci95": margin_ci95,
        },
        "dpo_loss": stats(values("dpo_loss")),
        "chosen_implicit_reward": stats(
            values("chosen_implicit_reward")
        ),
        "rejected_implicit_reward": stats(
            values("rejected_implicit_reward")
        ),
        "policy_chosen_mean_logprob": stats(
            values("policy_chosen_mean_logprob")
        ),
        "policy_rejected_mean_logprob": stats(
            values("policy_rejected_mean_logprob")
        ),
        "chosen_token_length": stats(values("chosen_token_length")),
        "rejected_token_length": stats(values("rejected_token_length")),
        "truncated_count": sum(record["truncated"] for record in records),
        "truncated_rate": (
            sum(record["truncated"] for record in records) / len(records)
            if records
            else None
        ),
    }
    non_truncated = [record for record in records if not record["truncated"]]
    non_truncated_margins = [float(record["margin"]) for record in non_truncated]
    summary["non_truncated"] = {
        "pair_count": len(non_truncated),
        "pairwise_accuracy": (
            sum(record["margin"] > 1e-9 for record in non_truncated)
            / len(non_truncated)
            if non_truncated
            else None
        ),
        "margin": stats(non_truncated_margins),
        "dpo_loss": stats(
            [float(record["dpo_loss"]) for record in non_truncated]
        ),
    }
    if metadata:
        summary["metadata"] = metadata
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-model", default="artifacts/local_dpo_smoke_v2")
    parser.add_argument(
        "--reference-model",
        default="HuggingFaceTB/SmolLM2-135M-Instruct",
    )
    parser.add_argument(
        "--preferences-file",
        default="data/low_memory/preference_validation.jsonl",
    )
    parser.add_argument("--output-dir", default="artifacts/evaluations/dpo_validation")
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for DPO preference evaluation")

    device = torch.device(args.device)
    preferences = load_preferences(args.preferences_file, args.limit)
    if not preferences:
        raise ValueError(f"No preference pairs found in {args.preferences_file}")

    tokenizer = load_tokenizer(args.policy_model)
    policy = load_causal_model(args.policy_model, device)
    reference = load_causal_model(args.reference_model, device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records_path = output_dir / "records.jsonl"
    summary_path = output_dir / "summary.json"
    output_records = []

    with records_path.open("w", encoding="utf-8") as handle:
        for index, preference in enumerate(preferences):
            prompt = preference["prompt"]
            chosen = preference["chosen"]
            rejected = preference["rejected"]
            chosen_ids, chosen_mask, chosen_truncated = encode_preference(
                tokenizer,
                prompt,
                chosen,
                args.max_length,
            )
            rejected_ids, rejected_mask, rejected_truncated = encode_preference(
                tokenizer,
                prompt,
                rejected,
                args.max_length,
            )
            policy_chosen = response_logprob(
                policy,
                chosen_ids,
                chosen_mask,
                device,
            )
            policy_rejected = response_logprob(
                policy,
                rejected_ids,
                rejected_mask,
                device,
            )
            reference_chosen = response_logprob(
                reference,
                chosen_ids,
                chosen_mask,
                device,
            )
            reference_rejected = response_logprob(
                reference,
                rejected_ids,
                rejected_mask,
                device,
            )

            chosen_implicit_reward = args.beta * (
                policy_chosen["sum_logprob"]
                - reference_chosen["sum_logprob"]
            )
            rejected_implicit_reward = args.beta * (
                policy_rejected["sum_logprob"]
                - reference_rejected["sum_logprob"]
            )
            margin = chosen_implicit_reward - rejected_implicit_reward
            dpo_loss = -F.logsigmoid(torch.tensor(margin)).item()
            record = {
                "index": index,
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
                "chosen_implicit_reward": chosen_implicit_reward,
                "rejected_implicit_reward": rejected_implicit_reward,
                "margin": margin,
                "correct": margin > 0,
                "dpo_loss": dpo_loss,
                "policy_chosen_mean_logprob": policy_chosen["mean_logprob"],
                "policy_rejected_mean_logprob": policy_rejected["mean_logprob"],
                "reference_chosen_mean_logprob": reference_chosen["mean_logprob"],
                "reference_rejected_mean_logprob": reference_rejected["mean_logprob"],
                "chosen_token_length": policy_chosen["token_count"],
                "rejected_token_length": policy_rejected["token_count"],
                "chosen_char_length": len(chosen),
                "rejected_char_length": len(rejected),
                "truncated": chosen_truncated or rejected_truncated,
            }
            output_records.append(record)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(
                f"PAIR {index + 1:04d}/{len(preferences)} "
                f"accuracy-so-far "
                f"{sum(item['correct'] for item in output_records) / len(output_records):.3f} "
                f"margin {margin:.4f}"
            )

    metadata = {
        "policy_model": args.policy_model,
        "reference_model": args.reference_model,
        "preferences_file": args.preferences_file,
        "beta": args.beta,
        "max_length": args.max_length,
    }
    summary = summarize_preference_records(output_records, metadata)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Wrote records to {records_path}")
    print(f"Wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
