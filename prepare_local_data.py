"""Create desktop-sized JSONL slices without downloading full datasets."""

import argparse
import hashlib
import json
from pathlib import Path

from datasets import load_dataset


def prompt_bucket(prompt):
    digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % 10


def split_name(prompt, data_config):
    bucket = prompt_bucket(prompt)
    train_end = int(data_config["train_ratio"] * 10)
    validation_end = train_end + int(data_config["validation_ratio"] * 10)
    if bucket < train_end:
        return "train"
    if bucket < validation_end:
        return "validation"
    return "test"


def message_text(value):
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return value[-1]["content"]
    raise ValueError(f"Unsupported message value: {type(value)}")


def sft_record(row):
    messages = row["messages"]
    prompt = next(
        message["content"] for message in messages if message["role"] == "user"
    )
    response = next(
        message["content"]
        for message in reversed(messages)
        if message["role"] == "assistant"
    )
    return {"prompt": prompt, "response": response}


def preference_record(row):
    return {
        "prompt": str(row["prompt"]),
        "chosen": message_text(row["chosen"]),
        "rejected": message_text(row["rejected"]),
    }


def has_text(record, fields):
    return all(
        isinstance(record[field], str) and record[field].strip()
        for field in fields
    )


def write_jsonl(path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def unique_prompts(rows):
    return list(dict.fromkeys(row["prompt"] for row in rows))


def stable_prompt_order(prompts):
    return sorted(
        prompts,
        key=lambda prompt: hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/low_memory.json")
    parser.add_argument("--output-dir", default="data/low_memory")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as handle:
        config = json.load(handle)
    data = config["data"]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sft = load_dataset(
        data["sft_dataset"],
        split=f"{data['sft_split']}[:{data['sft_examples']}]",
    )
    preferences = load_dataset(
        data["preference_dataset"],
        split=f"{data['preference_split']}[:{data['preference_examples']}]",
    )
    eval_source_examples = data.get(
        "eval_source_examples",
        data["preference_examples"],
    )
    if eval_source_examples == data["preference_examples"]:
        eval_preferences = preferences
    else:
        eval_preferences = load_dataset(
            data["preference_dataset"],
            split=f"{data['preference_split']}[:{eval_source_examples}]",
        )

    skipped_sft = 0
    sft_rows = {}
    for row in sft:
        record = sft_record(row)
        if not has_text(record, ("prompt", "response")):
            skipped_sft += 1
            continue
        sft_rows.setdefault(split_name(record["prompt"], data), []).append(record)

    skipped_preferences = 0
    preference_rows = {}
    for row in preferences:
        record = preference_record(row)
        if (
            not has_text(record, ("prompt", "chosen", "rejected"))
            or record["chosen"] == record["rejected"]
        ):
            skipped_preferences += 1
            continue
        preference_rows.setdefault(
            split_name(record["prompt"], data), []
        ).append(record)

    skipped_eval_preferences = 0
    eval_preference_rows = {}
    for row in eval_preferences:
        record = preference_record(row)
        if (
            not has_text(record, ("prompt", "chosen", "rejected"))
            or record["chosen"] == record["rejected"]
        ):
            skipped_eval_preferences += 1
            continue
        eval_preference_rows.setdefault(
            split_name(record["prompt"], data), []
        ).append(record)

    for split, rows in sft_rows.items():
        write_jsonl(output_dir / f"sft_{split}.jsonl", rows)
    for split, rows in preference_rows.items():
        write_jsonl(output_dir / f"preference_{split}.jsonl", rows)

    train_prompts = unique_prompts(preference_rows.get("train", []))[
        :data["prompt_examples"]
    ]
    write_jsonl(
        output_dir / "grpo_prompts.jsonl",
        [{"prompt": prompt} for prompt in train_prompts],
    )
    train_prompt_set = set(unique_prompts(preference_rows.get("train", [])))
    eval_candidates = unique_prompts(
        eval_preference_rows.get("validation", [])
        + eval_preference_rows.get("test", [])
    )
    eval_prompts = [
        prompt
        for prompt in stable_prompt_order(eval_candidates)
        if prompt not in train_prompt_set
    ][:data.get("eval_prompt_examples", 200)]
    if train_prompt_set.intersection(eval_prompts):
        raise RuntimeError("Evaluation prompts overlap with training prompts")
    write_jsonl(
        output_dir / "eval_prompts.jsonl",
        [{"prompt": prompt} for prompt in eval_prompts],
    )
    print(
        f"Wrote slices to {output_dir}: "
        f"{sum(map(len, sft_rows.values()))} SFT, "
        f"{sum(map(len, preference_rows.values()))} preferences, "
        f"{len(train_prompts)} GRPO prompts, "
        f"{len(eval_prompts)} eval prompts; "
        f"skipped {skipped_sft} SFT, "
        f"{skipped_preferences} preferences, "
        f"{skipped_eval_preferences} eval preferences"
    )


if __name__ == "__main__":
    main()
