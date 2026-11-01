"""Local 6 GB DPO/GRPO entry point. Not executed in this environment."""

import argparse
import json
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from trl import DPOConfig, DPOTrainer, GRPOConfig, GRPOTrainer
from trl import SFTConfig, SFTTrainer


class RewardScorer:
    def __init__(self, model_name, device):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=1,
            dtype=torch.float16,
        ).to(device)
        self.model.eval()
        self.device = device

    @torch.no_grad()
    def __call__(self, prompts, completions, **kwargs):
        texts = [
            f"{prompt}\n\n{completion}"
            for prompt, completion in zip(prompts, completions)
        ]
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=384,
            return_tensors="pt",
        ).to(self.device)
        return self.model(**inputs).logits[:, 0].float().cpu().tolist()


def load_jsonl(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"JSONL file does not exist: {path}. "
            "Run prepare_local_data.py before training."
        )
    return load_dataset("json", data_files=str(path), split="train")


def build_model_and_tokenizer(config, device):
    tokenizer = AutoTokenizer.from_pretrained(config["policy_model"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        config["policy_model"],
        dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).to(device)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    lora = config["lora"]
    model = get_peft_model(
        model,
        LoraConfig(
            r=lora["rank"],
            lora_alpha=lora["alpha"],
            lora_dropout=lora["dropout"],
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        ),
    )
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--algorithm",
        choices=("sft", "dpo", "grpo"),
        required=True,
    )
    parser.add_argument("--config", default="configs/low_memory.json")
    parser.add_argument("--data-dir", default="data/low_memory")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("train_hf_local.py requires a CUDA GPU")
    with open(args.config, "r", encoding="utf-8") as handle:
        config = json.load(handle)
    training = config["training"]
    max_length = args.max_length or training["max_length"]
    output_dir = Path(args.output_dir or f"artifacts/local_{args.algorithm}")
    data_dir = Path(args.data_dir)
    device = torch.device("cuda")
    model, tokenizer = build_model_and_tokenizer(config, device)
    common = {
        "output_dir": str(output_dir),
        "per_device_train_batch_size": training["batch_size"],
        "gradient_accumulation_steps": training["gradient_accumulation_steps"],
        "max_steps": (
            args.max_steps
            if args.max_steps is not None
            else training["max_steps"][args.algorithm]
        ),
        "logging_steps": 1,
        "save_steps": 50,
        "save_total_limit": 2,
        "gradient_checkpointing": True,
        "fp16": True,
        "report_to": "none",
        "remove_unused_columns": False,
    }

    if args.algorithm == "sft":
        dataset = load_jsonl(data_dir / "sft_train.jsonl")
        dataset = dataset.map(
            lambda row: {
                "text": f"User: {row['prompt']}\nAssistant: {row['response']}"
            }
        )
        trainer = SFTTrainer(
            model=model,
            args=SFTConfig(
                **common,
                learning_rate=training["sft_learning_rate"],
                max_length=max_length,
                dataset_text_field="text",
            ),
            train_dataset=dataset,
            processing_class=tokenizer,
        )
    elif args.algorithm == "dpo":
        dataset = load_jsonl(data_dir / "preference_train.jsonl")
        trainer = DPOTrainer(
            model=model,
            ref_model=None,
            args=DPOConfig(
                **common,
                learning_rate=training["dpo_learning_rate"],
                beta=training["dpo_beta"],
                max_length=max_length,
            ),
            train_dataset=dataset,
            processing_class=tokenizer,
        )
    else:
        dataset = load_jsonl(data_dir / "grpo_prompts.jsonl")
        reward_scorer = RewardScorer(config["reward_model"], device)
        trainer = GRPOTrainer(
            model=model,
            reward_funcs=[reward_scorer],
            args=GRPOConfig(
                **common,
                learning_rate=training["grpo_learning_rate"],
                beta=training["grpo_beta"],
                max_completion_length=training["max_completion_length"],
                num_generations=training["grpo_generations"],
                temperature=1.0,
                use_vllm=False,
            ),
            train_dataset=dataset,
            processing_class=tokenizer,
        )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Saved {args.algorithm.upper()} adapter to {output_dir}")


if __name__ == "__main__":
    main()
