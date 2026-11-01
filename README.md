# RLHF Mini Pipeline

[English](README.md) | [中文](README_zh.md)

A teaching-oriented RLHF project for readers learning deep learning and machine
learning. It keeps the upstream pure-PyTorch route, completes the original
"PPO-style" update as a runnable PPO loop, and adds a frozen reference model,
KL control, preference reward modeling, and policy comparison.

The project has three tracks:

- **CPU algorithm lab:** character/word-level LSTM, toy data, and runnable PPO,
  GRPO, and DPO implementations.
- **Low-memory track:** `SmolLM2-135M-Instruct` with LoRA and small dataset
  slices, targeting roughly 6GB of GPU memory.
- **High-memory extension:** Hugging Face Transformer PPO, targeting roughly
  24GB of GPU memory.

See [DEVELOPMENT.md](DEVELOPMENT.md) for implementation notes and experiment
history. See [ATTRIBUTION.md](ATTRIBUTION.md) for upstream attribution,
licensing status, and the exact differences from the original project.

> This is an educational implementation, not production-grade RLHF. Toy-scale
> rewards are not evidence of real alignment quality.

## 1. What It Implements

The small pipeline covers:

```text
prompt
  |
  v
SFT policy (MiniLM: embedding -> LSTM -> LM head + value head)
  |
  +-------------------- frozen reference copy
  |
  v
sample response -> RewardModel score
  |
  v
token reward = task reward - beta * (log pi - log pi_ref)
  |
  v
GAE -> PPO clipped objective + value loss + entropy bonus
```

Key behavior:

- Bradley-Terry preference loss:
  `-log sigmoid(r_chosen - r_rejected)`.
- PPO records old log-probabilities and uses ratios, clipped surrogates, GAE,
  a value head, entropy regularization, and gradient clipping.
- The reference policy is a frozen copy of the SFT policy.
- Every generated response token can receive a sampled KL penalty; the reward
  model score is added to the final valid token.
- Policy and reference are compared with the same reward model before and after
  PPO.
- DPO uses response-only log-probability differences and needs no rollout.
- GRPO normalizes rewards across multiple responses to the same prompt and does
  not use a value head.

## 2. Repository Layout

```text
mini_lm.py                  LSTM actor-critic policy
reward_model.py             Sequence-level scalar reward model
preference_loss.py          Bradley-Terry preference loss
tokenizer.py                Word tokenizer, special tokens, padding
toy_dataset.py              Toy SFT and preference data
checkpoints.py              Shared checkpoint loading
train_sft.py                SFT / initial policy training
train_reward_model.py       Reward-model training and pairwise accuracy
ppo_step.py                 Rollout, KL, GAE, PPO update, reference comparison
train_ppo.py                PPO training entry point
grpo_step.py                GRPO sampling, group advantages, update
train_grpo.py               GRPO training entry point
dpo_step.py                 Response-only log-probability and DPO loss
train_dpo.py                DPO training entry point
configs/low_memory.json     Low-memory model, data, and step budget
pyproject.toml              Python version, dependencies, local/scale extras
uv.lock                     Reproducible uv dependency lock
.python-version             Python 3.12
prepare_local_data.py       Fixed-size Hugging Face dataset slicing
train_hf_local.py           Low-memory LoRA SFT/DPO/GRPO entry point
scaleup_ppo.py              Hugging Face Transformer PPO extension
evaluate_hf_policy.py       Policy/reference evaluation on fixed prompts
evaluate_dpo_preferences.py Held-out DPO preference validation
run_mini_pipeline.py        CPU SFT -> reward model -> PPO pipeline
tests/                      Algorithm and evaluation tests
DEVELOPMENT.md              Development notes and experiment history
ATTRIBUTION.md              Upstream attribution and change summary
LICENSE                     MIT license
```

## 3. Python Environment

The project uses [uv](https://docs.astral.sh/uv/) for Python, virtual
environments, and dependency management. Python is pinned in `.python-version`;
declared dependencies live in `pyproject.toml`; resolved versions live in
`uv.lock`.

Install uv if needed:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Create the Python 3.12 environment and install the default CPU dependencies:

```bash
uv python install
uv sync
```

Run the complete toy pipeline:

```bash
uv run python run_mini_pipeline.py
```

Select one algorithm:

```bash
uv run python run_mini_pipeline.py --algorithm ppo
uv run python run_mini_pipeline.py --algorithm grpo
uv run python run_mini_pipeline.py --algorithm dpo
```

Or run the stages separately:

```bash
uv run python train_sft.py
uv run python train_reward_model.py
uv run python train_ppo.py
uv run python train_grpo.py
uv run python train_dpo.py
```

Run the tests:

```bash
uv run python -m unittest discover -s tests -v
```

`uv run` uses the project `.venv` automatically. If you prefer a direct
`python` command, activate the environment first:

```bash
source .venv/bin/activate
python run_mini_pipeline.py
```

Activation only applies to the current shell. A direct `python` command usually
points to the system interpreter and cannot see packages installed inside
`.venv`.

You do not need to run `uv sync` before every command. Run it when dependencies
change, `uv.lock` changes, or the virtual environment is missing.

Default toy artifacts are written under `artifacts/`:

```text
sft_policy.pt
reward_model.pt
ppo_policy.pt
grpo_policy.pt
dpo_policy.pt
```

## 4. PPO Equations

For one response trajectory, the reference model supplies a token-level
reference distribution. KL reward shaping is:

```text
token_reward_t = -beta * (log pi(a_t | s_t) - log pi_ref(a_t | s_t))
```

The reward-model score is added to the final valid token. Advantages use GAE:

```text
delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
A_t = delta_t + gamma * lambda * A_{t+1}
```

The clipped PPO surrogate is:

```text
ratio_t = exp(log pi_new(a_t) - log pi_old(a_t))
L_policy = -mean(min(
    ratio_t * A_t,
    clip(ratio_t, 1 - eps, 1 + eps) * A_t
))
```

The total objective also contains value regression and an entropy bonus. A
production RLHF system would additionally need more data, reward calibration,
length-bias controls, distributed training, independent evaluation, and human
audit.

## 5. Algorithm Modes

| Algorithm | Reward model | Rollout | Local status |
|---|---|---|---|
| PPO | Yes | Yes, with value head | Tested |
| GRPO | Yes | Yes, multiple responses per prompt | Tested |
| DPO | No | No | Tested |

GRPO uses the standard group-relative advantage:

```text
A_i = (reward_i - mean(group_reward)) / (std(group_reward) + eps)
```

DPO compares chosen and rejected responses through implicit rewards:

```text
z = beta * (
    (log pi(chosen) - log pi_ref(chosen))
    - (log pi(rejected) - log pi_ref(rejected))
)
L = -log sigmoid(z)
```

The toy CPU experiment raises DPO margin from `0.0468` to `2.0200` and GRPO
sampled reward from `0.0518` to about `0.0618`. These numbers only demonstrate
that the chains run; the toy data is too small to support generalization claims.

## 6. Low-Memory Configuration

[configs/low_memory.json](configs/low_memory.json) uses:

- Base policy: `HuggingFaceTB/SmolLM2-135M-Instruct` (about 134.5M parameters).
- Training: LoRA rank 8, batch size 1, gradient accumulation 8.
- Context: 384 tokens; DPO maximum prompt 160; GRPO maximum completion 224.
- SFT slice: 2,000 examples from `UltraChat 200k`.
- Preference slice: 4,000 pairs from `UltraFeedback Binarized`.
- GRPO prompts: up to 500 unique prompts from the training preference split.
- Step limits: DPO 200, GRPO 80.
- Rough time budget: SFT 0.5-2 hours, DPO 1-3 hours, GRPO 3-8 hours.

This is not full-dataset training. Prepare fixed-size slices first:

```bash
uv sync --extra local
uv run python prepare_local_data.py
```

Run low-memory SFT, DPO, or GRPO:

```bash
uv run python train_hf_local.py --algorithm sft
uv run python train_hf_local.py --algorithm dpo
uv run python train_hf_local.py --algorithm grpo
```

Smoke-test DPO with a short run and longer context:

```bash
uv run python train_hf_local.py \
  --algorithm dpo \
  --max-steps 20 \
  --max-length 1024 \
  --output-dir artifacts/local_dpo_smoke
```

Then run an intermediate experiment:

```bash
uv run python train_hf_local.py \
  --algorithm dpo \
  --max-steps 100 \
  --max-length 1024 \
  --output-dir artifacts/local_dpo_100
```

The repository does not include datasets, model weights, checkpoints, or
automated GPU integration tests. Always start with a smoke test and verify
memory, loss, and adapter saving before a full run.

## 7. Reference Model

The reference model is a frozen copy of the SFT policy. It serves two roles:

1. A KL anchor that limits how far PPO moves from the initial language policy.
2. A baseline evaluated with the same reward model as the policy.

The CPU example intentionally avoids downloading a separate language model.
A frozen SFT copy is algorithmically valid and keeps the teaching version
self-contained.

## 8. High-Memory Transformer Extension

`scaleup_ppo.py` extends the small PPO path to Hugging Face causal language
models:

- Default policy/reference: `HuggingFaceTB/SmolLM2-135M-Instruct`
- Default reward model: `OpenAssistant/reward-model-deberta-v3-base`
- Custom policy/reward models, JSONL prompts, and batch sizes are supported.
- BF16, gradient checkpointing, GAE, KL penalty, and PPO clipping are used.

Install the extension dependencies:

```bash
uv sync --extra scale
```

Check that PyTorch can see CUDA:

```bash
uv run python -c \
  "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

Example run:

```bash
uv run python scaleup_ppo.py \
  --iterations 50 \
  --batch-size 2 \
  --max-new-tokens 48 \
  --output-dir artifacts/scaleup_ppo
```

If a cloud image already contains a compatible GPU PyTorch build, verify the
exact interpreter before attempting to reuse it:

```bash
deactivate 2>/dev/null || true
python -c "import sys, torch; print(sys.executable); print(torch.__version__)"
```

Then create a system-site environment with that interpreter:

```bash
TORCH_PYTHON=/path/to/interpreter/that/imported/torch

rm -rf .venv
uv venv --system-site-packages --python "$TORCH_PYTHON"
uv sync --locked --extra scale --no-install-package torch
uv run --no-sync python scaleup_ppo.py \
  --iterations 1 \
  --batch-size 1 \
  --max-new-tokens 8 \
  --output-dir artifacts/scaleup_smoke
```

`--no-install-package torch` does not make an arbitrary system installation
visible. Only use it after confirming that the environment can import the
intended CUDA build. If inheritance fails, let uv manage torch with
`uv sync --locked --extra scale`.

The repository does not include an automated GPU regression test. Validate the
model revision, reward interface, memory footprint, generation quality, KL
stability, and save/load behavior before a long run.

## 9. Fixed Evaluation

`prepare_local_data.py` creates a deterministic evaluation set from the
validation/test prompts in the preference data and excludes preference-training
prompts. The evaluation source is pinned to the first
`eval_source_examples=4000` preference rows, so later training-set expansion does
not silently change the evaluation set.

```bash
uv sync --extra scale
uv run python prepare_local_data.py
```

Generated files include:

```text
data/low_memory/grpo_prompts.jsonl
data/low_memory/eval_prompts.jsonl
```

Empty prompts, empty responses, and identical chosen/rejected pairs are skipped.
After confirming the evaluation set, freeze a snapshot:

```bash
cp data/low_memory/eval_prompts.jsonl data/eval_prompts_v1.jsonl
sha256sum data/eval_prompts_v1.jsonl
```

Compare reference and policy outputs on the fixed set:

```bash
uv run python evaluate_hf_policy.py \
  --policy-model artifacts/scaleup_ppo \
  --reference-model HuggingFaceTB/SmolLM2-135M-Instruct \
  --prompts-file data/eval_prompts_v1.jsonl \
  --limit 200 \
  --batch-size 1 \
  --max-new-tokens 256 \
  --output-dir artifacts/evaluations/scaleup_ppo
```

The evaluator uses the same chat template and reward model as training, with
greedy decoding by default. It writes:

```text
artifacts/evaluations/scaleup_ppo/records.jsonl
artifacts/evaluations/scaleup_ppo/summary.json
```

Per-prompt records include both responses, rewards, reward delta, sampled KL,
token lengths, and the prompt-file hash in the summary metadata.

For PEFT adapters, install the local extra:

```bash
uv sync --extra local
uv run python evaluate_hf_policy.py \
  --policy-model artifacts/local_dpo \
  --prompts-file data/eval_prompts_v1.jsonl \
  --limit 200 \
  --batch-size 1 \
  --max-new-tokens 256 \
  --output-dir artifacts/evaluations/local_dpo
```

Use the same prompt file, decoding settings, token limit, and reward model when
comparing PPO, GRPO, and DPO.

### 9.1 DPO Preference Validation

Generated reward is not the most direct DPO metric. Use held-out preference
pairs and response-only log-probabilities:

```text
chosen_reward = beta * (log pi(chosen) - log pi_ref(chosen))
rejected_reward = beta * (log pi(rejected) - log pi_ref(rejected))
margin = chosen_reward - rejected_reward
accuracy = fraction(margin > 0)
loss = -mean(log_sigmoid(margin))
```

The strict pairwise accuracy treats zero margins as non-wins. The evaluator also
reports:

```text
pairwise_accuracy_with_ties_half
pairwise_accuracy_non_tied
tie_rate
zero_margin_count
non_truncated metrics
```

Run preference validation:

```bash
uv run python evaluate_dpo_preferences.py \
  --policy-model artifacts/local_dpo_smoke_v2 \
  --reference-model HuggingFaceTB/SmolLM2-135M-Instruct \
  --preferences-file data/low_memory/preference_validation.jsonl \
  --limit 200 \
  --max-length 1024 \
  --beta 0.1 \
  --output-dir artifacts/evaluations/dpo_preference_validation
```

`--beta` must match `training.dpo_beta` in the training configuration.

## 10. Known Limitations

- The original toy set has only 6 SFT examples and 8 preference pairs.
- The word tokenizer does not understand morphology and is not production BPE.
- The small toy reward model is uncalibrated and vulnerable to reward hacking.
- CPU tests validate shapes and gradients, not generalization.
- Small preference runs can remain at chance-level pairwise accuracy.
- The model can repeat text or exploit reward-model length/style preferences.
- The fixed evaluations are smoke/pilot evidence, not paper-scale conclusions.
- The full pipeline uses different parameterizations across PPO and LoRA paths;
  algorithm comparisons must align training budget, data, and evaluation.

## 11. Suggested Next Steps

1. Freeze the evaluation prompt set and report confidence intervals.
2. Run a longer DPO experiment at a context length that avoids truncation.
3. Evaluate intermediate checkpoints, not only the final adapter.
4. Add task-specific checks for factual correctness, format, and repetition.
5. Compare DPO, GRPO, and PPO under aligned data and training budgets.
6. Add reward calibration, pairwise accuracy, KL tracking, and human spot checks.

## References

- Ouyang et al. (2022), *Training language models to follow instructions with
  human feedback*
- Christiano et al. (2017), *Deep Reinforcement Learning from Human Preferences*
- Schulman et al. (2017), *Proximal Policy Optimization Algorithms*
- Rafailov et al. (2023), *Direct Preference Optimization*

## Attribution and License

Upstream project:
[Iamyulx/rlhf-mini-implementation](https://github.com/Iamyulx/rlhf-mini-implementation).

This repository is a substantially modified educational derivative. See
[ATTRIBUTION.md](ATTRIBUTION.md) for the upstream commit, licensing status, and
detailed differences.

Released under the MIT License. See [LICENSE](LICENSE).
