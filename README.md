<div align="center">

# 🤖 RLHF Mini Pipeline

**A minimal, readable implementation of Reinforcement Learning from Human Feedback — reward modeling, preference learning, and PPO-style policy optimization from scratch**

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white) ![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white) ![RLHF](https://img.shields.io/badge/RLHF-mini--pipeline-6366f1) ![License](https://img.shields.io/badge/License-MIT-22c55e) ![Status](https://img.shields.io/badge/Status-active-brightgreen)

</div>

---


## 🔍 Overview

Modern LLMs like ChatGPT and Claude are aligned to human preferences through **RLHF** —
a multi-stage training process involving supervised pretraining, reward modeling from human
comparisons, and reinforcement learning to optimize the policy toward those rewards.

This project implements **the full RLHF pipeline at minimal scale** using pure PyTorch,
with no external alignment libraries. Every component is readable and hackable in under
200 lines total.

> **Why this exists:** Most RLHF tutorials skip the implementation details or rely on
> opaque wrappers. This repo shows exactly what each stage does mathematically and in code.

---

## 🔄 Pipeline

```
Raw Text Prompts
      │
      ▼
  Tokenizer               ← whitespace tokenization + vocab encoding
      │
      ▼
  MiniLM (Policy)         ← Embedding → LSTM → Linear (language model)
      │
      ▼
  RewardModel             ← Embedding → LSTM → Scalar score
      │
      ▼
  Preference Loss         ← L = -log σ(r_chosen - r_rejected)
      │
      ▼
  PPO-style Update        ← maximize expected reward via policy gradient
```

---

## 🧩 Components

| Module | Description |
|---|---|
| `mini_lm.py` | MiniLM policy model: Embedding → LSTM → Linear |
| `reward_model.py` | RewardModel: scores responses with a scalar value |
| `preference_loss.py` | Bradley-Terry preference loss for pairwise ranking |
| `ppo_step.py` | Simplified PPO policy gradient update |
| `tokenizer.py` | Whitespace tokenizer with vocabulary builder |
| `toy_dataset.py` | Synthetic preference pairs for training and evaluation |
| `train_reward_model.py` | Training script: evaluates preference ranking |

---

## 🧠 Model Architectures

### Policy Model — `MiniLM`

A lightweight causal language model used to generate token probability distributions.

```
Input tokens
    │
    ▼
Embedding(vocab_size, 128)
    │
    ▼
LSTM(128 → 256, batch_first=True)
    │
    ▼
Linear(256 → vocab_size)          ← logits over vocabulary
```

### Reward Model

Scores a response with a single scalar — higher score = more preferred.

```
Input tokens
    │
    ▼
Embedding(vocab_size, 128)
    │
    ▼
LSTM(128 → 256, batch_first=True)
    │
    ▼
Linear(256 → 1)                   ← scalar reward
```

---

## 📐 Mathematical Foundations

### Preference Loss (Bradley-Terry model)

Given a chosen response `c` and rejected response `r`, the reward model is trained to
maximize the margin between their scores:

```
L = -log σ(r_chosen - r_rejected)
```

Where `σ` is the sigmoid function. This is the same loss used in InstructGPT and Claude's RLHF.

### PPO-Style Policy Update

The policy is updated to maximize the expected reward signal from the reward model:

```
loss = -E[reward_model(actions)]
```

This is a simplified REINFORCE-style update — a full PPO implementation would add
a clipped surrogate objective and KL penalty from the reference policy.

---

## ⚙️ Installation

```bash
git clone https://github.com/Iamyulx/rlhf-mini-implementation.git
cd rlhf-mini-implementation
pip install -r requirements.txt
```

---

## 🚀 Quickstart

**Train and evaluate the reward model:**
```bash
python train_reward_model.py
```

**Use components programmatically:**
```python
from tokenizer import Tokenizer
from mini_lm import MiniLM
from reward_model import RewardModel
from preference_loss import preference_loss
from ppo_step import ppo_step
import torch

# Build vocabulary
tokenizer = Tokenizer()
tokenizer.build_vocab(["explain phishing", "what is malware"])

vocab_size = tokenizer.vocab_size

# Initialize models
policy = MiniLM(vocab=vocab_size, embed=128)
reward  = RewardModel(vocab=vocab_size)

# Score a preference pair
chosen_ids   = torch.tensor([tokenizer.encode("phishing is a cyber attack")])
rejected_ids = torch.tensor([tokenizer.encode("phishing is a type of fish")])

r_chosen   = reward(chosen_ids)
r_rejected = reward(rejected_ids)

loss = preference_loss(r_chosen, r_rejected)
print(f"Preference loss: {loss.item():.4f}")

# PPO step
ppo_loss = ppo_step(policy, reward, chosen_ids)
print(f"PPO loss: {ppo_loss.item():.4f}")
```

---

## 📊 Results

> ⚠️ **Placeholder values.** Run `train_reward_model.py` and update with real output.

| Metric | Value |
|---|---|
| Reward model avg loss | ~0.661 |
| Chosen response avg score | ~0.050 |
| Rejected response avg score | ~-0.023 |
| Margin (chosen − rejected) | ~0.073 |
| Vocabulary size | 33 tokens |
| PPO avg reward | ~0.041 |

**Sample output from `train_reward_model.py`:**

```
Prompt: Explain phishing
  Chosen  (score:  0.0905): Phishing is a cyber attack
  Rejected (score:  0.0032): Phishing is a type of fish
  Loss: 0.6505

Prompt: What is malware
  Chosen  (score: -0.0093): Malware is malicious software
  Rejected (score: -0.0495): Malware is a computer mouse
  Loss: 0.6732

Average RewardModel loss: 0.6618
```

---

## 📦 Toy Dataset

The project includes a minimal cybersecurity preference dataset — a nod to the
domain expertise behind the project:

```python
preferences = [
    {
        "prompt":    "Explain phishing",
        "chosen":    "Phishing is a cyber attack that tricks users into revealing credentials",
        "rejected":  "Phishing is a type of fish"
    },
    {
        "prompt":    "What is malware",
        "chosen":    "Malware is malicious software designed to damage or gain unauthorized access",
        "rejected":  "Malware is a computer mouse"
    },
]
```

---

## ⚠️ Limitations & Honest Notes

This is intentionally a **toy implementation**. Key simplifications vs. production RLHF:

| This repo | Production RLHF (InstructGPT / Claude) |
|---|---|
| LSTM policy | Transformer (GPT-2, LLaMA, etc.) |
| Simplified policy gradient | Full PPO with clipped surrogate + KL penalty |
| 33-token vocabulary | 50k+ BPE tokens |
| 2 preference pairs | 10k–1M human comparisons |
| No reference model | KL divergence from frozen SFT model |

---

## 🗺️ Roadmap

- [ ] Replace LSTM with a small Transformer policy
- [ ] Implement full PPO with clipping and KL penalty
- [ ] Add BPE tokenizer
- [ ] Scale preference dataset
- [ ] Add W&B logging
- [ ] HuggingFace `trl` integration comparison

---

## 📚 References

- Ouyang et al. (2022) — [Training language models to follow instructions with human feedback (InstructGPT)](https://arxiv.org/abs/2203.02155)
- Christiano et al. (2017) — [Deep Reinforcement Learning from Human Preferences](https://arxiv.org/abs/1706.03741)
- Schulman et al. (2017) — [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- Bai et al. (2022) — [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073)

---

## 📄 License

MIT © [Iamyulx](https://github.com/Iamyulx)
