# Attribution and Upstream Differences

## Upstream

- Repository: [Iamyulx/rlhf-mini-implementation](https://github.com/Iamyulx/rlhf-mini-implementation)
- Inspected commit: `9e493b2739b392ee3631449267ac0a15c63a9735`
- Upstream README license statement: `MIT (c) Iamyulx`
- The inspected upstream commit does not contain a standalone `LICENSE` file.

This repository is a substantially modified educational derivative. The
upstream copyright notice and MIT terms are preserved in `LICENSE`.

## Main Differences

| Area | Upstream | This project |
|---|---|---|
| Core scope | Minimal LSTM RLHF example | Teaching pipeline with PPO, DPO, GRPO, and Transformer-based extensions |
| Policy update | Simplified policy-gradient step | PPO with old log-probs, clipped objective, GAE, value head, entropy, and KL shaping |
| Reference policy | Not present | Frozen SFT/reference copy used for KL and comparison |
| Training stages | Reward-model training only | SFT, reward model, PPO, DPO, and GRPO entry points |
| Language model | Tiny word-level LSTM | Optional pretrained Transformer path with LoRA support |
| Data | Tiny in-code toy examples | Configurable Hugging Face dataset slices with deterministic splits |
| Evaluation | Toy reward accuracy | Fixed prompt evaluation plus DPO preference validation |
| Environment | Single `requirements.txt` | Python 3.12 with `pyproject.toml`, `uv.lock`, and optional dependency groups |
| Tests | No dedicated test suite | Unit tests for PPO, GRPO, DPO, tokenization, and evaluation metrics |
| Hardware docs | Generic | Abstract low-memory and high-memory profiles |

## Third-Party Assets

Models and datasets referenced by configuration keep their own licenses and
terms. Users are responsible for reviewing those terms before redistribution or
commercial use. No model weights, downloaded datasets, checkpoints, credentials,
or experiment logs are included in this repository.

## Publishing Notes

The upstream README declares MIT, which permits copying, modification,
redistribution, sublicensing, and commercial use provided that the copyright and
permission notice are preserved. This repository includes those notices in
`LICENSE`.

The inspected upstream commit has no standalone `LICENSE` file. For a
high-confidence legal review, ask the upstream author to add the standard MIT
license file or otherwise confirm the grant in writing.

GitHub write permission is separate from copyright licensing. A fork or new
independent repository can be published without upstream write access. Directly
pushing to the upstream default branch still requires collaborator permission;
otherwise the normal route is a fork plus pull request.

This document is a project-maintenance record, not legal advice.
