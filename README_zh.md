# RLHF Mini Pipeline

[English](README.md) | [中文](README_zh.md)

一个面向深度学习与机器学习初学者的教学级 RLHF 项目。它保留上游的纯 PyTorch
路线，但把原本不完整的 "PPO-style" 更新补成可运行的 PPO 闭环，并加入冻结
reference model、KL 约束、偏好奖励模型和策略对比评估。

项目分为三条路线：

- **CPU 算法教学版**：字符/单词级 LSTM，小数据，支持 PPO、GRPO、DPO，
  已经完成实际测试。
- **低显存路线**：`SmolLM2-135M-Instruct` + LoRA + 小数据切片，
  以约 6GB 显存为目标，适合资源受限环境。
- **高显存扩展版**：Hugging Face Transformer PPO，以约 24GB 显存为目标。

完整开发记录、取舍、实测结果与后续建议见
[DEVELOPMENT.md](DEVELOPMENT.md)。

本项目是基于原项目的教学扩展。上游来源、许可证状态和功能差异见
[ATTRIBUTION.md](ATTRIBUTION.md)。

## 1. 当前能力

小版本包含一个标准的 SFT -> Reward Model -> PPO 流程：

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

关键实现：

- 偏好奖励模型使用 Bradley-Terry loss：
  `-log sigmoid(r_chosen - r_rejected)`。
- PPO 使用旧策略采样的 `old_logprob`、new/old ratio、clip surrogate、
  GAE、value head、entropy bonus 和梯度裁剪。
- reference model 是 SFT 策略的冻结副本，不参与更新。
- 每个响应 token 都有采样形式的 KL 惩罚，最后一步额外加入奖励模型分数。
- 训练前后用同一奖励模型比较 policy 与 reference 的平均奖励、胜率和 KL。
- DPO 直接使用 chosen/rejected 的参考策略 log-ratio，不需要 rollout 或 value。
- GRPO 对同一 prompt 的多个响应做组内奖励标准化，不使用 value head。

这不是工业级 RLHF，不应把玩具数据上的分数当作对齐质量结论。

## 2. 目录

```text
mini_lm.py                  LSTM actor-critic 策略
reward_model.py             序列级标量奖励模型
preference_loss.py          Bradley-Terry 偏好损失
tokenizer.py                单词 tokenizer、特殊 token、padding
toy_dataset.py              SFT 与偏好数据
checkpoints.py              统一加载 SFT 与 reward checkpoint
train_sft.py                SFT / 初始策略训练
train_reward_model.py       奖励模型训练与 pairwise accuracy
ppo_step.py                 rollout、KL、GAE、PPO 更新、reference 对比
train_ppo.py                PPO 训练入口
grpo_step.py                GRPO 组采样、组内优势与更新
train_grpo.py               GRPO 训练入口
dpo_step.py                 response-only logprob 与 DPO loss
train_dpo.py                DPO 训练入口
configs/low_memory.json     低显存路线的模型、数据和步数预算
pyproject.toml              项目 Python 版本、依赖和 local/scale 可选依赖
uv.lock                     uv 解析后的完整依赖锁文件
.python-version             固定 Python 3.12
prepare_local_data.py       只取固定数量样本的 HF 数据准备脚本
train_hf_local.py           低显存 DPO/GRPO LoRA 入口
scaleup_ppo.py              未实测的 GPU Transformer 扩展入口
evaluate_hf_policy.py       固定 held-out prompts 上的 policy/reference 对比评估
run_mini_pipeline.py        依次运行本地三阶段
tests/test_mini_pipeline.py 核心算法断言测试
DEVELOPMENT.md              完整开发、验证和扩展记录
ATTRIBUTION.md              上游归属、许可证说明和相对原项目差异
LICENSE                     MIT 许可证
```

## 3. Python 环境与本地运行

项目使用 [uv](https://docs.astral.sh/uv/) 管理 Python、虚拟环境和依赖。Python
版本固定在 `.python-version`，依赖统一定义在 `pyproject.toml`，实际安装版本由
`uv.lock` 锁定。

服务器尚未安装 uv 时：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

创建 Python 3.12 环境并安装默认的 CPU 小流程依赖：

```bash
uv python install
uv sync
```

运行完整小流程：

```bash
uv run python run_mini_pipeline.py
```

选择本地算法：

```bash
uv run python run_mini_pipeline.py --algorithm ppo
uv run python run_mini_pipeline.py --algorithm grpo
uv run python run_mini_pipeline.py --algorithm dpo
```

也可以分阶段运行：

```bash
uv run python train_sft.py
uv run python train_reward_model.py
uv run python train_ppo.py
uv run python train_grpo.py
uv run python train_dpo.py
```

运行测试：

```bash
uv run python -m unittest discover -s tests -v
```

`uv run` 会自动使用项目 `.venv`，不需要先手工执行 `activate`。

如果希望后续直接使用 `python` 命令，需要先激活 uv 创建的环境：

```bash
source .venv/bin/activate
python run_mini_pipeline.py
```

激活只对当前终端会话有效。新开终端后需要重新执行 `source .venv/bin/activate`；
否则 `python` 通常指向系统 Python，看不到 `.venv` 中安装的依赖。`uv sync` 不需要
每次运行都执行，只在依赖声明或 `uv.lock` 变化、环境缺失时执行。

产物默认写入 `artifacts/`：

- `sft_policy.pt`
- `reward_model.pt`
- `ppo_policy.pt`
- `grpo_policy.pt`
- `dpo_policy.pt`

## 4. PPO 公式

对一条响应轨迹，reference 提供逐 token 参考分布。KL 奖励塑形为：

```text
token_reward_t = -beta * (log pi(a_t | s_t) - log pi_ref(a_t | s_t))
```

最后一个有效 token 再加上奖励模型给出的序列分数。使用 GAE 得到优势：

```text
delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
A_t = delta_t + gamma * lambda * A_{t+1}
```

PPO 的裁剪替代目标为：

```text
ratio_t = exp(log pi_new(a_t) - log pi_old(a_t))
L_policy = -mean(min(
    ratio_t * A_t,
    clip(ratio_t, 1 - eps, 1 + eps) * A_t
))
```

总损失还包含 value regression 与 entropy bonus。生产 RLHF 还会加入更多数据、
奖励模型校准、长度偏置处理、分布式训练、评估集和人工审计，这里刻意没有展开。

## 5. 算法模式

| 算法 | 是否需要 reward model | 是否需要 rollout | 本地状态 |
|---|---|---|---|
| PPO | 是 | 是，带 value head | 已实测 |
| GRPO | 是 | 是，同 prompt 多响应 | 已实测 |
| DPO | 否 | 否 | 已实测 |

GRPO 的标准组内优势为：

```text
A_i = (reward_i - mean(group_reward)) / (std(group_reward) + eps)
```

DPO 使用 chosen/rejected 相对 reference 的隐式奖励：

```text
z = beta * (
    (log pi(chosen) - log pi_ref(chosen))
    - (log pi(rejected) - log pi_ref(rejected))
)
L = -log sigmoid(z)
```

本地 CPU 实测中，DPO 的 toy training margin 从 `0.0468` 上升到 `2.0200`；
GRPO 的平均采样奖励从 `0.0518` 上升到约 `0.0618`。这仍只证明算法链路可运行，
因为数据只有 8 条偏好对，不能证明泛化能力。

## 6. 低显存配置

[configs/low_memory.json](configs/low_memory.json) 选择：

- 基座：`HuggingFaceTB/SmolLM2-135M-Instruct`，约 134.5M 参数。
- 训练方式：LoRA，rank 8，单样本 batch，8 步梯度累积。
- 上下文：384 token；DPO 最大 prompt 160；GRPO 最大生成 224。
- SFT 切片：2,000 条 `UltraChat 200k`。
- 偏好切片：4,000 对 `UltraFeedback Binarized`。
- GRPO prompts：最多 500 条，来自偏好训练切片。
- 步数上限：DPO 200 step，GRPO 80 step。
- 粗略时间预算：SFT 约 0.5 到 2 小时，DPO 约 1 到 3 小时，
  GRPO 约 3 到 8 小时。

这不是在完整数据集上训练。数据准备脚本只取配置中的固定数量样本：

```bash
uv sync --extra local
uv run python prepare_local_data.py
```

低显存 SFT/DPO/GRPO 入口：

```bash
uv run python train_hf_local.py --algorithm sft
uv run python train_hf_local.py --algorithm dpo
uv run python train_hf_local.py --algorithm grpo
```

首次测试可临时覆盖配置中的步数：

```bash
uv run python train_hf_local.py \
  --algorithm dpo \
  --max-steps 20 \
  --max-length 1024 \
  --output-dir artifacts/local_dpo_smoke
```

验证 smoke test 后，可扩大到一个中间 checkpoint：

```bash
uv run python train_hf_local.py \
  --algorithm dpo \
  --max-steps 100 \
  --max-length 1024 \
  --output-dir artifacts/local_dpo_100
```

仓库不附带数据集、模型权重或 GPU 自动化集成测试。首次运行时建议先执行
smoke test，确认显存、loss 和 adapter 保存正常后再跑完整配置。

## 7. Reference Model 的作用

本项目的 reference model 采用工业界常见语义：对 SFT 策略做深拷贝并冻结。
它同时承担两种职责：

1. KL 锚点，防止 PPO 为追逐奖励而偏离初始语言能力太远。
2. 对比基准，用相同奖励模型比较 policy 与 reference 的响应。

小版本没有额外下载 GPT-2 等语言模型，原因是不同 tokenizer 与架构会让
教学代码复杂化。冻结 SFT 副本在算法意义上是正确的标准基线，而且没有网络依赖。

## 8. 高显存扩展

`scaleup_ppo.py` 将小型 PPO 扩展到 Hugging Face 因果语言模型：

- 默认策略与 reference：`HuggingFaceTB/SmolLM2-135M-Instruct`
- 默认奖励模型：`OpenAssistant/reward-model-deberta-v3-base`
- 支持自定义 `--policy-model`、`--reward-model`、JSONL prompts 和 batch size
- 使用 bfloat16、gradient checkpointing、GAE、KL penalty 和 PPO clipping

安装扩展依赖：

```bash
uv sync --extra scale
```

默认让 uv 安装锁文件中的 torch，环境最可靠：

```bash
uv sync --extra scale
uv run python -c \
  "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

只有确认 CUDA 可用后再运行训练或评估。

首次在另一台机器或另一种 uv 版本上同步时，uv 可能会更新锁文件。应先提交更新后的
`uv.lock`，确认 `uv lock --check` 通过；之后才适合在 CI 或部署命令中使用
`--locked`。如果当前服务器的 `pyproject.toml` 和 `uv.lock` 不是同时更新，
`uv sync --locked` 会直接拒绝执行。

示例启动命令：

```bash
uv run python scaleup_ppo.py \
  --iterations 50 \
  --batch-size 2 \
  --max-new-tokens 48 \
  --output-dir artifacts/scaleup_ppo
```

如果云镜像已经预装 GPU 版 PyTorch，并且希望复用它，不能在未验证的情况下直接
使用 `--no-install-package torch`。先在项目 `.venv` 之外找到真正能导入 torch
的解释器：

```bash
deactivate 2>/dev/null || true
python -c "import sys, torch; print(sys.executable); print(torch.__version__)"
```

然后使用打印出的解释器路径重建环境：

```bash
TORCH_PYTHON=/path/to/the/interpreter/that/imported/torch

rm -rf .venv
uv venv --system-site-packages --python "$TORCH_PYTHON"
uv sync --locked --extra scale --no-install-package torch
uv run --no-sync python scaleup_ppo.py \
  --iterations 1 \
  --batch-size 1 \
  --max-new-tokens 8 \
  --output-dir artifacts/scaleup_smoke
```

这里的 `--no-sync` 会防止 `uv run` 重新安装锁文件中的 PyTorch、覆盖镜像预装的
GPU 版 PyTorch。该预镜像下的后续 Python 命令也应统一使用
`uv run --no-sync`；安装其他新依赖后，需要重新执行对应的 `uv sync`。

如果 `uv run --no-sync python -c "import torch"` 仍然失败，说明该解释器的
system site-packages 无法被 venv 继承。此时不要继续使用
`--no-install-package torch`。直接删除 `.venv` 并执行
`uv sync --locked --extra scale`，让 uv 管理 torch。

不希望每次都写 `uv run --no-sync` 时，激活同一个环境后即可直接使用 `python`：

```bash
source .venv/bin/activate
python scaleup_ppo.py \
  --iterations 1 \
  --batch-size 1 \
  --max-new-tokens 8 \
  --output-dir artifacts/scaleup_smoke
```

重要：该入口**没有在本地或 GPU 上实测**。它只通过语法检查。正式使用前至少需要
验证模型 revision、奖励模型接口、显存占用、生成质量、KL 稳定性和保存/加载。

## 9. 固定评估集与算法对比

`prepare_local_data.py` 会从偏好数据的 validation/test prompts 中按 hash 稳定抽取
固定评估集，并保证与 preference train prompts 无交集。评估来源固定为前
`eval_source_examples=4000` 条偏好数据，因此以后扩大训练切片不会改变评估集：

```bash
uv sync --extra scale
uv run python prepare_local_data.py
```

生成的文件包括：

```text
data/low_memory/grpo_prompts.jsonl
data/low_memory/eval_prompts.jsonl
```

数据准备会跳过空 prompt、空回答以及 chosen/rejected 完全相同的偏好记录，并在
输出中报告跳过数量。

第一次确认评估集后，把它快照为固定版本；后续实验不要重新生成或改写这个文件：

```bash
cp data/low_memory/eval_prompts.jsonl data/eval_prompts_v1.jsonl
sha256sum data/eval_prompts_v1.jsonl
```

先在固定评估集上比较 reference 与 PPO 输出：

```bash
uv run python evaluate_hf_policy.py \
  --policy-model artifacts/scaleup_ppo \
  --reference-model HuggingFaceTB/SmolLM2-135M-Instruct \
  --prompts-file data/eval_prompts_v1.jsonl \
  --limit 200 \
  --max-new-tokens 48 \
  --output-dir artifacts/evaluations/scaleup_ppo
```

评估器使用与训练一致的 chat template 和 DeBERTa reward，并固定 greedy 解码。
它输出：

```text
artifacts/evaluations/scaleup_ppo/records.jsonl
artifacts/evaluations/scaleup_ppo/summary.json
```

`records.jsonl` 保存每个 prompt 的 reference/policy 响应、reward、长度和 sampled KL；
`summary.json` 汇总平均/中位 reward、逐 prompt 胜率、长度和 KL。

DPO/GRPO 保存的是 PEFT adapter，评估时需要包含 PEFT 的 local 依赖：

```bash
uv sync --extra local
uv run python evaluate_hf_policy.py \
  --policy-model artifacts/local_dpo \
  --prompts-file data/eval_prompts_v1.jsonl \
  --output-dir artifacts/evaluations/local_dpo
```

GRPO 只需把 `--policy-model` 改为 `artifacts/local_grpo`。比较不同算法时，
必须使用同一个 `eval_prompts.jsonl`、`--max-new-tokens`、解码设置和 reward model。

### 9.1 DPO 偏好验证

生成的回答 reward 不是 DPO 最直接的验证指标。`evaluate_dpo_preferences.py`
直接使用 held-out preferred/rejected pairs 计算 response-only logprob：

```text
chosen_reward = beta * (log pi(chosen) - log pi_ref(chosen))
rejected_reward = beta * (log pi(rejected) - log pi_ref(rejected))
margin = chosen_reward - rejected_reward
accuracy = fraction(margin > 0)
loss = -mean(log_sigmoid(margin))
```

`pairwise_accuracy` 是把 `margin <= 0` 都视为未胜出的严格胜率。为了正确处理
`margin == 0`，评估还会报告：

```text
pairwise_accuracy_with_ties_half：tie 计 0.5
pairwise_accuracy_non_tied：排除 tie 后的二分类准确率
tie_rate：margin 为 0 的比例
```

运行前确保 local 依赖已经安装：

```bash
uv sync --extra local
```

先评估 200 个 validation pairs：

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

输出：

```text
artifacts/evaluations/dpo_preference_validation/records.jsonl
artifacts/evaluations/dpo_preference_validation/summary.json
```

重点观察 `pairwise_accuracy`、`margin.mean`、`margin.ci95`、`dpo_loss` 和
`truncated_rate`。`--beta` 应与训练配置中的 `training.dpo_beta` 一致。

## 10. 已知限制

- 仅 6 条 SFT 样本和 8 条偏好对，模型很容易过拟合。
- 单词 tokenizer 不懂形态变化，也没有 BPE 的生产效率。
- 奖励模型是未校准的小型 LSTM，存在 reward hacking 空间。
- CPU 测试只证明训练链路和数值形状正确，不代表泛化能力。
- 当前小版本采用全 batch PPO，便于阅读；大规模版本才需要 minibatch。
- 奖励分数绝对值没有可解释意义，应主要观察相对 margin、KL 和独立评估。

## 11. 建议的后续方向

建议先讨论并选择一项，而不是同时扩大所有部分：

1. 加入独立偏好测试集和长度偏置分析，先确认奖励模型是否学到了期望行为。
2. 把高显存路线的最小实验固定为单卡：135M policy + 独立 reward，
   记录峰值显存和每秒生成 token。
3. 增加 LoRA，以便在单卡上尝试更大策略模型，但要单独评估 reference 与
   adapter 的保存、加载和 KL 计算。
4. 在真实小切片上比较 DPO、GRPO、PPO 的训练时间、KL、独立奖励和响应长度。
5. 增加 reward model calibration、pairwise accuracy、KL 和人工抽检，
   不再只看训练 reward。

## 12. 参考资料

- Ouyang et al. (2022), Training language models to follow instructions with
  human feedback
- Christiano et al. (2017), Deep Reinforcement Learning from Human Preferences
- Schulman et al. (2017), Proximal Policy Optimization Algorithms
- Rafailov et al. (2023), Direct Preference Optimization

原项目：[Iamyulx/rlhf-mini-implementation](https://github.com/Iamyulx/rlhf-mini-implementation)

## License

MIT. See [LICENSE](LICENSE) and [ATTRIBUTION.md](ATTRIBUTION.md).
