# RLHF Mini 开发与扩展记录

记录日期：2026-10-19

当前 Python 环境和依赖由 uv 管理：Python 版本固定在 `.python-version`，依赖声明
位于 `pyproject.toml` 和 `uv.lock`。下文保留迁移前的历史实测记录，但可复现命令
已经改为 `uv sync` 与 `uv run`。

## 1. 目标拆解

本项目面向已经了解神经网络、损失函数、梯度下降，但没有 RLHF 工程经验的读者。
目标不是复现大模型，而是让每个算法组件都能单独读懂、运行和检查。

本阶段工作分为四件事：

1. 修复并补全一个小型 PPO 策略梯度闭环。
2. 引入冻结 reference model 作为 KL 锚点和对比基线。
3. 写一个可迁移到高显存 GPU 的 Transformer 扩展入口。
4. 记录实现过程、设计动机、权衡、实测结果和后续可讨论方向。

## 2. 上游现状审计

上游只有 9 个文件，README 中的若干接口与实际代码不一致。

### 2.1 不能形成 PPO 闭环

原始 `ppo_step.py` 做了以下事情：

```python
logits = policy(input_ids)
probs = softmax(logits)
actions = multinomial(probs)
reward = reward_model(actions)
loss = -reward.mean()
```

主要问题：

- 没有记录行为策略的 `old_logprob`，无法计算 PPO ratio。
- 没有优势函数，reward 直接被当成 policy loss。
- 没有裁剪目标，也没有 value head 和 GAE。
- 没有 reference model，更没有 KL 约束。
- 把动作 token ID 直接输入奖励模型 embedding，和文本响应语义不对应。
- policy 与 reward 的输入形状、token 含义没有统一。

因此原来的实现更接近一段无法正确反传 reward 的示意代码，而不是 PPO。

### 2.2 训练脚本无法按原样运行

`train_reward_model.py` 使用了不存在的包路径：

```python
from data.toy_dataset import ...
from models.reward_model import ...
from utils.tokenizer import ...
from training.preference_loss import ...
```

仓库实际是扁平文件结构。README 还使用了不存在的 `Tokenizer` 类。
因此先统一接口，再重建训练入口。

### 2.3 数据和模型过于简化

- 只有 2 条偏好数据，2 个 epoch 级别的打印不能称为奖励模型训练。
- 没有特殊 token，也没有 batch padding。
- 奖励模型固定取最后位置，遇到 padding 会读错序列。
- policy 没有 value head。
- 没有 SFT 初始化，reference model 与 PPO 起点不清晰。

## 3. 设计决策

### 3.1 保留 LSTM，而不是立即替换 Transformer

小版本的目标是验证算法，不是追求语言质量。LSTM 训练快、依赖少，在 CPU 上
数秒即可完成若干百 epoch，因此适合作为教学 smoke test。

高算力版才切换为 Transformer，并保留相同 PPO 语义。这样读者可以先看清
rollout、advantage、ratio 和 KL，再处理 Hugging Face 的生成与显存工程。

### 3.2 加入 SFT 阶段

标准 RLHF 的 reference model 通常来自 SFT checkpoint。没有 SFT 时，随机初始
策略会生成无意义 token，奖励模型和教学信号都很差。

因此新增 `train_sft.py`。它只在 response token 上计算 cross entropy，
prompt 本身不参与语言模型损失。训练完成后：

- 同一 checkpoint 作为 PPO policy 初始化；
- 深拷贝并冻结为 reference model。

这是最小但语义正确的选择。没有单独下载外部 reference 模型，因为外部模型通常
使用不同 tokenizer 和架构，无法直接给当前 LSTM 提供逐 token KL。

### 3.3 使用 token-level KL reward shaping

PPO 在每个 step 计算：

```text
kl_sample_t = log pi(a_t | s_t) - log pi_ref(a_t | s_t)
reward_t = -beta * kl_sample_t
```

最后一个有效 token 加入 reward model 的序列分数。选择这种做法而不是把 KL
直接塞进 loss，是因为它与 InstructGPT 类实现一致，KL 可以由 PPO 的 advantage
统一处理。

注意：单样本的 log ratio 可以小于 0，所以训练日志中的 `mean KL` 可能短暂为负，
它是采样估计而非严格的非负 `KL(q || p)`。评估时应看长期均值与独立评估集。

### 3.4 reward 放在轨迹末端

奖励模型只判断完整回答，不能给每个 token 一个可靠的人类偏好标签。因此外部
任务奖励只加到最后一个 token，前面 token 仅承担 KL shaping。GAE 再把终端奖励
向前传播。

这是简化但常见的设计。它牺牲了 credit assignment 的精细度，换来奖励模型接口
简单、可解释。

### 3.5 全 batch PPO

小版本只有 8 到 14 条 prompt，使用一次完整 rollout 做若干 PPO epoch 即可。
没有 minibatch、分布式 sampler 和 gradient accumulation，因为这些只会增加
教学负担，不能验证新算法语义。

高算力扩展版保留同样算法，但将模型、tokenizer、数据和大 batch 换成生产组件。

## 4. 实现流程

### 4.1 统一 tokenizer

`tokenizer.py` 现在提供：

- `<pad>`, `<unk>`, `<bos>`, `<eos>` 固定特殊 token；
- `Tokenizer.build_vocab`、`encode`、`decode`；
- `pad_sequences` 返回 ids 和有效位置 mask；
- 保留上游的 `build_vocab`、`tokenize_text` 兼容函数。

### 4.2 策略模型与奖励模型

`MiniLM` 是 LSTM actor-critic：

- LM head 输出词表 logits；
- value head 输出每个位置的 baseline；
- LSTM 单向，位置 t 不会看到未来 token。

`RewardModel` 接收完整 prompt-response 序列，通过长度索引取最后一个真实 token，
避免 padding 干扰分数。

### 4.3 偏好奖励训练

`train_reward_model.py` 同时转发 chosen/rejected 序列，使用 Bradley-Terry
pairwise loss 训练，并报告 pairwise accuracy 与 reward margin。

### 4.4 PPO rollout

`collect_rollouts` 对每个 prompt：

1. 编码 prompt 并添加 `<bos>`。
2. 逐步采样响应 token。
3. 同时记录 policy logprob、reference logprob 和 value。
4. 用 reward model 对完整序列打分。
5. 构造 token rewards，计算 GAE 和 returns。

所有 rollout 使用 `torch.no_grad()`，之后在同一 batch 上做 PPO 更新。

### 4.5 PPO 更新

`_build_batch` 将变长轨迹右 padding。动作位置是
`prompt_length - 1 + response_step`，因为该位置的 LM head 预测下一个响应 token。

更新包含：

- clipped surrogate objective；
- value MSE；
- entropy bonus；
- advantage normalization；
- 梯度裁剪；
- 多个 PPO epoch，但 rollout 不重新采样。

### 4.6 Reference 对比

`compare_to_reference` 使用 greedy decoding 分别生成 policy 与 reference 响应，
再用同一奖励模型评分，输出：

- policy average reward；
- reference average reward；
- policy win rate；
- policy token KL；
- 两侧文本响应。

这使 reference 不只是 KL 计算对象，也是可观察的 baseline。

## 5. 本地验证

环境记录（uv 迁移前实测）：

- 本地操作系统
- Python 3.12.13
- PyTorch 2.14.1 CPU
- NumPy 2.5.4

### 5.1 单元测试

命令：

```powershell
uv sync
uv run python -m unittest discover -s tests -v
```

结果：3 个测试全部通过。

覆盖内容：

- tokenizer round trip 与 padding mask；
- GAE 手算值；
- PPO 更新 policy，但 reference 参数完全不变；
- 训练统计值全部有限。

### 5.2 三阶段 smoke test

命令：

```powershell
uv run python run_mini_pipeline.py
```

实际运行参数使用脚本默认值：SFT 300 epoch、reward 250 epoch、PPO 30 iteration。

观测：

- SFT loss 从 4.3831 降到约 0.0014。
- reward model 在 toy preference 上 pairwise accuracy 为 1.000。
- PPO 前 policy/reference reward 均为 0.0518。
- PPO 后 policy reward 为 0.0522，reference 为 0.0518。
- deterministic comparison win rate 为 0.125，mean sampled KL 为 0.1023。

这些数字不能证明策略质量提升。数据只有 6 条 SFT、8 条偏好，reward model
很容易记住样本；PPO 后 reward 只略微变化，win rate 也不高。该实验的结论是：

1. 数据形状、rollout、反传、reference 冻结和评估链路都能工作。
2. 当前数据规模不足以证明泛化或对齐效果。
3. 下一步应优先增加独立验证集，而不是继续调大训练 iteration。

### 5.3 响应样例

已见 prompt 能生成接近 SFT 答案的响应，例如：

```text
Explain phishing
phishing is a cyber attack that tricks users into revealing information
```

未见 prompt 会明显退化，例如把 social engineering 回答成 firewall 内容。
这与 6 条训练样本和单词级 LSTM 的容量一致，也说明必须用 held-out prompt
评估，而不能只看训练 prompt。

## 6. 高算力扩展

新增 `scaleup_ppo.py`，默认：

- policy/reference：`HuggingFaceTB/SmolLM2-135M-Instruct`
- reward：`OpenAssistant/reward-model-deberta-v3-base`
- 单卡 bfloat16
- gradient checkpointing
- batch rollout + PPO clipped update

选择 135M 模型是为了让 policy、冻结 reference、value head 和奖励模型更可能放入
24GB 高显存设备。更大的 `--policy-model` 仍需重新测量显存。

### 6.1 未测试项

早期开发阶段没有下载大模型，也没有在 GPU 上运行 `scaleup_ppo.py`。
目前只做了 Python 语法检查。以下内容都没有经验证据：

- 默认模型和奖励模型 revision 是否仍可下载；
- 三者总显存和峰值 activation；
- Hugging Face reward model 的输出契约；
- chat template 与 reward tokenizer 拼接方式；
- bfloat16 下 ratio/loss 的稳定性；
- 每秒生成 token 数；
- 保存后的 policy 加载是否完整。

因此不能声称扩展版已经可复现，只能说代码路径已经落盘。

### 6.2 上 GPU 后的最小实验

建议按以下顺序，不要一次扩大：

1. 1 iteration、1 prompt、16 new tokens，确认模型和 reward 接口。
2. 记录加载后、rollout 后、backward 后的峰值显存。
3. 比较 policy 和 reference 相同输入的 logits 是否一致。
4. 跑 10 iteration，检查 KL 是否漂移、响应是否出现重复。
5. 加入 held-out preference set，报告独立 pairwise accuracy。
6. 最后才扩大 batch、response length 和模型。

## 7. 权衡与风险

### 7.1 奖励模型过拟合

toy 数据上 accuracy 1.000 主要说明模型能记忆 8 个 pair，不能说明 reward 可泛化。
应增加独立 held-out prompts、人工抽检和 reward margin 分布。

### 7.2 长度偏置

奖励模型可能通过长度、重复词或特定 token 走捷径。当前没有长度归一化或长度惩罚。
进一步扩展时应记录 chosen/rejected 的 token 长度并做控制实验。

### 7.3 Reward hacking

PPO 会主动寻找奖励模型的高分区。小模型上可能出现重复、套模板或输出偏长。
KL 只限制偏离 reference 的程度，不能修复错误奖励。

### 7.4 Reference 不是外部质量证明

reference 与 SFT 起点相同，所以对比能衡量“是否偏离以及奖励模型是否更喜欢新策略”，
不能证明新回答事实正确或更安全。需要独立人工评估。

### 7.5 单词 tokenizer

这套 tokenizer 适合解释流程，不适合中文或复杂生产语料。如果要面向中文教学，
下一步可替换为最小 BPE，或直接在高算力版使用 pretrained tokenizer。

## 8. 可选改进方向

按优先级建议：

### 方案 A：先把评估做可信

- 新增 held-out preference pairs。
- 报告 reward margin、pairwise accuracy、KL、响应长度。
- 固定随机种子并跑 3 次。
- 这是最小工作量、最高信息增益的方案。

### 方案 B：高显存最小 Transformer 实验

- 使用默认 135M 模型。
- 只训练 10 到 50 iteration。
- 记录显存、吞吐、KL 和生成失败率。
- 暂不加入 LoRA，先验证全参数小模型路径。

### 方案 C：LoRA 扩展

- 只对 policy 加 LoRA。
- reference 保持完整基础权重或冻结 adapter。
- 需明确保存、加载和 logprob 一致性测试。
- 适合在 24GB 级显存内尝试更大模型，但工程复杂度会上升。

### 方案 D：DPO 对照

本地 DPO 已实现，不需要在线 rollout，算力需求低，适合和 PPO 做教学对照。
它不能让读者深入理解 advantage 和 KL reward shaping，但可以回答
"在有限算力下是否需要 PPO"。

### 方案 E：奖励模型校准

- 加入 reward normalization 或 running mean/std。
- 分开验证集做早停。
- 检查长度偏置和 OOD prompt。

下一条路线是先做 A，然后在低显存设备上跑 1 到 5 step 的 HF smoke test。

## 9. 复现边界

已执行：

- 代码语法检查；
- 5 个算法单元测试；
- CPU 上完整 SFT、reward model、PPO 训练；
- CPU 上 DPO 与 GRPO 训练；
- reference 参数冻结检查；
- policy/reference greedy 对比；
- 真实响应样例检查。

未执行：

- GPU 训练；
- `scaleup_ppo.py`；
- `train_hf_local.py`；
- `prepare_local_data.py`；
- Hugging Face 大模型下载；
- 多随机种子、held-out 泛化评估；
- 人工偏好或安全评估。

## 10. 2026-10-19 后续扩展记录

### 10.1 新增算法

- `dpo_step.py` 实现 response-only logprob 和标准 DPO loss。
- `train_dpo.py` 使用冻结 SFT 副本作为 reference，训练本地 DPO。
- `grpo_step.py` 实现同 prompt 多响应采样、组内 reward 标准化、PPO ratio
  裁剪和显式 KL 正则。
- `train_grpo.py` 复用 reward model 和 reference，但不训练 value head。
- `run_mini_pipeline.py --algorithm ppo|grpo|dpo|all` 提供统一入口。

本地 toy 结果：

| 算法 | 观察 |
|---|---|
| DPO | loss `0.6704 -> 0.1362`，margin `0.0468 -> 2.0200` |
| GRPO | 平均采样 reward `0.0518 -> 0.0618`，win rate `0.375` |

这些结果只用于检查实现。数据量很小，不能据此判断算法优劣。

### 10.2 低显存配置

`configs/low_memory.json` 固定：

- `SmolLM2-135M-Instruct`
- LoRA rank 8
- 384 token 上下文
- 单样本 batch，8 步梯度累积
- 2,000 条 SFT 和 4,000 对偏好
- DPO 最多 200 step，GRPO 最多 80 step
- SFT 最多 200 step

选择该模型的原因是 134.5M 参数在 FP16 下权重约 0.27 GB，配合 LoRA、
短上下文和梯度检查点后，比直接加载 0.5B 模型更适合 6GB 显存。4k 偏好对和
步数上限用于限制总训练时间，而不是追求完整数据集覆盖。

`prepare_local_data.py` 只通过 HF split slice 读取固定数量样本，并按 prompt hash
划分 train/validation/test。`train_hf_local.py` 使用 TRL 的 SFTTrainer、
DPOTrainer 和 GRPOTrainer，policy 使用 PEFT LoRA；GRPO 奖励来自小型 DeBERTa
reward model。

早期开发阶段没有执行这两个 HF 脚本，因此以下项目仍未验证：

- 实际峰值显存；
- 本地系统下 FP16/LoRA 的兼容性；
- TRL 参数与当前安装版本是否匹配；
- 4k 偏好切片上的收敛速度；
- 报告中的 1 到 3 小时、3 到 8 小时预算是否准确。

第一次在目标低显存设备上运行时，应把 `max_steps` 改为 1 到 5，观察显存和
loss 后再放大。
