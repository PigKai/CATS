# CATS

**Acceptance-Oriented Critical Token Adaptive Selection for Multimodal Speculative Decoding**

CATS 基于 EAGLE-2 草稿模型进行两阶段训练：Stage I 在所有有效回答 token 上做全局对齐；Stage II 只在动态困难 token（HTDS）与视觉关键 token（VTS）的并集上计算精修损失。可选的视觉隐藏状态增强（VHE）用于训练，推理仍使用共享的草稿树与目标模型验证流程。

## 模型与数据

目标模型包括 **LLaVA-1.5-7B/13B、Qwen2.5-VL-3B/7B**。统一入口从目标模型配置读取架构和维度；每个目标模型分别生成缓存、训练草稿模型，不混用权重和缓存。

| 数据 | `--benchmark` | 训练样本数 |
| --- | --- | ---: |
| LLaVA-Instruct | `llava_instruct` | 68,000 |
| ScienceQA | `scienceqa` | 12,726 |
| A-OKVQA | `aokvqa` | 17,056 |
| TextVQA | `textvqa` | 34,602 |

评估从每个下游数据集的 test split 均匀无放回抽取 500 条，分别使用温度 0 和 1。数据入口支持 Hugging Face 数据集、本地 `save_to_disk` 数据、JSON/JSONL 和 Parquet，需提供相应 split、图像和题目字段；ScienceQA 还需 `choices`。相对图像路径使用 `--image_root` 指定根目录。

## 安装与接口

在 Python 3.10、已安装适配硬件的 PyTorch 的环境中，从仓库根目录执行：

```bash
pip install -e .
pip install datasets==3.2.0
```

`setup.py` 固定 `transformers==4.57.1` 和 `accelerate==1.6.0`。历史发行包名称仍为 `eagle-eye`，当前模型接口为 `model.ea_model.EaModel`。完整模型训练与推理需要相应 GPU、目标模型权重和数据；推理使用 float16 和 eager attention。

## 论文与实现

| 论文位置 | 实现 |
| --- | --- |
| 式 (5)–(8)：概率比、EMA 与动态阈值 | `model/cats.py:HardTokenSelector` |
| 式 (9)：有图/无图目标模型对数概率差 | `distillation/generate.py:cache_trajectory` |
| 式 (10)：视觉依赖 top 10% | `model/cats.py:vision_critical_mask` |
| 式 (11)、(17)：dense / sparse NLL | `model/cats.py:nll_loss`、`train/cats.py:run_stage` |
| 式 (12)–(14)：视觉池化与隐藏状态增强 | `model/cats.py:enhance_hidden_states` |
| 式 (15)–(16)：HTDS 与 VTS 并集 | `model/cats.py:critical_mask` |
| 附录 B.3：隐藏状态回归与概率蒸馏组合 | `model/cats.py:distillation_loss` |
| 评估与指标汇总 | `evaluation/benchmark.py`、`evaluation/speed.py` |

默认 `--objective nll` 对应主文 NLL。Stage II 要求由 `--target_sampling` 生成的目标模型采样轨迹，同一轨迹用于困难 token 选择和精修监督。

附录 B.1 的 greedy 数据构造和 B.3 的组合蒸馏目标是另一种显式配置：数据生成去掉 `--target_sampling`，训练使用 `--objective distillation --v_w 1.0 --p_w 0.1`。该目标包含 Smooth L1 隐藏状态损失与概率级交叉熵，不能与主文 NLL 总损失视为相同目标。两种配置使用独立输出目录及匹配的 Stage-I checkpoint，不跨目标恢复优化器状态。

## 数据与训练

以下命令从仓库根目录执行。以 LLaVA-7B / ScienceQA 为例，替换路径、`BENCHMARK` 和输出目录即可配置其他模型或任务。

```bash
export BASE_MODEL_PATH=models/llava-v1.5-7b-hf
export LLAVA_INSTRUCT_DATA=datasets/llava-instruct-68k
export BENCHMARK=scienceqa
export BENCHMARK_DATA=datasets/scienceqa
export STAGE1_DATA_DIR=cache/llava7b/stage1
export STAGE2_DATA_DIR=cache/llava7b/scienceqa-sampled
export STAGE1_OUT_DIR=checkpoints/llava7b/stage1
export STAGE2_OUT_DIR=checkpoints/llava7b/scienceqa-cats

python -m distillation.generate \
  --model_path "$BASE_MODEL_PATH" --dataset_path "$LLAVA_INSTRUCT_DATA" \
  --benchmark llava_instruct --split train --outdir "$STAGE1_DATA_DIR"

python -m distillation.generate \
  --model_path "$BASE_MODEL_PATH" --dataset_path "$BENCHMARK_DATA" \
  --benchmark "$BENCHMARK" --split train --outdir "$STAGE2_DATA_DIR" \
  --max_new_tokens 256 --gamma 0.1 --seed 0 --target_sampling

accelerate launch -m train.cats \
  --basepath "$BASE_MODEL_PATH" --stage1_datadir "$STAGE1_DATA_DIR" \
  --stage1_outdir "$STAGE1_OUT_DIR" --only_stage1 --objective nll \
  --stage1_epochs 20 --stage1_lr 3e-5 --seed 0

accelerate launch -m train.cats \
  --basepath "$BASE_MODEL_PATH" --benchmark "$BENCHMARK" \
  --stage2_datadir "$STAGE2_DATA_DIR" \
  --init_stage2_from_stage1_state "$STAGE1_OUT_DIR/state_20" \
  --stage2_outdir "$STAGE2_OUT_DIR" --only_stage2 --method cats --objective nll \
  --stage2_epochs 5 --stage2_lr 8e-6 \
  --selection_mode union --vhe --pool_type mean --eta_q 0.02 --eta_a 0.1 --seed 0
```

Stage I 缓存 LLaVA-Instruct 原始对话的 assistant 监督位置；Stage II 缓存目标生成回答、回答 mask、视觉依赖得分和图像特征。下游生成器默认仍为 greedy，beam size 为 1，最多 256 个新 token；上面的 `--target_sampling` 显式选择主文的采样路线。ScienceQA 使用带候选项的推理提示，A-OKVQA/TextVQA 添加 OCR 指令。

Stage II 默认使用 AdamW、betas `(0.9, 0.95)`、bf16、batch size 1、梯度累积 1、最大序列长度 2048。CATS 默认 warmup 500 steps、梯度裁剪 0.5、EMA 动量 0.99、阈值系数 0.8、视觉保留比例 0.1。每个 epoch 保存到 `state_N`；`progress.json` 记录训练配置和共享初始化的权重 SHA-256。恢复训练使用对应的 `--resume_stage1_state` 或 `--resume_stage2_state`。

## 基线与消融

`--method all` 从相同 epoch-20 checkpoint 分别训练三个分支：EAGLE-2 在 Stage-I 数据上继续至 epoch 25；SFT 在下游数据上 dense 精修 5 epochs；CATS 在下游数据上 sparse 精修 5 epochs。SFT 默认 warmup 100 steps、梯度裁剪 1.0。

```bash
accelerate launch -m train.cats \
  --basepath "$BASE_MODEL_PATH" --benchmark "$BENCHMARK" \
  --stage1_datadir "$STAGE1_DATA_DIR" --stage2_datadir "$STAGE2_DATA_DIR" \
  --init_stage2_from_stage1_state "$STAGE1_OUT_DIR/state_20" \
  --method all --objective nll \
  --eagle2_outdir checkpoints/llava7b/eagle2 \
  --sft_outdir checkpoints/llava7b/scienceqa-sft \
  --stage2_outdir "$STAGE2_OUT_DIR" --seed 0
```

消融均从相同 epoch-20 初始化独立训练，保留相同数据、目标和预算，并为每个变体指定独立 `--stage2_outdir`。在 Stage-II 命令中替换相应选项：

| 变体 | 选项 |
| --- | --- |
| HTDS | `--selection_mode hard --no-vhe` |
| VTS | `--selection_mode visual --no-vhe` |
| HTDS + VTS | `--selection_mode union --no-vhe` |
| HTDS + VHE | `--selection_mode hard --vhe --pool_type mean` |
| 完整 CATS | `--selection_mode union --vhe --pool_type mean` |
| Dense 精修 | `--method sft` |
| VHE 回答侧池化 | `--vhe --pool_type mean`、`max` 或 `mean_max` |

VHE 的问题侧固定 mean pooling，`--pool_type` 控制回答侧池化，默认 `mean`；本实现的 `mean_max` 为 `(mean + max) / 2`。增强系数由 `--eta_q`、`--eta_a` 控制。困难 token 选择始终使用原始隐藏状态；增强特征只用于精修。Stage-I-only 结果直接评估 `state_20`。

## 评估

```bash
python -m evaluation.benchmark \
  --base-model-path "$BASE_MODEL_PATH" --ea-model-path "$STAGE2_OUT_DIR/state_5" \
  --dataset_path "$BENCHMARK_DATA" --benchmark "$BENCHMARK" \
  --split test --sample_size 500 --seed 0 --temperature 0 1 --mode both \
  --total_token 60 --depth 7 --top_k 10 --max_new_tokens 256 --max_length 4096 \
  --output outputs/llava7b/scienceqa-cats.jsonl

python -m evaluation.speed outputs/llava7b/scienceqa-cats.jsonl
```

更换方法时替换草稿 checkpoint 与输出路径，保留相同抽样 seed、目标模型和推理参数。逐样本 JSONL 保存生成文本、token 数、耗时与接受长度，`.summary.json` 按温度汇总。速度比为 target 总墙钟时间 / speculative 总墙钟时间；平均接受长度为接受草稿 token 总数 / 验证轮数，不把目标模型直接产生的树根 token 计入接受草稿数。
