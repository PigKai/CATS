# stage1 断点继续
CUDA_VISIBLE_DEVICES=2 accelerate launch /media/user/e/xyk/eagle-eye2/eagle-eye2/train/train_qwenvl2.5_endoSpec.py \
  --only_stage1 \
  --resume_stage1_state /media/user/d/xyk/model/Eagle_Qwen2.5vl_draft/endoSpec_distill/ScienceQA/stage1/state_25

# stage2 从 stage1 初始化
CUDA_VISIBLE_DEVICES=3 accelerate launch /media/user/e/xyk/eagle-eye2/eagle-eye2/train/train_qwenvl2.5_endoSpec.py \
  --only_stage2 \
  --init_stage2_from_stage1_state /media/user/d/xyk/model/Eagle_Qwen2.5vl_draft/endoSpec_distill/ScienceQA/stage1/state_20

# stage2 断点继续
CUDA_VISIBLE_DEVICES=0 accelerate launch /media/user/e/xyk/eagle-eye2/eagle-eye2/train/train_qwenvl2.5_endoSpec.py \
  --only_stage2 \
  --resume_stage2_state /path/to/stage2/state_2

CUDA_VISIBLE_DEVICES=2 accelerate launch /media/user/e/xyk/eagle-eye2/eagle-eye2/train/train_qwenvl2.5_endoSpec.py \
  --only_stage1 \

# qwen2.5vl 包括hidden增强和visual增强

CUDA_VISIBLE_DEVICES=3 accelerate launch /media/user/e/xyk/eagle-eye2/eagle-eye2/train/train_qwen2.5vl_endoSpec_hidden_visual.py \
  --only_stage2 \
  --stage2_input_hidden_key mean_hidden_aug_a10_q2 \
  --stage2_target_hidden_key hidden_state \
  --selection_mode sr_visual_or \
  --stage2_datadir /media/user/d/xyk/data1/llava_instruct/qwen2.5vl/qwen2.5vl_7B_0_67999/0 \
  --stage2_outdir /media/user/d/xyk/model/Eagle_Qwen2.5vl_draft/endoSpec_distill/llava_instruct_3_17 \
  --init_stage2_from_stage1_state /media/user/d/xyk/model/Eagle_Qwen2.5vl_draft/endoSpec_distill/ScienceQA/stage1/state_3


  mean_hidden_aug_a10_q2