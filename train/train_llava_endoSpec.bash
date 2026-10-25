# 从头跑stage1和stage2
CUDA_VISIBLE_DEVICES=1 accelerate launch /media/user/e/xyk/eagle-eye2/eagle-eye2/train/train_llava_endoSpec.py

# 只跑stage1
CUDA_VISIBLE_DEVICES=2 accelerate launch /media/user/e/xyk/eagle-eye2/eagle-eye2/train/train_llava_endoSpec.py \
  --only_stage1 \
  --resume_stage1_state /media/user/d/xyk/model/Eagle_Qwen2.5vl_draft/endoSpec_distill/ScienceQA/stage1/state_25

# 给出state，只跑stage2
CUDA_VISIBLE_DEVICES=3 accelerate launch /media/user/e/xyk/eagle-eye2/eagle-eye2/train/train_llava_endoSpec.py \
  --only_stage2 \
  --init_stage2_from_stage1_state /media/user/d/xyk/model/Eagle_llava_draft/endoSpec_distill/ScienceQA/Stage1/epoch_20

