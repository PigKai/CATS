# 快速测试：先加 --max_train_samples 1000，确认能跑通后再去掉该参数用完整数据
deepspeed --include localhost:2 main_deepspeed_ppl.py \
  --deepspeed --deepspeed_config /home/dhz/eagle-eye2/train/ds_config.json \
  --prefix  /home/dhz/eagle-eye2/evaluation \
  --basepath /data1/dhz/Model/llava-v1.5-7b-hf \
  --config_path /home/dhz/eagle-eye2/train/llava_7B_config.json \
  --tmpdir /data1/dhz/get-data-llava-68k \
  --cpdir /data1/dhz/checkpoint/eagle-eye-llava-68k-PPL \
  --epochs 20 \
  --time_out_path /data1/dhz/checkpoint/eagle-eye-llava-68k-PPL/total_training_time_metric.csv \
  --filter_start_epoch 1 \
  --quantile 0.5 \
  --index 1 \
  --max_train_samples 100
