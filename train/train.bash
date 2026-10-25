CUDA_VISIBLE_DEVICES=2,3 accelerate launch \
  --mixed_precision=bf16 \
  -m train.train_llava \
  --tmpdir  /data1/dhz/get-data-llava-ocr \
  --cpdir /data1/dhz/checkpoint/eagle-eye-llava-ocr\
  --configpath /home/dhz/eagle-eye2/train/llava_7B_config.json \

