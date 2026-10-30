set -euo pipefail
cd "$(dirname "$0")/.."
accelerate launch -m train.cats \
  --basepath "${BASE_MODEL_PATH:?}" \
  --stage1_datadir "${STAGE1_DATA_DIR:?}" \
  --stage2_datadir "${STAGE2_DATA_DIR:?}" \
  --stage1_epochs 20 --stage2_epochs 5 \
  --stage1_lr 3e-5 --stage2_lr 8e-6 \
  --stage1_outdir "${STAGE1_OUT_DIR:-checkpoints/cats/stage1}" \
  --stage2_outdir "${STAGE2_OUT_DIR:-checkpoints/cats/stage2}" \
  "$@"
