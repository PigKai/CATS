set -euo pipefail
cd "$(dirname "$0")/.."
accelerate launch -m train.train_llava \
  --basepath "${BASE_MODEL_PATH:?}" \
  --stage1_datadir "${STAGE1_DATA_DIR:?}" \
  --stage1_outdir "${STAGE1_OUT_DIR:-checkpoints/cats/stage1}" \
  "$@"
