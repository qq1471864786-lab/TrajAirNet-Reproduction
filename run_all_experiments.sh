#!/bin/bash
set -euo pipefail

# Rebuild experiment queue for the new vertical-state-conditioned relation model.
# Usage:
#   nohup bash run_all_experiments.sh > run_all.log 2>&1 &

DEVICE="${DEVICE:-cuda:0}"
SAVE_DIR="${SAVE_DIR:-save_model_rebuild}"
DATASET_VARIANT="${DATASET_VARIANT:-social}"
DATASET_NAME="${DATASET_NAME:-7days1}"
PROTOCOL="${PROTOCOL:-bestof5}"
EPOCHS="${EPOCHS:-50}"
SEED="${SEED:-42}"

variants=(base interaction state full naive)

echo "=========================================="
echo "Rebuild queue start: $(date)"
echo "device=${DEVICE} save_dir=${SAVE_DIR}"
echo "dataset=${DATASET_VARIANT}/${DATASET_NAME} protocol=${PROTOCOL}"
echo "epochs=${EPOCHS} seed=${SEED}"
echo "=========================================="

for variant in "${variants[@]}"; do
  echo ""
  echo "[${variant}] start: $(date)"
  python train.py \
    --device "${DEVICE}" \
    --variant "${variant}" \
    --protocol "${PROTOCOL}" \
    --dataset_variant "${DATASET_VARIANT}" \
    --dataset_name "${DATASET_NAME}" \
    --epochs "${EPOCHS}" \
    --seed "${SEED}" \
    --save_dir "${SAVE_DIR}"
  echo "[${variant}] done: $(date)"
done

echo ""
echo "=========================================="
echo "Rebuild queue complete: $(date)"
echo "=========================================="
