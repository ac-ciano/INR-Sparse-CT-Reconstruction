#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python "$SCRIPT_DIR/SIREN_train_nodule.py" \
  --batch-size 2048 \
  --lr 5e-5 \
  --coord-noise 0.0001 \
  --dropout 0.5 \
  --omega-0 30 \
  --epochs 4000 \
  --patience 750 \
  --nodule-id LIDC-IDRI-0601_nodule_3