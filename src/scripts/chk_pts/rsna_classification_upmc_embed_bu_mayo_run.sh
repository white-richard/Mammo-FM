#!/bin/bash

### USER INPUT SECTION — set your checkpoint here:

CKPT_KEY="upmc_embed_bu_mayo"
CLIP_CKPT="/restricted/projectnb/batmanlab/shawn24/PhD/Breast-CLIP/src/codebase/outputs/upmc_embed_bu_mayo_clip/b5_detector_n_modernbert_2048/checkpoints/fold_0/MammoCLIP_Mayo_UPMC_EMBED_BU_epoch3_BatmanlabTrained.tar"


### Define global log base path
LOG_BASE="/restricted/projectnb/batmanlab/shawn24/PhD/Breast-CLIP-downstream/src/scc_logs/breast"
#SUB_DIR="epoch3/mayo_trained"
SUB_DIR="epoch3/batmanlab_trained"
############################################
### RSNA SETUP
############################################

source "/restricted/projectnb/batmanlab/shawn24/PhD/Breast-CLIP-downstream/src/scripts/configs/rsna_configs.sh"

RSNA_DATA_FRACS=("1.0" "0.5" "0.1")
#RSNA_DATA_FRACS=("1.0")
RSNA_ARCHS=("breast_clip_det_b5_period_n_ft" "breast_clip_det_b5_period_n_lp")

for ARCH in "${RSNA_ARCHS[@]}"; do
  for FRAC in "${RSNA_DATA_FRACS[@]}"; do
    if [[ "$ARCH" == *"lp"* && "$FRAC" != "1.0" ]]; then
      continue
    fi
    export CLIP_CKPT="$CLIP_CKPT"
    export ARCH="$ARCH"
    export DATA_FRAC="$FRAC"
    export DATASET_NAME="$DATASET_NAME"
    export DATA_DIR="$DATA_DIR"
    export IMG_DIR="$IMG_DIR"
    export CSV_FILE="$CSV_FILE"
    export LABEL="$LABEL"
    export EXTRA_ARGS="--label 'cancer' --n_folds 1"

    timestamp=$(date +"%Y-%m-%d-%H-%M-%S-%N")
    jobname="${ARCH//[^a-zA-Z0-9]/_}_${DATASET_NAME}_${FRAC}_${CKPT_KEY}_${timestamp}"
    logdir="${LOG_BASE}/classification_${DATASET_NAME,,}/${CKPT_KEY}/${SUB_DIR}"
    mkdir -p "$logdir"

    echo "Submitting RSNA: $jobname"
    qsub -j y -N "$jobname" \
      -o "${logdir}/${jobname}.qlog" \
      -v ARCH,DATASET_NAME,DATA_DIR,IMG_DIR,CSV_FILE,DATA_FRAC,CLIP_CKPT,LABEL,EXTRA_ARGS \
      "/restricted/projectnb/batmanlab/shawn24/PhD/Breast-CLIP-downstream/src/scripts/qsub_templates/base_classifier.qsub"
  done
done
