#!/bin/bash

### USER INPUT SECTION — set your checkpoint here:
CKPT_KEY="upmc_embed_bu_mayo"
CLIP_CKPT="/restricted/projectnb/batmanlab/shawn24/PhD/Breast-CLIP/src/codebase/outputs/upmc_embed_bu_mayo_clip/b5_detector_n_modernbert_2048/checkpoints/fold_0/MammoCLIP_Mayo_UPMC_EMBED_BU_epoch3_BatmanlabTrained.tar"



### Define global log base path
LOG_BASE="/restricted/projectnb/batmanlab/shawn24/PhD/Breast-CLIP-downstream/src/scc_logs/breast"
#SUB_DIR="epoch3/mayo_trained"
SUB_DIR="epoch3/batmanlab_trained"

source "/restricted/projectnb/batmanlab/shawn24/PhD/Breast-CLIP-downstream/src/scripts/configs/vindr_configs.sh"

VINDR_LABELS=("Mass" "Suspicious Calcification")
VINDR_DATA_FRACS=("1.0" "0.5" "0.1")
#VINDR_DATA_FRACS=("1.0")

###########################################
## VINDr DETECTION
###########################################

VINDR_DETECTION_ARCHS=("breast_clip_det_b5")
FREEZE_OPTIONS=("n" "y")

for LABEL in "${VINDR_LABELS[@]}"; do
  for FREEZE in "${FREEZE_OPTIONS[@]}"; do
    if [[ "$FREEZE" == "y" ]]; then
      FRACS=("1.0")
    else
      FRACS=("1.0" "0.5" "0.1")
    fi

    for FRAC in "${FRACS[@]}"; do
      for ARCH in "${VINDR_DETECTION_ARCHS[@]}"; do
        export CLIP_CKPT="$CLIP_CKPT"
        export ARCH="$ARCH"
        export DATA_FRAC="$FRAC"
        export FREEZE_BACKBONE="$FREEZE"
        export DATASET_NAME="$DATASET_NAME"
        export DATA_DIR="$DATA_DIR"
        export IMG_DIR="$IMG_DIR"
        export CSV_FILE="$CSV_FILE"
        export LABEL="$LABEL"

        echo $ARCH
        echo $DATASET_NAME
        echo $DATA_DIR
        echo $IMG_DIR
        echo $CSV_FILE
        echo $DATA_FRAC
        echo $LABEL
        echo $CLIP_CKPT

        timestamp=$(date +"%Y-%m-%d-%H-%M-%S-%N")
        label_tag="${LABEL// /_}"  # Replace spaces for safe jobname/log path
        jobname="${ARCH//[^a-zA-Z0-9]/_}_det_${label_tag}_${FREEZE}_${FRAC}_${CKPT_KEY}_${timestamp}"
        logdir="${LOG_BASE}/detection_vindr/${CKPT_KEY}/${SUB_DIR}"
        mkdir -p "$logdir"

        echo "Submitting VinDr DETECTION: $jobname"
        qsub -j y -N "$jobname" \
             -o "${logdir}/${jobname}.qlog" \
             -v ARCH,CLIP_CKPT,DATASET_NAME,DATA_DIR,IMG_DIR,CSV_FILE,DATA_FRAC,FREEZE_BACKBONE,LABEL \
             qsub_templates/base_detector.qsub
      done
    done
  done
done

