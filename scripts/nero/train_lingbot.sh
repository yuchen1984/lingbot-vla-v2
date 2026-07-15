#!/bin/bash
# Launch LingBot-VLA-2.0 LoRA fine-tune on a NERO task (cube|cloth), single 4090.
# Waits for (a) the 6-shard weight download to finish and (b) the GPU to free
# (the eval / other-agent RL must release it), then runs the LoRA BC train via
# torchrun --nproc-per-node 1. A smoke run (SMOKE=1) does a few steps to validate
# load + LoRA + one fwd/bwd fits VRAM (the P0 gate) before the full run.
#
#   TASK=nerocube SMOKE=1 bash scripts/nero/train_lingbot.sh     # P0 gate (~8 steps)
#   TASK=nerocube          bash scripts/nero/train_lingbot.sh     # full run
set -u

REPO=$HOME/dev/lingbot-vla-v2
TASK=${TASK:-nerocube}
CFG=$REPO/configs/vla/nero/${TASK}.yaml
MODEL_DIR=$HOME/models/lingbot-vla-v2-6b
THRESH_MIB=${THRESH_MIB:-20000}   # LoRA 6.4B bf16 (~15GB) + activations; Isaac must be OFF
LOG=$REPO/output/train_${TASK}.log
mkdir -p "$REPO/output"
log(){ echo "[train-lingbot] $(date '+%F %T') $*" | tee -a "$LOG"; }

# conda env from tools/create_train_env.sh
source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda deactivate 2>/dev/null || true
conda activate lingbotvla
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export PATH="$CUDA_HOME/bin:$PATH"
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
# We disable depth/video distillation (align_params:{}); the released ckpt still
# carries align heads/embs -> skip those extra keys in the strict post-train loader.
export LINGBOT_SKIP_ALIGN_HEADS=1
# Tactile-augmented tasks widen the canonical state (max_state_dim 55->94) with a
# tactile.position slot, so state_proj / action projections are larger than the
# released ckpt's. Zero-pad the pretrained weights up to the wider shape at load
# (preserves the pose prior, tactile columns start at zero + train from there).
case "$TASK" in
  *tac4*|*tactile*) export LINGBOT_PAD_PROJ=1
    log "tactile task: LINGBOT_PAD_PROJ=1 (zero-pad state_proj/action projections)";;
esac

# 1. Wait for all 6 weight shards.
log "waiting for 6 weight shards in $MODEL_DIR ..."
until [ "$(ls "$MODEL_DIR"/model-*-of-000*.safetensors 2>/dev/null | wc -l)" -ge 6 ] \
      && [ "$(find "$MODEL_DIR/.cache/huggingface/download" -name '*.incomplete' 2>/dev/null | wc -l)" -eq 0 ]; do
  sleep 60
done
log "weights present ($(du -sh "$MODEL_DIR" | cut -f1))"

# 2. Wait for the GPU to free.
log "waiting for >= ${THRESH_MIB} MiB free ..."
until [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')" -ge "$THRESH_MIB" ]; do
  sleep 120
done
log "GPU free — launching TASK=$TASK SMOKE=${SMOKE:-0}"

EXTRA=""
if [ "${SMOKE:-0}" = "1" ]; then
  EXTRA="--train.max_steps 8 --train.save_steps 1000000 --train.output_dir $REPO/output/${TASK}_smoke"
  log "SMOKE MODE: max_steps=8, no checkpoint save"
fi
[ -n "${MAX_STEPS:-}" ] && EXTRA="$EXTRA --train.max_steps $MAX_STEPS"

cd "$REPO"
log "=== $CFG ==="
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES NPROC_PER_NODE=1 \
  bash train.sh tasks/vla/train_lingbotvla.py "$CFG" $EXTRA >> "$LOG" 2>&1
rc=$?
log "TRAIN DONE (rc=$rc) TASK=$TASK"
