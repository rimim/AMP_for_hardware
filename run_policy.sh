#!/bin/bash
function usage() {
  echo >&2 "Usage: $0 --task=<task> [--run_name=<run> --checkpoint=<number> --mujoco --num_envs=<number>]"
  echo >&2
  echo >&2 "Example: $0 --task==myrobot --mujoco"
  echo >&2
}

# Parse the arguments
USE_MUJOCO=0
TASK=
RUN_NAME=
NUM_ENVS=1
CHECKPOINT_NUMBER=
while [[ "$1" =~ ^-- ]]; do
  case "$1" in
    --task=*)
      TASK="${1#*=}"
      ;;
    --checkpoint=*)
      CHECKPOINT_NUMBER="${1#*=}"
      ;;
    --num_envs=*)
      NUM_ENVS="${1#*=}"
      ;;
    --mujoco)
      USE_MUJOCO=1
      ;;
    *)
      echo "Unknown option: $1"
      usage
      ;;
  esac
  shift
done

# Check if all required arguments are provided
if [[ -z "$TASK" ]]; then
  echo "Error: Missing required --task argument."
  usage
  exit 1
fi

HISTORY_DIR="history/$TASK"
if [[ -z "$RUN_NAME" ]]; then
  LATEST_CHECKPOINT_DIR=$(find "$HISTORY_DIR" -name '*_' -type d -exec stat --format="%Y %n" {} \; | sort -n | tail -1 | cut -d" " -f2)
  RUN_NAME=$(basename $LATEST_CHECKPOINT_DIR)
echo LATEST_CHECKPOINT_DIR: $LATEST_CHECKPOINT_DIR
fi
echo RUN_NAME: $RUN_NAME
if [[ -z "$RUN_NAME" ]]; then
  echo "Error: No checkpoint history found in $HISTORY_DIR"
  exit 1
fi
if [ $USE_MUJOCO -eq 1 ]; then
  if [[ -z "$CHECKPOINT_NUMBER" ]]; then
    CHECKPOINT_POLICY=$(find "$HISTORY_DIR/$RUN_NAME" -name '*.onnx' -type f -exec stat --format='%Y %n' {} \; | sort -n | tail -1 | cut -d' ' -f2)
    if [[ -z "$CHECKPOINT_POLICY" ]]; then
      echo "Error: No checkpoint found in $HISTORY_DIR/$RUN_NAME"
      exit 1
    fi
    CHECKPOINT=$(basename "$CHECKPOINT_POLICY")
    CHECKPOINT_NUMBER="${CHECKPOINT#policy_}"
    CHECKPOINT_NUMBER="${CHECKPOINT_NUMBER%.*}"
  fi
  ONNX_POLICY="$HISTORY_DIR/$RUN_NAME"/policy_${CHECKPOINT_NUMBER}.onnx
  if [ ! -f "$ONNX_POLICY" ]; then
    echo "Error: No checkpoint found in $HISTORY_DIR/$RUN_NAME"
    exit 1
  fi
  echo Running: $ONNX_POLICY
  echo python onnx_AMP_mujoco.py -o "$ONNX_POLICY" --hide-menu
  python onnx_AMP_mujoco.py -o "$ONNX_POLICY" --hide-menu
else
  if [[ -z "$CHECKPOINT_NUMBER" ]]; then
    CHECKPOINT_POLICY=$(find "$HISTORY_DIR/$RUN_NAME" -name '*.pt' -type f -exec stat --format='%Y %n' {} \; | sort -n | tail -1 | cut -d' ' -f2)
    if [[ -z "$CHECKPOINT_POLICY" ]]; then
      echo "Error: No checkpoint found in $HISTORY_DIR/$RUN_NAME"
      exit 1
    fi
    CHECKPOINT=$(basename "$CHECKPOINT_POLICY")
    CHECKPOINT_NUMBER="${CHECKPOINT#model_}"
    CHECKPOINT_NUMBER="${CHECKPOINT_NUMBER%.*}"
  fi
  if [ ! -f logs/"$TASK"/"$RUN_NAME"/model_${CHECKPOINT_NUMBER}.pt ]; then
    mkdir -p logs/"$TASK"/"$RUN_NAME"
    rsync -auv "$HISTORY_DIR"/"$RUN_NAME"/model_${CHECKPOINT_NUMBER}.pt logs/"$TASK"/"$RUN_NAME"
  fi
  if [ ! -f logs/"$TASK"/"$RUN_NAME"/model_${CHECKPOINT_NUMBER}.pt ]; then
    echo "Error: No ${CHECKPOINT_NUMBER} checkpoint found"
    exit 1
  fi
  echo Running: $RUN_NAME/model_${CHECKPOINT_NUMBER}.pt
  python legged_gym/scripts/play.py --task="$TASK" --run_name="$RUN_NAME" --checkpoint="$CHECKPOINT_NUMBER" --num_envs=$NUM_ENVS
fi