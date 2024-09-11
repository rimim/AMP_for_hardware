#!/bin/bash
nohup python legged_gym/scripts/train.py --task=bdx_amp --num_envs 22000 --headless --rl_device=cuda:1 --sim_device=cuda:1 --headless > output.log 2>&1 &
nc -zv localhost 6006 > /dev/null
if [ $? -ne 0 ]; then
  echo "TensorBoard is not running. Starting TensorBoard..."
  nohup tensorboard --host 0.0.0.0 --logdir logs --port 6006 > tensorboard_output.log 2>&1 &
else
  echo "TensorBoard is already running on localhost:6006"
fi
tail -f output.log
