#!/bin/bash

algo="madi"
for seed in {48..50}
do
python task_ur5_visual_reacher.py --algorithm $algo --batch_size 128 --env_steps 100000 --work_dir "/home/gautham/madi/results/${algo}_clean" --seed $seed
done
z
