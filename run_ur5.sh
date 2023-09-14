#!/bin/bash

#algo="madi"
#for seed in {48..50}
#do
#python task_ur5_visual_reacher.py --algorithm $algo --batch_size 128 --env_steps 100000 --work_dir "/home/gautham/madi/results/${algo}_clean" --seed $seed
#done


init_seed=65
repeats=3

for ((i=0; i<$repeats; i++)); do
    current_seed=$((init_seed + i))
    python task_ur5_visual_reacher.py --algorithm 'rad' --background_color 'black' --seed $current_seed --work_dir "/home/bgrooten/code/relod/results/rad_black_bg" --camera_id 1
    current_seed=$((init_seed + i + repeats))
    python task_ur5_visual_reacher.py --algorithm 'madi' --background_color 'black' --save_mask --seed $current_seed --work_dir  "/home/bgrooten/code/relod/results/madi_black_bg" --camera_id 1
done


