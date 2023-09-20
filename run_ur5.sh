#!/bin/bash

#algo="madi"
#for seed in {48..50}
#do
#python task_ur5_visual_reacher.py --algorithm $algo --batch_size 128 --env_steps 100000 --work_dir "/home/gautham/madi/results/${algo}_clean" --seed $seed
#done


init_seed=300
repeats=3

for ((i=0; i<$repeats; i++)); do
    current_seed=$((init_seed + i))
    python task_ur5_visual_reacher.py --algorithm 'madi' --work_dir "/home/gautham/madi/results" --camera_id 0 --description "eval_madi" --seed $current_seed  --init_steps 1000 --device "cuda:0" --save_mask --strong_augment 'conv'
    current_seed=$((init_seed + i + repeats))
    python task_ur5_visual_reacher.py --algorithm 'rad' --work_dir "/home/gautham/madi/results" --camera_id 0 --description "eval_rad" --seed $current_seed  --init_steps 1000 --device "cuda:0" --save_mask --strong_augment 'conv'
done
