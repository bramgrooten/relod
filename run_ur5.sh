#!/bin/bash

#algo="madi"
#for seed in {48..50}
#do
#python task_ur5_visual_reacher.py --algorithm $algo --batch_size 128 --env_steps 100000 --work_dir "/home/gautham/madi/results/${algo}_clean" --seed $seed
#done


init_seed=1320
repeats=5

for ((i=0; i<$repeats; i++)); do
    current_seed=$((init_seed + i))
    python task_ur5_visual_reacher.py --algorithm 'rad' --work_dir "/home/gautham/madi/results" --camera_id 0 --description "eval_sac_only" --seed $current_seed --rad_offset 0

    current_seed=$((init_seed + i + repeats))
    python task_ur5_visual_reacher.py --algorithm 'svea' --work_dir "/home/gautham/madi/results" --camera_id 0 --description "eval_svea" --seed $current_seed --strong_augment 'conv' 
done
