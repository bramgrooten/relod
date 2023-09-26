#!/bin/bash

#algo="madi"
#for seed in {48..50}
#do
#python task_ur5_visual_reacher.py --algorithm $algo --batch_size 128 --env_steps 100000 --work_dir "/home/gautham/madi/results/${algo}_clean" --seed $seed
#done


for current_seed in 1600 1601 1602
do
    python task_ur5_visual_reacher.py --algorithm 'madi' --work_dir "/home/gautham/madi/results" --camera_id 0 --description "overlay_beforemask_noanneal_masker_lr-1e-3" --seed $current_seed --strong_augment 'overlay' --save_augm --save_mask --masker_lr 0.001
done
