#!/bin/bash

#algo="madi"
#for seed in {48..50}
#do
#python task_ur5_visual_reacher.py --algorithm $algo --batch_size 128 --env_steps 100000 --work_dir "/home/gautham/madi/results/${algo}_clean" --seed $seed
#done


init_seed=1500
repeats=5

for ((i=0; i<$repeats; i++)); do
    current_seed=$((init_seed + i))
    python task_ur5_visual_reacher.py --algorithm 'madi' --work_dir "/home/gautham/madi/results" --camera_id 0 --description "conv_beforemask_noanneal" --seed $current_seed --strong_augment 'conv' --save_augm --save_mask

    current_seed=$((init_seed + i + repeats))
    offset=$(( i*0.02 ))
    python task_ur5_visual_reacher.py --algorithm 'rad' --work_dir "/home/gautham/madi/results" --camera_id 0 --description "rad_offset_higher" --seed $current_seed --rad_offset $offset

    # current_seed=$((init_seed + i + repeats + repeats))
    # python task_ur5_visual_reacher.py --algorithm 'madi' --work_dir "/home/gautham/madi/results" --camera_id 0 --description "overlay_beforemask_anneal10k" --seed $current_seed --strong_augment 'overlay' --save_augm --save_mask --anneal_masker_lr 'cosine10k'

    # current_seed=$((init_seed + i + repeats + repeats + repeats))
    # python task_ur5_visual_reacher.py --algorithm 'madi' --work_dir "/home/gautham/madi/results" --camera_id 0 --description "conv_beforemask_anneal10k" --seed $current_seed --strong_augment 'conv' --save_augm --save_mask --anneal_masker_lr 'cosine10k'
done
