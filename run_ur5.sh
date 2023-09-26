#!/bin/bash


init_seed=1550
repeats=5

#prev_offset=0.04
#offset=$(echo "2 * $prev_offset" | bc)
#prev_offset=$offset

for ((i=0; i<$repeats; i++)); do
    current_seed=$((init_seed + i))
    python task_ur5_visual_reacher.py --algorithm 'rad' --work_dir "/home/gautham/madi/results" --camera_id 0 --description "rad_reruns" --seed $current_seed
done


for ((i=0; i<$repeats; i++)); do
    current_seed=$((init_seed + i + repeats))
    python task_ur5_visual_reacher.py --algorithm 'madi' --work_dir "/home/gautham/madi/results" --camera_id 0 --description "overlay_aftermask" --seed $current_seed --strong_augment 'overlay' --when_augm 'after' --save_augm --save_mask

    current_seed=$((init_seed + i + 2*repeats))
    python task_ur5_visual_reacher.py --algorithm 'madi' --work_dir "/home/gautham/madi/results" --camera_id 0 --description "overlay_b-and-a-mask" --seed $current_seed --strong_augment 'overlay' --when_augm 'both' --save_augm --save_mask
done
