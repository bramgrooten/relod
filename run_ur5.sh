#!/bin/bash


init_seed=1940
repeats=4

#prev_offset=0.04
#offset=$(echo "2 * $prev_offset" | bc)
#prev_offset=$offset

# for ((i=0; i<$repeats; i++)); do
#     current_seed=$((init_seed + i))
#     python task_ur5_visual_reacher.py --algorithm 'rad' --work_dir "/home/gautham/madi/results" --camera_id 0 --description "rad_reruns" --seed $current_seed
# done


# for ((i=0; i<$repeats; i++)); do
#     current_seed=$((init_seed + i))
#     python task_ur5_visual_reacher.py --algorithm 'madi' --work_dir "/home/bgrooten/code/results" --camera_id 0 --description "conv_aftermask" --seed $current_seed --strong_augment 'conv' --when_augm 'after' --save_augm --save_mask

#     # current_seed=$((init_seed + i + repeats))
#     # python task_ur5_visual_reacher.py --algorithm 'madi' --work_dir "/home/gautham/madi/results" --camera_id 0 --description "overlay_b-and-a-mask" --seed $current_seed --strong_augment 'overlay' --when_augm 'both' --save_augm --save_mask
# done

# --work_dir "/home/bgrooten/code/results" 
# --work_dir "/home/gautham/madi/results"

init_seed=1950
repeats=3
for ((i=0; i<$repeats; i++)); do
    current_seed=$((init_seed + i))
    python task_ur5_visual_reacher.py --algorithm 'madi' --work_dir "/home/bgrooten/code/results" --camera_id 0 --description "overlay_aftermask" --seed $current_seed --strong_augment 'overlay' --when_augm 'after' --save_augm --save_mask
done



init_seed=1960
repeats=5
for ((i=0; i<$repeats; i++)); do
    current_seed=$((init_seed + i))
    python transfer_ur5_visual_reacher.py --algorithm 'madi' --work_dir "/home/bgrooten/code/results" --camera_id 0 --description "transfer_madi_overlay_reinit" --seed $current_seed --strong_augment 'overlay' --reinit_policy --save_augm --save_mask
done


init_seed=1970
repeats=5
for ((i=0; i<$repeats; i++)); do
    current_seed=$((init_seed + i))
    python transfer_ur5_visual_reacher.py --algorithm 'madi' --work_dir "/home/bgrooten/code/results" --camera_id 0 --description "transfer_madi_convafter_reinit" --seed $current_seed --strong_augment 'conv' --when_augm 'after' --reinit_policy --save_augm --save_mask

    current_seed=$((init_seed + i + repeats))
    python transfer_ur5_visual_reacher.py --algorithm 'madi' --work_dir "/home/bgrooten/code/results" --camera_id 0 --description "transfer_madi_conv_reinit" --seed $current_seed --strong_augment 'conv' --reinit_policy --save_augm --save_mask
done


