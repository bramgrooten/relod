#!/bin/bash
# --work_dir "/home/bgrooten/code/results"
# --work_dir "/home/gautham/madi/results"


#init_seed=1940
#repeats=4
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


# init_seed=2000
# repeats=1
# for ((i=0; i<$repeats; i++)); do
#     current_seed=$((init_seed + i))
#     python task_ur5_visual_reacher.py --algorithm 'madi' --work_dir "/home/gautham/madi/results" --camera_id 0 --description "overlay_longrun" --seed $current_seed --strong_augment 'overlay' --env_steps 1000000 --save_augm --save_mask
# done


#init_seed=1960
#repeats=5
#for ((i=0; i<$repeats; i++)); do
#    current_seed=$((init_seed + i))
#    python transfer_ur5_visual_reacher.py --algorithm 'madi' --work_dir "/home/bgrooten/code/results" --camera_id 0 --description "transfer_madi_overlay_reinit" --seed $current_seed --strong_augment 'overlay' --reinit_policy --save_augm --save_mask
#done
#


# init_seed=2080
# repeats=3

# for ((i=0; i<$repeats; i++)); do
#     current_seed=$((init_seed + i))
#     python task_ur5_visual_reacher.py --algorithm 'svea' --work_dir "/home/gautham/madi/results/" --mode 'l' --train_env_mode "clean" --seed $current_seed --use_sparse_reward --description "dm_control_rewards_svea_overlay" --save_mask --env_steps 200100 --strong_augment overlay --save_augm

#     current_seed=$((init_seed + i + repeats))
#     python task_ur5_visual_reacher.py --algorithm 'madi' --work_dir "/home/gautham/madi/results/" --mode 'l' --train_env_mode "clean" --seed $current_seed --use_sparse_reward --description "dm_control_rewards_madi_overlay" --save_mask --env_steps 200100 --strong_augment overlay --save_augm

#     current_seed=$((init_seed + i + 2*repeats))
#     python task_ur5_visual_reacher.py --algorithm 'rad' --rad_offset 0 --work_dir "/home/gautham/madi/results/" --mode 'l' --train_env_mode "clean" --seed $current_seed --use_sparse_reward --description "dm_control_rewards_sac_only" --save_mask --env_steps 200100
# done


# python task_ur5_visual_reacher.py --algorithm 'rad' --work_dir "/home/gautham/madi/results/" --mode 'l' --train_env_mode "clean" --seed 2060 --use_sparse_reward --description "dm_control_rewards_rad" --env_steps 200100


# Last two runs
# init_seed=4010
# repeats=5

# for ((i=0; i<$repeats; i++)); do
#     current_seed=$((init_seed + i))
#     python task_ur5_visual_reacher.py --algorithm 'madi' --work_dir "/home/gautham/madi/results/" --mode 'l' --train_env_mode "clean" --seed $current_seed --use_sparse_reward --description "dm_control_rewards_madi_overlay" --save_mask --env_steps 200100 --strong_augment overlay --save_augm

#     current_seed=$((init_seed + i + repeats))
#     python task_ur5_visual_reacher.py --algorithm 'svea' --work_dir "/home/gautham/madi/results/" --mode 'l' --train_env_mode "clean" --seed $current_seed --use_sparse_reward --description "dm_control_rewards_svea_overlay" --env_steps 200100 --strong_augment overlay --save_augm

#     current_seed=$((init_seed + i + 2*repeats))
#     python task_ur5_visual_reacher.py --algorithm 'rad' --rad_offset 0 --work_dir "/home/gautham/madi/results/" --mode 'l' --train_env_mode "clean" --seed $current_seed --use_sparse_reward --description "dm_control_rewards_sac_only" --env_steps 200100

#     current_seed=$((init_seed + i + 3*repeats))
#     python task_ur5_visual_reacher.py --algorithm 'rad' --work_dir "/home/gautham/madi/results/" --mode 'l' --train_env_mode "clean" --seed $current_seed --use_sparse_reward --description "dm_control_rewards_rad" --env_steps 200100
# done


init_seed=4010
repeats=5
for ((i=0; i<$repeats; i++)); do
    current_seed=$((init_seed + i))
    python task_ur5_visual_reacher.py --algorithm drq --work_dir "/home/bgrooten/code/results/" --seed $current_seed --description "drq-runs"
done

init_seed=4020
repeats=5
for ((i=0; i<$repeats; i++)); do
    current_seed=$((init_seed + i))
    python task_ur5_visual_reacher.py --algorithm drq --work_dir "/home/bgrooten/code/results/" --seed $current_seed --description "drq-sparse-rew" --use_sparse_reward --env_steps 200100
done

