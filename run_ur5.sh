#!/bin/bash

#algo="madi"
#for seed in {48..50}
#do
#python task_ur5_visual_reacher.py --algorithm $algo --batch_size 128 --env_steps 100000 --work_dir "/home/gautham/madi/results/${algo}_clean" --seed $seed
#done


init_seed=257
repeats=1

for ((i=0; i<$repeats; i++)); do
    current_seed=$((init_seed + i))
    python task_ur5_visual_reacher.py --algorithm 'madi' --seed $current_seed --work_dir "/home/gautham/madi/results/madi" --description "video-bg-madi" --train_env_mode "video_easy_5" --save_mask
done

# init_seed=260

# for ((i=0; i<$repeats; i++)); do
#     current_seed=$((init_seed + i))
#     python task_ur5_visual_reacher.py --algorithm 'rad' --seed $current_seed --work_dir "/home/gautham/madi/results/sac_rad" --description "video-bg-sac-rad" --train_env_mode "video_easy_5"
# done

# init_seed=265

# for ((i=0; i<$repeats; i++)); do
#     current_seed=$((init_seed + i))
#     python task_ur5_visual_reacher.py --algorithm 'rad' --seed $current_seed --work_dir "/home/gautham/madi/results/sac_only" --rad_offset 0 --description "video-bg-sac-only" --train_env_mode "video_easy_5"
# done

