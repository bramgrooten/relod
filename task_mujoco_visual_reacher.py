import torch
import argparse
import time
from relod.algo.local_wrapper import LocalWrapper
from relod.algo.sac_rad_agent import SACRADPerformer, SACRADLearner
from relod.algo.sac_madi_agent import MaDiPerformer, MaDiLearner
from relod.algo.sac_svea_agent import SVEAPerformer, SVEALearner
import relod.utils as utils
from relod.envs.mujoco_visual_reacher.env import ReacherWrapper
from relod.algo.comm import MODE
from relod.logger import Logger
import os
import wandb


config = {
    'conv': [
        # in_channel, out_channel, kernel_size, stride
        [-1, 32, 3, 2],
        [32, 32, 3, 2],
        [32, 32, 3, 2],
        [32, 32, 3, 1],
    ],
    
    'latent': 50,

    'mlp': [
        [-1, 1024],
        [1024, 1024],
        [1024, -1]
    ],
}

def parse_args():
    parser = argparse.ArgumentParser()
    # environment
    parser.add_argument('--target_type', default='visual_reacher', type=str)
    parser.add_argument('--image_height', default=125, type=int)
    parser.add_argument('--image_width', default=200, type=int)
    parser.add_argument('--stack_frames', default=3, type=int)
    parser.add_argument('--tol', default=0.036, type=float)
    parser.add_argument('--image_period', default=1, type=int)
    parser.add_argument('--episode_length_time', default=50, type=int)
    parser.add_argument('--dt', default=1, type=int)
    # replay buffer
    parser.add_argument('--replay_buffer_capacity', default=100000, type=int)
    parser.add_argument('--rad_offset', default=0.01, type=float)
    # train
    parser.add_argument('--algorithm', default='rad', type=str, help="Algorithms in ['rad', 'madi', 'svea']")
    parser.add_argument('--init_steps', default=1000, type=int)
    parser.add_argument('--env_steps', default=20000, type=int)
    parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--async_mode', default=True, action='store_true')
    parser.add_argument('--max_updates_per_step', default=1, type=float)
    parser.add_argument('--update_every', default=50, type=int)
    parser.add_argument('--update_epochs', default=50, type=int)
    # critic
    parser.add_argument('--critic_lr', default=1e-3, type=float)
    parser.add_argument('--critic_tau', default=0.01, type=float)
    parser.add_argument('--critic_target_update_freq', default=1, type=int)
    parser.add_argument('--bootstrap_terminal', default=0, type=int)
    # actor
    parser.add_argument('--actor_lr', default=1e-3, type=float)
    parser.add_argument('--actor_update_freq', default=1, type=int)
    # encoder
    parser.add_argument('--encoder_tau', default=0.05, type=float)
    # sac
    parser.add_argument('--discount', default=1., type=float)
    parser.add_argument('--init_temperature', default=0.1, type=float)
    parser.add_argument('--alpha_lr', default=1e-4, type=float)
    # madi
    parser.add_argument('--masker_lr', default=3e-4, type=float)  # was 1e-3 in MaDi work, but 3e-4 is standard here. Can try 1e-3 later
    parser.add_argument('--save_mask', default=False, action='store_true')
    parser.add_argument('--save_mask_freq', default=1000, type=int)
    parser.add_argument('--strong_augment', default='none', type=str, help="Augmentations in ['none', 'conv', 'overlay']")
    parser.add_argument('--anneal_masker_lr', default='none', type=str, help="['none', 'cosine', 'cosine10k']")
    parser.add_argument('--when_augm', default='before', type=str,
                        help="When to augment. Before or after masking (or both)", choices=['before', 'after', 'both'])
    # agent
    parser.add_argument('--remote_ip', default='localhost', type=str)
    parser.add_argument('--port', default=9876, type=int)
    parser.add_argument('--mode', default='l', type=str, help="Modes in ['r', 'l', 'rl'] ")
    # misc
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--work_dir', default='.', type=str)
    parser.add_argument('--save_tb', default=False, action='store_true')
    parser.add_argument('--save_model', default=False, action='store_true')
    parser.add_argument('--save_model_freq', default=1000, type=int)
    parser.add_argument('--load_model', default=-1, type=int)
    parser.add_argument('--device', default='cuda:0', type=str)
    parser.add_argument('--lock', default=False, action='store_true')
    parser.add_argument('--save_path', default='', type=str, help="For saving SAC buffer")
    parser.add_argument('--load_path', default='', type=str, help="Path to SAC buffer file")
    parser.add_argument('--wandb_mode', default='online', type=str, help="Either online, offline, or disabled")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    if args.mode == 'r':
        mode = MODE.REMOTE_ONLY
    elif args.mode == 'l':
        mode = MODE.LOCAL_ONLY
    elif args.mode == 'rl':
        mode = MODE.REMOTE_LOCAL
    else:
        raise  NotImplementedError()

    if args.device is '':
        args.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    if not 'conv' in config:
        image_shape = (0, 0, 0)
    else: 
        image_shape = (3*args.stack_frames, args.image_height, args.image_width)

    args.work_dir += f'/results/{args.target_type}_' \
                     f'seed={args.seed}_' \
                     f'tol={args.tol}/'

    utils.make_dir(args.work_dir)

    model_dir = utils.make_dir(os.path.join(args.work_dir, 'model'))
    args.model_dir = model_dir
    L = Logger(args.work_dir, use_tb=args.save_tb)

    env = ReacherWrapper(args.tol, image_shape, args.image_period, use_ground_truth=True)
    utils.set_seed_everywhere(args.seed, env)

    args.image_shape = env.image_space.shape
    args.proprioception_shape = env.proprioception_space.shape
    args.action_shape = env.action_space.shape
    args.net_params = config
    args.env_action_space = env.action_space

    # start a new wandb run to track this script
    wandb.init(
        project="madi",
        config=vars(args),
        name=f"mujocoReacher-{args.algorithm}-seed{args.seed}-batch{args.batch_size}",
        entity="gauthamv",
        mode=args.wandb_mode,
    )

    episode_length_step = int(args.episode_length_time / args.dt)
    agent = LocalWrapper(episode_length_step, mode, remote_ip=args.remote_ip, port=args.port)
    agent.send_data(args)

    if args.algorithm == 'rad':
        agent.init_performer(SACRADPerformer, args)
        agent.init_learner(SACRADLearner, args, agent.performer)
    elif args.algorithm == 'madi':
        agent.init_performer(MaDiPerformer, args)
        agent.init_learner(MaDiLearner, args, agent.performer)
    elif args.algorithm == 'svea':
        agent.init_performer(SVEAPerformer, args)
        agent.init_learner(SVEALearner, args, agent.performer)
    else:
        raise NotImplementedError()

    # sync initial weights with remote
    agent.apply_remote_policy(block=True)
    
    episode, episode_reward, episode_step, done = 0, 0, 0, True
    image, propri = env.reset()
    agent.send_init_ob((image, propri))
    start_time = time.time()
    for step in range(args.env_steps + args.init_steps):
        action = agent.sample_action((image, propri))

        next_image, next_propri, reward, done, _ = env.step(action)

        episode_reward += reward
        episode_step += 1

        agent.push_sample((image, propri), action, reward, (next_image, next_propri), done)
        
        if done or (episode_step == episode_length_step): # set time out here
            if mode == MODE.LOCAL_ONLY:
                L.log('train/duration', time.time() - start_time, step)
                L.log('train/episode_reward', episode_reward, step)
                L.dump(step)
                L.log('train/episode', episode+1, step)

            next_image, next_propri = env.reset()
            agent.send_init_ob((next_image, next_propri))
            episode_reward = 0
            episode_step = 0
            episode += 1
            start_time = time.time()
        
        stat = agent.update_policy(step)
        if stat is not None:
            for k, v in stat.items():
                L.log(k, v, step)
        
        image = next_image
        propri = next_propri

        if args.save_model and (step+1) % args.save_model_freq == 0:
            agent.save_policy_to_file(args.model_dir, step)
            agent.save_buffer()


        time.sleep(0.04)

    if args.save_model:
        agent.save_policy_to_file(args.model_dir, step)
    # Clean up
    agent.close()
    env.close()
    print('Train finished')


if __name__ == '__main__':
    main()
