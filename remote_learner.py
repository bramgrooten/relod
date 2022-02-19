import torch
import argparse
from algo.remote_wrapper import RemoteWrapper
from algo.sac_rad_agent import SACRADLearner, SACRADPerformer
import socket
from algo.comm import recv_message
from logger import Logger
import time
import utils
import os

def parse_args():
    parser = argparse.ArgumentParser()

    # server
    parser.add_argument('--args_port', default=9630, type=int)
    # agent
    parser.add_argument('--port', default=9876, type=int)
    # misc
    parser.add_argument('--device', default='cuda:0', type=str)

    args = parser.parse_args()
    return args

def main():
    server_args = parse_args()

    server_args_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_args_sock.bind(('', server_args.args_port))

    server_args_sock.listen(1)
    print('agent args socket created, listening...')

    (args_sock, address) = server_args_sock.accept()
    print('Connection accepted, ip:', address)

    args = recv_message(args_sock)
    args_sock.close()
    utils.set_seed_everywhere(args.seed)

    utils.make_dir(args.work_dir)

    model_dir = utils.make_dir(os.path.join(args.work_dir, 'model'))
    args.model_dir = model_dir

    if server_args.device is '':
        args.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    else:
        args.device = server_args.device

    performer = SACRADPerformer(args)
    learner = SACRADLearner(performer, args)
    remote_wrapper = RemoteWrapper(server_args.port, performer=None, learner=learner)

    L = Logger(args.work_dir, use_tb=args.save_tb)

    episode, episode_reward, episode_step, done = 0, 0, 0, True
    episode_length_step = int(args.episode_length_time / args.dt)
    (image, propri) = remote_wrapper.receive_init_ob()
    start_time = time.time()
    for step in range(args.env_steps + args.init_steps):
        action = remote_wrapper.sample_action((image, propri), step)
        
        (reward, (next_image, next_propri), done) = remote_wrapper.receive_sample()
        
        episode_reward += reward
        episode_step += 1

        learner.push_sample((image, propri), action, reward, (next_image, next_propri), done)

        if done or (episode_step == episode_length_step): # set time out here
            L.log('train/duration', time.time() - start_time, step)
            L.log('train/episode_reward', episode_reward, step)
            L.dump(step)
            learner.pause_update()
            (next_image, next_propri) = remote_wrapper.receive_init_ob()
            learner.resume_update()
            episode_reward = 0
            episode_step = 0
            episode += 1
            L.log('train/episode', episode, step)
            start_time = time.time()

        stat = remote_wrapper.update_policy(step)
        if stat is not None:
            for k, v in stat.items():
                L.log(k, v, step)

        (image, propri) = (next_image, next_propri)

        if args.save_model and (step+1) % args.save_model_freq == 0:
            performer.save_policy_to_file(step)

    performer.save_policy_to_file(step)
    learner.pause_update()
    remote_wrapper.close()
    print('Training finished')

if __name__ == '__main__':
    main()
