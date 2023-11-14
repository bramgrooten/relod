import torch
import time
from relod.algo.sac_rad_agent import SACRADLearner, SACRADPerformer
from relod.augmentations import random_shift


class SACDrQPerformer(SACRADPerformer):
    def __init__(self, args) -> None:
        super().__init__(args)
        assert self._args.rad_offset == 0, "DrQ needs rad_offset == 0"


class SACDrQLearner(SACRADLearner):
    def __init__(self, args, performer=None) -> None:
        super().__init__(args, performer)
        assert self._args.rad_offset == 0, "DrQ needs rad_offset == 0"
        assert performer != None, "DrQ needs the performer to be SACDrQPerformer"
        assert 'conv' in self._args.net_params, "DrQ needs image input"

    def _update(self, images, propris, actions, rewards, next_images, next_propris, dones):
        tic = time.time()
        # regular update of SAC_RAD, sequentially augment data and train
        if images is not None:
            images = torch.as_tensor(images, device=self._args.device).float()
            next_images = torch.as_tensor(next_images, device=self._args.device).float()

            # DrQ uses random shift augmentation
            images = random_shift(images)
            next_images = random_shift(next_images)

        if propris is not None:
            propris = torch.as_tensor(propris, device=self._args.device).float()
            next_propris = torch.as_tensor(next_propris, device=self._args.device).float()
        actions = torch.as_tensor(actions, device=self._args.device)
        rewards = torch.as_tensor(rewards, device=self._args.device)
        dones = torch.as_tensor(dones, device=self._args.device)
        
        stats = self._update_critic(images, propris, actions, rewards, next_images, next_propris, dones)
        if self._num_updates % self._args.actor_update_freq == 0:
            actor_stats = self._update_actor_and_alpha(images, propris)
            stats = {**stats, **actor_stats}
        if self._num_updates % self._args.critic_target_update_freq == 0:
            self._soft_update_target()
        stats['train/batch_reward'] = rewards.mean().item()
        stats['train/num_updates'] = self._num_updates
        self._num_updates += 1
        if self._num_updates % 100 == 0:
            print("Update {} took {:.4f}s to update the model".format(self._num_updates, time.time()-tic))
        return stats
