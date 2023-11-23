import torch
import time
from copy import deepcopy
import torch.nn.functional as F
from relod.algo.sac_rad_agent import SACRADLearner, SACRADPerformer
from relod.algo.models import SODAPredictor
from relod.augmentations import strong_augment
import relod.utils as utils


class SODAPerformer(SACRADPerformer):
    def __init__(self, args) -> None:
        super().__init__(args)


class SODALearner(SACRADLearner):
    """ Implementation of SODA for relod.
    Imitating code from https://github.com/nicklashansen/dmcontrol-generalization-benchmark
    """
    def __init__(self, args, performer=None) -> None:
        super().__init__(args, performer)
        assert performer != None, "SGQN needs the performer to be SGQNPerformer"
        assert 'conv' in self._args.net_params, "SGQN needs image input"

        self.soda_predictor = SODAPredictor(self._critic.encoder, self._args.soda_projection_dim).to(self._args.device)
        self.soda_predictor_target = deepcopy(self.soda_predictor)

        self._aux_optimizer = torch.optim.Adam(
            self.soda_predictor.parameters(), lr=self._args.aux_lr, betas=(0.9, 0.999))
        self.train()

    def train(self, is_training=True):
        self._performer.train(is_training)
        self.soda_predictor.train(is_training)
        self.is_training = is_training

    def compute_soda_loss(self, x0, x1, propris):
        h0 = self.soda_predictor(x0, propris)
        with torch.no_grad():
            h1 = self.soda_predictor_target.encoder(x1, propris)
        h0 = F.normalize(h0, p=2, dim=1)
        h1 = F.normalize(h1, p=2, dim=1)
        return F.mse_loss(h0, h1)

    def _update_aux(self, images, propris):
        # images = replay_buffer.sample_soda(self.soda_batch_size)  # we don't use this, instead we
        # train on the same batch that was sampled from the replay buffer earlier

        images_augm = strong_augment(images, self._args.strong_augment)
        soda_loss = self.compute_soda_loss(images_augm, images, propris)

        self._aux_optimizer.zero_grad()
        soda_loss.backward()
        self._aux_optimizer.step()

        utils.soft_update_params(self.soda_predictor, self.soda_predictor_target, self._args.soda_tau)

    def _update(self, images, propris, actions, rewards, next_images, next_propris, dones):
        tic = time.time()
        if images is not None:
            images = torch.as_tensor(images, device=self._args.device).float()
            next_images = torch.as_tensor(next_images, device=self._args.device).float()
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

        # SODA specific update
        if self._num_updates % self._args.aux_update_freq == 0:
            aux_stats = self._update_aux(images, propris)
            stats = {**stats, **aux_stats}

        stats['train/batch_reward'] = rewards.mean().item()
        stats['train/num_updates'] = self._num_updates
        self._num_updates += 1
        if self._num_updates % 100 == 0:
            print("Update {} took {:.4f}s to update the model".format(self._num_updates, time.time()-tic))
        return stats
