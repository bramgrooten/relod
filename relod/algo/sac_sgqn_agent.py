import time
import random
import torch
import torch.nn.functional as F
from captum.attr import GuidedBackprop
from relod.algo.sac_rad_agent import SACRADLearner, SACRADPerformer
from relod.augmentations import strong_augment
from relod.algo.models import AttributionPredictor, ModelWrapper


class SGQNPerformer(SACRADPerformer):
    def __init__(self, args) -> None:
        super().__init__(args)


class SGQNLearner(SACRADLearner):
    def __init__(self, args, performer=None) -> None:
        super().__init__(args, performer)
        assert performer != None, "SGQN needs the performer to be SGQNPerformer"
        assert 'conv' in self._args.net_params, "SGQN needs image input"

        self._init_optimizers()

    def _init_optimizers(self):
        # trying to put it here instead of in __init__ to debug weird error saying:
        # AttributeError: 'SGQNLearner' object has no attribute 'attribution_predictor'
        self.attribution_predictor = AttributionPredictor(self._args.action_shape[0], self._critic.encoder).to(self._args.device)

        self._actor_optimizer = torch.optim.Adam(
            self._actor.parameters(), lr=self._args.actor_lr, betas=(0.9, 0.999))
        self._critic_optimizer = torch.optim.Adam(
            self._critic.parameters(), lr=self._args.critic_lr, betas=(0.9, 0.999),
            weight_decay=self._args.critic_weight_decay)  # SGQN needs weight decay on critic
        self._log_alpha_optimizer = torch.optim.Adam(
            [self._log_alpha], lr=self._args.alpha_lr, betas=(0.5, 0.999))
        self._aux_optimizer = torch.optim.Adam(
            self.attribution_predictor.parameters(), lr=self._args.aux_lr, betas=(0.9, 0.999))

    def _compute_attribution(self, images, proprioceptions, actions):
        model = ModelWrapper(self._critic, propris=proprioceptions, action=actions)
        gbp = GuidedBackprop(model)
        attribution = gbp.attribute(images)
        return attribution

    def _compute_attribution_mask(self, obs_grad, quantile=0.95):
        mask = []
        for i in [0, 3, 6]:
            attributions = obs_grad[:, i:i + 3].abs().max(dim=1)[0]
            q = torch.quantile(attributions.flatten(1), quantile, 1)
            mask.append((attributions >= q[:, None, None]).unsqueeze(1).repeat(1, 3, 1, 1))
        return torch.cat(mask, dim=1)

    def _compute_attribution_loss(self, images, actions, mask):
        mask = mask.float()
        attrib = self.attribution_predictor(images.detach(), actions.detach())
        aux_loss = F.binary_cross_entropy_with_logits(attrib, mask.detach())
        return attrib, aux_loss

    def _update_aux(self, images, propris, actions):
        """Updates the auxiliary network of SGQN: the AttributionPredictor."""
        obs_grad = self._compute_attribution(images, propris, actions.detach())
        mask = self._compute_attribution_mask(obs_grad, self._args.sgqn_quantile)

        s_tilde = strong_augment(images, self._args.strong_augment)
        self._aux_optimizer.zero_grad()
        pred_attrib, aux_loss = self._compute_attribution_loss(s_tilde, actions, mask)
        aux_loss.backward()
        self._aux_optimizer.step()
        return {'train/aux_loss': aux_loss.item()}

    def _update_critic(self, images, proprioceptions, actions, rewards, next_images, next_proprioceptions, dones):
        with torch.no_grad():
            _, policy_actions, log_pis, _ = self._actor(next_images, next_proprioceptions)
            target_Q1, target_Q2 = self._critic_target(next_images, next_proprioceptions, policy_actions)
            target_V = torch.min(target_Q1, target_Q2) - self._alpha.detach() * log_pis
            if self._args.bootstrap_terminal:
                # enable infinite bootstrap
                target_Q = rewards + (self._args.discount * target_V)
            else:
                target_Q = rewards + ((1.0 - dones) * self._args.discount * target_V)

        # get current Q estimates
        current_Q1, current_Q2 = self._critic(images, proprioceptions, actions, detach_encoder=False)

        critic_loss = torch.mean((current_Q1 - target_Q) ** 2 + (current_Q2 - target_Q) ** 2)

        # SGQN specific, adding a "consistency term" to the critic loss
        obs_grad = self._compute_attribution(images, proprioceptions, actions.detach())
        mask = self._compute_attribution_mask(obs_grad, self._args.sgqn_quantile)
        masked_images = images * mask
        masked_images[mask < 1] = random.uniform(images.view(-1).min(), images.view(-1).max())
        masked_Q1, masked_Q2 = self._critic(masked_images, proprioceptions, actions, detach_encoder=False)
        critic_loss += 0.5 * (torch.mean((current_Q1 - masked_Q1) ** 2 + (current_Q2 - masked_Q2) ** 2))

        # Optimize the critic
        self._critic_optimizer.zero_grad()
        critic_loss.backward()
        self._critic_optimizer.step()
        return {'train/critic_loss': critic_loss.item()}

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

        # SGQN specific update
        if self._num_updates % self._args.aux_update_freq == 0:
            aux_stats = self._update_aux(images, propris, actions)
            stats = {**stats, **aux_stats}

        stats['train/batch_reward'] = rewards.mean().item()
        stats['train/num_updates'] = self._num_updates
        self._num_updates += 1
        if self._num_updates % 100 == 0:
            print("Update {} took {:.4f}s to update the model".format(self._num_updates, time.time()-tic))
        return stats
