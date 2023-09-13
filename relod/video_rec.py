import os
import imageio
import torch
import torchvision
import wandb


class VideoRecorder(object):
    """Video recording class used for logging videos in MuJuCo environments.

    not tested for UR5 yet.

    """
    def __init__(self, dir_name, height=90, width=160, camera_id=0, fps=30):
        self.dir_name = dir_name
        self.height = height
        self.width = width
        self.camera_id = camera_id
        self.fps = fps
        self.frames = []

    def init(self, enabled=True):
        self.frames = []
        self.enabled = self.dir_name is not None and enabled

    def record(self, env, mode=None):
        if self.enabled:
            frame = env.render(
                mode='rgb_array',
                height=self.height,
                width=self.width,
                camera_id=self.camera_id
            )
            if mode is not None and 'video' in mode:
                _env = env
                while 'video' not in _env.__class__.__name__.lower():
                    _env = _env.env
                frame = _env.apply_to(frame)
            self.frames.append(frame)

    def save(self, file_name):
        if self.enabled:
            path = os.path.join(self.dir_name, file_name)
            imageio.mimsave(path, self.frames, fps=self.fps)


class MaskRecorder(object):
    """Class for the MaDi algorithm to record masks and masked observations."""
    def __init__(self, dir_name, args):
        self.dir_name = dir_name
        self.algorithm = args.algorithm
        self._args = args
        self.grey_transform = torchvision.transforms.ToPILImage(mode='L')
        self.num_frames = 3  # args.frame_stack
        self.num_masks = 3  # args.frame_stack
        self.save_all_frames = False  # set to True to save all (3) frames, instead of just the first one

    def init(self):
        self.obses = []

    def record(self, obs, agent, training_step, test_env, test_mode):
        obs = torch.as_tensor(obs, device=self._args.device).float()
        _test_env_name = f'_test_{test_mode}' if test_env else ''
        self.save_obs_per_frame(obs, training_step, _test_env_name)
        if self.algorithm == 'madi':
            self.save_masked_obs_per_frame(obs, agent, training_step, _test_env_name)
            self.save_mask_per_frame(obs, agent, training_step, _test_env_name)

    def save_obs_per_frame(self, obs, training_step, _test_env_name):
        for frame in range(self.num_frames):
            if frame == 0 or self.save_all_frames:
                torchvision.utils.save_image(
                    obs[0][3 * frame:3 * frame + 3] / 255.,
                    os.path.join(self.dir_name, f'step{training_step}{_test_env_name}_frame{frame}_obs.png'))

    def save_masked_obs_per_frame(self, obs, agent, training_step, _test_env_name):
        masked_obs = agent.performer.apply_mask(obs)
        for frame in range(self.num_frames):
            if frame == 0 or self.save_all_frames:
                torchvision.utils.save_image(
                    masked_obs[0][3 * frame:3 * frame + 3] / 255.,
                    os.path.join(self.dir_name, f'step{training_step}{_test_env_name}_frame{frame}_maskedobs.png'))

    def save_mask_per_frame(self, obs, agent, training_step, _test_env_name):
        frames = obs.chunk(self.num_frames, dim=1)
        for f in range(self.num_frames):
            if f == 0 or self.save_all_frames:
                mask = agent._masker(frames[f])
                mask_image = self.grey_transform(mask.squeeze())
                mask_image.save(os.path.join(self.dir_name, f'step{training_step}{_test_env_name}_frame{f}_mask.png'))
        log_mask_stats(mask, training_step, _test_env_name)


def log_mask_stats(mask, training_step: int, _test_env: str = ''):
    mask_log_data = {
        f'eval_soft_mask/avg{_test_env}': mask.mean(),
        f'eval_soft_mask/std{_test_env}': mask.std(),
        f'eval_soft_mask/min{_test_env}': mask.min(),
        f'eval_soft_mask/max{_test_env}': mask.max(),
        f'eval_soft_mask/median{_test_env}': mask.median(),
    }
    wandb.log(mask_log_data, step=training_step)
