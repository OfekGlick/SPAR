# Copyright 2023 OmniSafe Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Implementation of Evaluator."""

from __future__ import annotations

import json
import os
import warnings
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import gymnasium
from gymnasium.spaces import Box
from gymnasium.utils.save_video import save_video
import cv2
import wandb

from omnisafe.common import Normalizer
from omnisafe.envs.core import CMDP, make
from omnisafe.envs.wrapper import ActionRepeat, ActionScale, ObsNormalize, TimeLimit, ModalityObsNormalize, ModalityObsScale
from omnisafe.models.actor import ActorBuilder
from omnisafe.models.actor_critic import ConstraintActorCritic, ConstraintActorQCritic
from omnisafe.models.base import Actor
from omnisafe.utils.config import Config


class Evaluator:  # pylint: disable=too-many-instance-attributes
    """This class includes common evaluation methods for safe RL algorithms.

    Args:
        env (CMDP or None, optional): The environment. Defaults to None.
        actor (Actor or None, optional): The actor. Defaults to None.
        render_mode (str, optional): The render mode. Defaults to 'rgb_array'.
    """

    _cfgs: Config
    _dict_cfgs: dict[str, Any]
    _save_dir: str
    _model_name: str
    _cost_count: torch.Tensor

    # pylint: disable-next=too-many-arguments
    def __init__(
            self,
            env: CMDP | None = None,
            actor: Actor | None = None,
            actor_critic: ConstraintActorCritic | ConstraintActorQCritic | None = None,
            render_mode: str = 'rgb_array',
            use_wandb: bool = False,
    ) -> None:
        """Initialize an instance of :class:`Evaluator`."""
        self._env: CMDP | None = env
        self._actor: Actor | None = actor
        self._actor_critic: ConstraintActorCritic | ConstraintActorQCritic | None = actor_critic
        self._dividing_line: str = '\n' + '#' * 50 + '\n'
        self._use_wandb: bool = use_wandb

        self._safety_budget: torch.Tensor
        self._safety_obs = torch.ones(1)
        self._cost_count = torch.zeros(1)
        self.__set_render_mode(render_mode)

    def __set_render_mode(self, render_mode: str) -> None:
        """Set the render mode.

        Args:
            render_mode (str, optional): The render mode. Defaults to 'rgb_array'.

        Raises:
            NotImplementedError: If the render mode is not implemented.
        """
        # set the render mode
        if render_mode in ['human', 'rgb_array', 'rgb_array_list']:
            self._render_mode: str = render_mode
        else:
            raise NotImplementedError('The render mode is not implemented.')

    def __load_cfgs(self, save_dir: str) -> None:
        """Load the config from the save directory.

        Args:
            save_dir (str): Directory where the model is saved.

        Raises:
            FileNotFoundError: If the config file is not found.
        """
        cfg_path = os.path.join(save_dir, 'config.json')
        try:
            with open(cfg_path, encoding='utf-8') as file:
                kwargs = json.load(file)
        except FileNotFoundError as error:
            raise FileNotFoundError(
                f'The config file is not found in the save directory{save_dir}.',
            ) from error
        self._dict_cfgs = kwargs
        self._cfgs = Config.dict2config(kwargs)

    def _is_continuous_action_space(self, action_space: gymnasium.spaces.Space) -> bool:
        """Check if action space contains continuous actions.

        Args:
            action_space: The action space to check.

        Returns:
            True if the action space is Box or contains Box (for Tuple spaces).
        """
        if isinstance(action_space, gymnasium.spaces.Box):
            return True
        elif isinstance(action_space, gymnasium.spaces.Tuple):
            # Check if first element is Box (continuous)
            return isinstance(action_space[0], gymnasium.spaces.Box)
        return False

    # pylint: disable-next=too-many-branches
    def __load_model_and_env(
            self,
            save_dir: str,
            model_name: str,
            env_kwargs: dict[str, Any],
    ) -> None:
        """Load the model from the save directory.

        Args:
            save_dir (str): Directory where the model is saved.
            model_name (str): Name of the model.
            env_kwargs (dict[str, Any]): Keyword arguments for the environment.

        Raises:
            FileNotFoundError: If the model is not found.
        """
        # load the saved model
        model_path = os.path.join(save_dir, 'torch_save', model_name)
        try:
            model_params = torch.load(model_path)
        except FileNotFoundError as error:
            raise FileNotFoundError('The model is not found in the save directory.') from error

        # load the environment
        self._env = make(**env_kwargs)
        max_episode_steps = self._env.max_episode_steps

        observation_space = self._env.observation_space
        action_space = self._env.action_space
        if 'Saute' in self._cfgs['algo'] or 'Simmer' in self._cfgs['algo']:
            self._safety_budget = (
                    self._cfgs.algo_cfgs.safety_budget
                    * (1 - self._cfgs.algo_cfgs.saute_gamma ** self._cfgs.algo_cfgs.max_ep_len)
                    / (1 - self._cfgs.algo_cfgs.saute_gamma)
                    / self._cfgs.algo_cfgs.max_ep_len
                    * torch.ones(1)
            )
        # assert isinstance(observation_space, Box), 'The observation space must be Box.'
        # assert isinstance(action_space, Box), 'The action space must be Box.'
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        if self._cfgs['algo_cfgs']['obs_normalize']:
            obs_normalizer = Normalizer(shape=observation_space.shape, clip=5)
            obs_normalizer.load_state_dict(model_params['obs_normalizer'])
            obs_normalizer.eval()  # Set to evaluation mode to prevent stats updates
            self._env = ObsNormalize(self._env, device=torch.device(device), norm=obs_normalizer)

        if self._cfgs['algo_cfgs']['obs_modality_normalize']:
            spans = self._env.obs_mapping  # {'Kinematics': (0,6), 'Lidar': (6,86), ...}
            mask_len = len(spans)

            # First normalize each modality to bring them to the same scale
            per_mod_norm = nn.ModuleDict()
            for mod, (s, e) in spans.items():
                seg_len = int(e - s)
                per_mod_norm[mod] = Normalizer(shape=(seg_len,), clip=5)
            per_mod_norm.load_state_dict(model_params['obs_normalizer'])

            self._env = ModalityObsNormalize(
                self._env,
                device=torch.device(device),
                modality_to_span=spans,
                mask_length=mask_len,
                norm_per_mod_state=per_mod_norm,
            )
            self._env.freeze_stats(True)

            # Then apply scaling based on active modality count
            self._env = ModalityObsScale(
                self._env,
                device=torch.device(device),
            )


        if self._env.need_time_limit_wrapper:
            self._env = TimeLimit(self._env, device=torch.device(device), time_limit=max_episode_steps)

        if self._is_continuous_action_space(self._env.action_space):
            self._env = ActionScale(self._env, device=torch.device(device), low=-1.0, high=1.0)

        if hasattr(self._cfgs['algo_cfgs'], 'action_repeat'):
            self._env = ActionRepeat(
                self._env,
                device=torch.device(device),
                times=self._cfgs['algo_cfgs']['action_repeat'],
            )
        if hasattr(self._cfgs, 'algo') and self._cfgs['algo'] in [
            'LOOP',
            'SafeLOOP',
            'PETS',
            'CAPPETS',
            'RCEPETS',
            'CCEPETS',
        ]:
            raise ValueError(
                f"Model-based algorithm '{self._cfgs['algo']}' is no longer supported. "
                f"This codebase only supports on-policy and off-policy algorithms."
            )
        else:
            if 'Saute' in self._cfgs['algo'] or 'Simmer' in self._cfgs['algo']:
                observation_space = Box(
                    low=np.hstack((observation_space.low, -np.inf)),
                    high=np.hstack((observation_space.high, np.inf)),
                    shape=(observation_space.shape[0] + 1,),
                )
            actor_type = self._cfgs['model_cfgs']['actor_type']
            pi_cfg = self._cfgs['model_cfgs']['actor']
            weight_initialization_mode = self._cfgs['model_cfgs']['weight_initialization_mode']

            # Check if action space is Tuple (multihead actor case)
            if isinstance(action_space, gymnasium.spaces.Tuple):
                # Multihead actor requires env_act_space and mask_act_space
                actor_builder = ActorBuilder(
                    obs_space=observation_space,
                    act_space=action_space,
                    hidden_sizes=pi_cfg['hidden_sizes'],
                    activation=pi_cfg['activation'],
                    weight_initialization_mode=weight_initialization_mode,
                    env_act_space=action_space[0],
                    mask_act_space=action_space[1],
                )
            else:
                # Standard actor (Box or Discrete action space)
                actor_builder = ActorBuilder(
                    obs_space=observation_space,
                    act_space=action_space,
                    hidden_sizes=pi_cfg['hidden_sizes'],
                    activation=pi_cfg['activation'],
                    weight_initialization_mode=weight_initialization_mode,
                )
            self._actor = actor_builder.build_actor(actor_type)
            self._actor.load_state_dict(model_params['pi'])
            # Move actor to the same device as the environment uses
            self._actor.to(device)

    # pylint: disable-next=too-many-locals
    def load_saved(
            self,
            save_dir: str,
            model_name: str,
            render_mode: str = 'rgb_array',
            camera_name: str | None = None,
            camera_id: int | None = None,
            width: int = 256,
            height: int = 256,
    ) -> None:
        """Load a saved model.

        Args:
            save_dir (str): The directory where the model is saved.
            model_name (str): The name of the model.
            render_mode (str, optional): The render mode, ranging from 'human', 'rgb_array',
                'rgb_array_list'. Defaults to 'rgb_array'.
            camera_name (str or None, optional): The name of the camera. Defaults to None.
            camera_id (int or None, optional): The id of the camera. Defaults to None.
            width (int, optional): The width of the image. Defaults to 256.
            height (int, optional): The height of the image. Defaults to 256.
        """
        # load the config
        self._save_dir = save_dir
        self._model_name = model_name

        self.__load_cfgs(save_dir)

        self.__set_render_mode(render_mode)

        env_kwargs = {
            'env_id': self._cfgs['env_id'],
            'num_envs': 1,
            'render_mode': self._render_mode,
            # 'camera_id': camera_id,
            # 'camera_name': camera_name,
            # 'width': width,
            # 'height': height,
        }
        if self._dict_cfgs.get('env_cfgs') is not None:
            env_kwargs.update(self._dict_cfgs['env_cfgs'])

        # Force rendering capability for robosuite environments during evaluation
        # This enables video recording without requiring it during training
        env_id = self._cfgs['env_id']
        if 'robosuite' in env_id.lower() or 'door' in env_id.lower() or 'lift' in env_id.lower():
            env_kwargs['has_offscreen_renderer'] = True
            env_kwargs['has_renderer'] = False

        self.__load_model_and_env(save_dir, model_name, env_kwargs)

    def evaluate(
            self,
            num_episodes: int = 10,
            cost_criteria: float = 1.0,
            record_video: bool = False,
            video_top_k: int = 3,
            video_metric: str = 'reward',
            sample_efc = False,
    ) -> tuple[list[float], list[float]]:
        """Evaluate the agent for num_episodes episodes.

        Args:
            num_episodes (int, optional): The number of episodes to evaluate. Defaults to 10.
            cost_criteria (float, optional): The cost criteria. Defaults to 1.0.
            record_video (bool, optional): Whether to record video with sensor overlays. Defaults to False.
            video_top_k (int, optional): If recording, save only top-K episodes. Defaults to 10.
            video_metric (str, optional): Selection metric for top-K videos: 'reward' or 'length'. Defaults to 'reward'.

        Returns:
            (episode_rewards, episode_costs): The episode rewards and costs.

        Raises:
            ValueError: If the environment and the policy are not provided or created.
        """
        if self._env is None or self._actor is None:
            raise ValueError(
                'The environment and the actor must be provided or created before evaluating the agent.',
            )

        episode_rewards: list[float] = []
        episode_costs: list[float] = []
        episode_obs_masks = []
        episode_lengths: list[float] = []
        all_episode_frames = [] if record_video else None
        all_episode_masks = [] if record_video else None

        for episode in range(num_episodes):
            current_episode_masks = []
            current_episode_frames = [] if record_video else None
            obs, _ = self._env.reset()
            # Ensure _safety_obs is on the same device as observations
            # Handle both tensor and dict observations
            if isinstance(obs, torch.Tensor):
                device = obs.device
            elif isinstance(obs, dict):
                # Get device from any tensor value in dict, default to cpu
                device = next((v.device for v in obs.values() if isinstance(v, torch.Tensor)), torch.device('cpu'))
            else:
                device = torch.device('cpu')
            self._safety_obs = torch.ones(1, device=device)
            ep_ret, ep_cost, length = 0.0, 0.0, 0.0

            done = False
            while not done:
                if 'Saute' in self._cfgs['algo'] or 'Simmer' in self._cfgs['algo']:
                    # Ensure _safety_obs is on the same device before concatenation
                    self._safety_obs = self._safety_obs.to(obs.device)
                    obs = torch.cat([obs, self._safety_obs], dim=-1)
                with torch.no_grad():
                    act = self._actor.predict(obs, deterministic=True)
                obs, rew, cost, terminated, truncated, info = self._env.step(act)
                try:
                    current_episode_masks.append(info['sensor_mask'])
                except KeyError:
                    pass

                # Collect frames if recording video
                if record_video:
                    frame = self._env.render()
                    if frame is not None:
                        current_episode_frames.append(frame)

                if 'Saute' in self._cfgs['algo'] or 'Simmer' in self._cfgs['algo']:
                    self._safety_obs -= cost.unsqueeze(-1) / self._safety_budget
                    self._safety_obs /= self._cfgs.algo_cfgs.saute_gamma

                ep_ret += rew.item()
                ep_cost += (cost_criteria ** length) * cost.item()
                if (
                        'EarlyTerminated' in self._cfgs['algo']
                        and ep_cost >= self._cfgs.algo_cfgs.cost_limit
                ):
                    terminated = torch.as_tensor(True)
                length += 1

                done = bool(terminated or truncated)
            episode_masks_ratio = np.sum(current_episode_masks, axis=0) / length
            episode_obs_masks.append(episode_masks_ratio)
            episode_rewards.append(ep_ret)
            episode_costs.append(ep_cost)
            episode_lengths.append(length)

            # Store frames and masks for video generation
            if record_video and current_episode_frames:
                all_episode_frames.append(current_episode_frames)
                all_episode_masks.append(current_episode_masks)
            if not sample_efc and self._use_wandb:
                wandb.log({"Evaluation/EpReward": ep_ret, "Evaluation/EpCost": ep_cost})
            print(f'Episode {episode + 1} results:')
            print(f'Episode reward: {ep_ret}')
            print(f'Episode cost: {ep_cost}')
            print(f'Episode action masking: {episode_masks_ratio}')
            print(f'Episode length: {length}')

        print(self._dividing_line)
        print('Evaluation results:')
        print(f'Average episode reward: {np.mean(a=episode_rewards)}')
        print(f'Average episode cost: {np.mean(a=episode_costs)}')
        print(f'Cost of each Sensor: {[f"{cost:.3f}" for cost in self._env.costs]}')
        print(f'Average action masking: {np.mean(a=episode_obs_masks, axis=0)}')
        print(f'STD action masking: {np.std(a=episode_obs_masks, axis=0)}')
        print(f'Average episode length: {np.mean(a=episode_lengths)}')
        print(f'Sensors used: {self._env.obs_names}')
        print(self._dividing_line)
        results = {
            "episode_rewards": episode_rewards,
            "episode_costs": episode_costs,
            "episode_lengths": episode_lengths,
            "episode_obs_masks": episode_obs_masks,
            "sensor_costs": self._env.costs,
            "sensor_names": self._env.obs_names,
        }

        # Generate videos with sensor overlays if requested
        if record_video and all_episode_frames:
            # Select top-K episodes by the requested metric
            metric_values = episode_rewards if video_metric.lower() == 'reward' else episode_lengths
            # Determine number to keep
            k = max(0, min(video_top_k, len(metric_values)))
            if k > 0:
                # argsort returns ascending; reverse for descending (best first)
                top_indices = list(np.argsort(metric_values)[::-1][:k])

                selected_frames = [all_episode_frames[i] for i in top_indices]
                selected_masks = [all_episode_masks[i] for i in top_indices]
                selected_rewards = [episode_rewards[i] for i in top_indices]
                selected_costs = [episode_costs[i] for i in top_indices]

                self._generate_sensor_videos(
                    selected_frames,
                    selected_masks,
                    self._env.obs_names,
                    selected_rewards,
                    selected_costs,
                    episode_indices=top_indices,
                )

        self._env.close()
        return results

    def _generate_sensor_videos(
            self,
            all_episode_frames: list[list[np.ndarray]],
            all_episode_masks: list[list[np.ndarray]],
            sensor_names: list[str],
            episode_rewards: list[float],
            episode_costs: list[float],
            episode_indices: list[int] | None = None,
    ) -> None:
        """Generate videos with sensor overlay annotations.

        Args:
            all_episode_frames: List of frame lists for each episode
            all_episode_masks: List of mask lists for each episode
            sensor_names: Names of sensors/modalities
            episode_rewards: Rewards for each episode
            episode_costs: Costs for each episode
        """
        video_dir = os.path.join(self._save_dir, 'evaluation_videos')
        os.makedirs(video_dir, exist_ok=True)

        print(f'\nGenerating evaluation videos with sensor overlays...')
        print(f'Saving to: {video_dir}')

        for ep_idx, (frames, masks) in enumerate(zip(all_episode_frames, all_episode_masks)):
            if not frames:
                continue

            # Create annotated frames
            annotated_frames = []
            cumulative_cost = 0.0

            for frame_idx, frame in enumerate(frames):
                # Convert to BGR for OpenCV if needed
                if len(frame.shape) == 3 and frame.shape[2] == 3:
                    annotated_frame = frame.copy()
                else:
                    annotated_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                # Get current sensor mask (handle case where masks might be shorter than frames)
                if frame_idx < len(masks):
                    current_mask = masks[frame_idx]
                    cumulative_cost += np.sum(current_mask)
                else:
                    current_mask = np.zeros(len(sensor_names))

                # Add black semi-transparent overlay area at top
                h, w = annotated_frame.shape[:2]
                overlay_height = 30 + (len(sensor_names) * 25)
                overlay = annotated_frame.copy()
                cv2.rectangle(overlay, (0, 0), (w, overlay_height), (0, 0, 0), -1)
                annotated_frame = cv2.addWeighted(annotated_frame, 0.6, overlay, 0.4, 0)

                # Add episode info
                # Use original episode index if provided for labeling
                orig_ep = (episode_indices[ep_idx] + 1) if episode_indices is not None else (ep_idx + 1)
                cv2.putText(
                    annotated_frame,
                    f'Episode {orig_ep} | Step {frame_idx + 1}/{len(frames)} | Cost: {cumulative_cost:.1f}',
                    (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

                # Add sensor status for each modality
                y_offset = 45
                for i, sensor_name in enumerate(sensor_names):
                    is_active = current_mask[i] > 0.5 if i < len(current_mask) else False
                    color = (0, 255, 0) if is_active else (0, 0, 255)  # Green if active, red if inactive
                    status = "ON " if is_active else "OFF"

                    cv2.putText(
                        annotated_frame,
                        f'{sensor_name}: {status}',
                        (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        color,
                        1,
                        cv2.LINE_AA,
                    )
                    y_offset += 25

                annotated_frames.append(annotated_frame)

            # Save video using gymnasium's save_video
            save_video(
                annotated_frames,
                video_dir,
                fps=self.fps,
                episode_trigger=lambda x: True,
                episode_index=ep_idx,
                name_prefix=f'eval_ep{(episode_indices[ep_idx] + 1) if episode_indices is not None else (ep_idx + 1)}_reward{episode_rewards[ep_idx]:.1f}_cost{episode_costs[ep_idx]:.1f}',
            )

            # Save sensor mask CSV for this episode
            self._save_mask_csv(
                masks=masks,
                sensor_names=sensor_names,
                video_dir=video_dir,
                episode_index=(episode_indices[ep_idx] + 1) if episode_indices is not None else (ep_idx + 1),
                episode_reward=episode_rewards[ep_idx],
                episode_cost=episode_costs[ep_idx],
            )

            # Log to wandb
            video_path = os.path.join(
                video_dir,
                f'eval_ep{(episode_indices[ep_idx] + 1) if episode_indices is not None else (ep_idx + 1)}_reward{episode_rewards[ep_idx]:.1f}_cost{episode_costs[ep_idx]:.1f}-episode-{ep_idx}.mp4',
            )
            if os.path.exists(video_path) and self._use_wandb:
                wandb.log({
                    f"Evaluation/Video_Episode_{ep_idx + 1}": wandb.Video(video_path, fps=self.fps, format="mp4")
                })

        print(f'Generated {len(all_episode_frames)} evaluation videos')

    def _save_mask_csv(
            self,
            masks: list[np.ndarray],
            sensor_names: list[str],
            video_dir: str,
            episode_index: int,
            episode_reward: float,
            episode_cost: float,
    ) -> None:
        """Save a CSV file documenting the sensor masks for each step of an episode.

        Args:
            masks: List of mask arrays for each step
            sensor_names: Names of sensors/modalities
            video_dir: Directory to save the CSV
            episode_index: Episode number (1-indexed)
            episode_reward: Total episode reward
            episode_cost: Total episode cost
        """
        import csv

        # Create CSV filename matching the video filename
        csv_filename = f'eval_ep{episode_index}_reward{episode_reward:.1f}_cost{episode_cost:.1f}_masks.csv'
        csv_path = os.path.join(video_dir, csv_filename)

        # Get sensor costs from environment (if available)
        try:
            sensor_costs = self._env.costs
        except AttributeError:
            sensor_costs = [1.0] * len(sensor_names)  # Default cost of 1.0 per sensor

        # Prepare CSV data
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            # Create header: step, sensor columns, step_cost, cumulative_cost
            fieldnames = ['step'] + sensor_names + ['step_cost', 'cumulative_cost']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            cumulative_cost = 0.0
            for step_idx, mask in enumerate(masks):
                # Calculate step cost (sum of costs for active sensors)
                step_cost = sum(sensor_costs[i] * mask[i] for i in range(len(mask)))
                cumulative_cost += step_cost

                # Create row data
                row = {'step': step_idx + 1}  # 1-indexed
                for i, sensor_name in enumerate(sensor_names):
                    row[sensor_name] = int(mask[i]) if i < len(mask) else 0
                row['step_cost'] = f'{step_cost:.4f}'
                row['cumulative_cost'] = f'{cumulative_cost:.4f}'

                writer.writerow(row)

        print(f'  Saved mask CSV: {csv_filename}')

    @property
    def fps(self) -> int:
        """The fps of the environment.

        Raises:
            AssertionError: If the environment is not provided or created.
            AtrributeError: If the fps is not found.
        """
        assert (
                self._env is not None
        ), 'The environment must be provided or created before getting the fps.'
        try:
            fps = self._env.metadata['render_fps']
        except (AttributeError, KeyError):
            fps = 30
            warnings.warn('The fps is not found, use 30 as default.', stacklevel=2)

        return fps

    def render(  # pylint: disable=too-many-locals,too-many-arguments,too-many-branches,too-many-statements
            self,
            num_episodes: int = 10,
            save_replay_path: str | None = None,
            max_render_steps: int = 2000,
            cost_criteria: float = 1.0,
    ) -> None:  # pragma: no cover
        """Render the environment for one episode.

        Args:
            num_episodes (int, optional): The number of episodes to render. Defaults to 1.
            save_replay_path (str or None, optional): The path to save the replay video. Defaults to
                None.
            max_render_steps (int, optional): The maximum number of steps to render. Defaults to 2000.
            cost_criteria (float, optional): The discount factor for the cost. Defaults to 1.0.
        """
        assert (
                self._env is not None
        ), 'The environment must be provided or created before rendering.'
        assert (
                self._actor is not None
        ), 'The actor must be provided or created before rendering.'
        if save_replay_path is None:
            save_replay_path = os.path.join(self._save_dir, 'results', self._model_name.split('.')[0])
        os.makedirs(save_replay_path, exist_ok=True)
        result_path = os.path.join(save_replay_path, 'result.txt')
        print(self._dividing_line)
        print(f'Saving the results to {result_path}.')
        print(self._dividing_line)

        episode_rewards: list[float] = []
        episode_costs: list[float] = []
        episode_lengths: list[float] = []

        for episode_idx in range(num_episodes):
            obs, _ = self._env.reset()
            # Handle both tensor and dict observations
            if isinstance(obs, torch.Tensor):
                device = obs.device
            elif isinstance(obs, dict):
                device = next((v.device for v in obs.values() if isinstance(v, torch.Tensor)), torch.device('cpu'))
            else:
                device = torch.device('cpu')
            self._safety_obs = torch.ones(1, device=device)
            step = 0
            done = False
            ep_ret, ep_cost, length = 0.0, 0.0, 0.0
            while (
                    not done and step <= max_render_steps
            ):  # a big number to make sure the episode will end
                if 'Saute' in self._cfgs['algo'] or 'Simmer' in self._cfgs['algo']:
                    self._safety_obs = self._safety_obs.to(obs.device)
                    obs = torch.cat([obs, self._safety_obs], dim=-1)
                with torch.no_grad():
                    act = self._actor.predict(obs, deterministic=True)
                obs, rew, cost, terminated, truncated, info = self._env.step(act)
                if 'Saute' in self._cfgs['algo'] or 'Simmer' in self._cfgs['algo']:
                    self._safety_obs -= cost.unsqueeze(-1) / self._safety_budget
                    self._safety_obs /= self._cfgs.algo_cfgs.saute_gamma
                step += 1
                done = bool(terminated or truncated)
                ep_ret += rew.item()
                ep_cost += (cost_criteria ** length) * cost.item()
                if (
                        'EarlyTerminated' in self._cfgs['algo']
                        and ep_cost >= self._cfgs.algo_cfgs.cost_limit
                ):
                    terminated = torch.as_tensor(True)
                length += 1

            episode_rewards.append(ep_ret)
            episode_costs.append(ep_cost)
            episode_lengths.append(length)
            with open(result_path, 'a+', encoding='utf-8') as f:
                print(f'Episode {episode_idx + 1} results:', file=f)
                print(f'Episode reward: {ep_ret}', file=f)
                print(f'Episode cost: {ep_cost}', file=f)
                print(f'Episode length: {length}', file=f)

        with open(result_path, 'a+', encoding='utf-8') as f:
            print(self._dividing_line, file=f)
            print('Evaluation results:', file=f)
            print(f'Average episode reward: {np.mean(episode_rewards)}', file=f)
            print(f'Average episode cost: {np.mean(episode_costs)}', file=f)
            print(f'Average episode length: {np.mean(episode_lengths)}', file=f)

        # No need to close env here - it's managed by the evaluator
