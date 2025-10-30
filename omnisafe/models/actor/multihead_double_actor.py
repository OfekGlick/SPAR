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
"""Implementation of MultiHeadDoubleActor with independent networks."""

import torch
import torch.nn as nn
from torch.distributions import Normal, Bernoulli
from omnisafe.models.base import Actor
from omnisafe.utils.model import build_mlp_network


class MultiHeadDoubleActor(Actor):
    """MultiHeadDoubleActor with completely independent networks for continuous env actions and binary sensor mask.

    Unlike MultiHeadActor which uses a shared backbone, this actor uses two completely separate networks:
    - One network for environment actions (continuous, Gaussian distribution)
    - One network for sensor mask (binary, Bernoulli distribution)

    This architecture allows for:
    - Independent learning rates and gradient flow for each action type
    - Different network capacities (hidden sizes) for each head
    - No gradient conflicts between action types
    - Simpler debugging and analysis

    Args:
        obs_space: Observation space.
        cont_act_space: Continuous action space for environment actions (Box).
        disc_act_space: MultiBinary action space for sensor mask.
        env_hidden_sizes: List of hidden layer sizes for environment action network.
        mask_hidden_sizes: List of hidden layer sizes for mask action network.
        activation: Activation function name.
        weight_initialization_mode: Weight initialization mode.
    """

    def __init__(
            self,
            obs_space,
            cont_act_space,
            disc_act_space,
            env_hidden_sizes,
            mask_hidden_sizes,
            activation='relu',
            weight_initialization_mode='kaiming_uniform',
    ):
        """Initialize the MultiHeadDoubleActor with independent networks."""
        # Pass cont_act_space as act_space to base class
        super().__init__(obs_space, cont_act_space, env_hidden_sizes, activation, weight_initialization_mode)

        # Environment action network (continuous, like GaussianLearningActor)
        self.env_net = build_mlp_network(
            sizes=[self._obs_dim, *env_hidden_sizes, cont_act_space.shape[0]],
            activation=activation,
            weight_initialization_mode=weight_initialization_mode,
            output_activation='tanh',
            norm=True,
        )
        self.log_std = nn.Parameter(torch.zeros(cont_act_space.shape[0]), requires_grad=True)

        # Mask action network (independent, outputs logits for Bernoulli)
        self.mask_net = build_mlp_network(
            sizes=[self._obs_dim, *mask_hidden_sizes, disc_act_space.shape[0]],
            activation=activation,
            weight_initialization_mode=weight_initialization_mode,
            output_activation='identity',  # Raw logits for Bernoulli
            norm=True,
        )

        self.cont_action_space = cont_act_space
        self.disc_action_space = disc_act_space
        self._current_env_dist = None
        self._current_mask_dist = None

    def _distribution(self, obs):
        """Generate and return distributions for continuous env action and sensor mask.

        Processes observation through TWO INDEPENDENT networks.

        Args:
            obs: Observation tensor.

        Returns:
            Tuple of (Normal distribution for env action, Bernoulli distribution for mask).
        """
        # Environment action distribution (Gaussian/Normal)
        env_mean = self.env_net(obs)
        std = torch.exp(self.log_std)
        env_dist = Normal(env_mean, std)

        # Sensor mask distribution (Bernoulli)
        mask_logits = self.mask_net(obs)
        mask_dist = Bernoulli(logits=mask_logits)

        return env_dist, mask_dist

    def forward(self, obs):
        """Generate and cache action distributions.

        Args:
            obs: Observation tensor.

        Returns:
            Tuple of (Normal distribution, Bernoulli distribution).
        """
        self._current_env_dist, self._current_mask_dist = self._distribution(obs)
        self._after_inference = True
        return self._current_env_dist, self._current_mask_dist

    def predict(self, obs, deterministic=True):
        """Predict both continuous environment action and sensor mask.

        Args:
            obs: Observation tensor.
            deterministic: If True, use mean for continuous and round for binary; if False, sample.

        Returns:
            Tuple of (env_action, sensor_mask) tensors.
        """
        self._current_env_dist, self._current_mask_dist = self._distribution(obs)

        # Predict continuous environment action
        if deterministic:
            env_action = self._current_env_dist.mean
        else:
            env_action = self._current_env_dist.rsample()

        # Predict sensor mask
        if deterministic:
            sensor_mask = self._current_mask_dist.probs.round()
        else:
            sensor_mask = self._current_mask_dist.sample()

        self._after_inference = True
        return env_action, sensor_mask

    def log_prob(self, act):
        """Calculate log probability of the given actions.

        Returns separate log probabilities for decoupled PPO with independent trust regions.

        Args:
            act: Tuple of (env_action, sensor_mask) or combined tensor.

        Returns:
            tuple: (env_log_prob, mask_log_prob) for decoupled ratio computation.
        """
        assert self._current_env_dist is not None and self._current_mask_dist is not None, \
            "Distributions not found. Call forward() or predict() before log_prob()."

        # Split the action into continuous env action and sensor mask
        if isinstance(act, tuple):
            env_action = act[0]
            mask_action = act[1].squeeze(-1)
        else:
            env_action = act[:, :self.cont_action_space.shape[0]]
            mask_action = act[:, self.cont_action_space.shape[0]:].squeeze(-1)

        # Compute log probabilities
        env_log_prob = self._current_env_dist.log_prob(env_action).sum(dim=-1)
        mask_log_prob = self._current_mask_dist.log_prob(mask_action).sum(dim=-1)

        # Return as tuple for decoupled PPO (each head gets independent clipping)
        return (env_log_prob, mask_log_prob)

    @property
    def std(self) -> float:
        """Get the standard deviation of the normal distribution."""
        return torch.exp(self.log_std).detach().cpu().numpy()

    @std.setter
    def std(self, std: float) -> None:
        """Set the standard deviation of the normal distribution."""
        self.log_std.data.fill_(torch.log(torch.tensor(std, device=self.log_std.device)))