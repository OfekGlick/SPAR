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
"""Implementation of MultiHeadDiscreteActor."""

import torch
import torch.nn as nn
from torch.distributions import Categorical, Bernoulli
from omnisafe.models.base import Actor
from omnisafe.utils.model import build_mlp_network


class MultiHeadDiscreteActor(Actor):
    """MultiHeadDiscreteActor with shared layers for discrete environment actions and binary sensor mask.

    This actor is designed for environments with:
    - Discrete environment actions (e.g., DiscreteMetaAction with 5 actions)
    - Binary sensor mask (MultiBinary for selecting which sensors to use)

    Args:
        obs_space: Observation space.
        disc_env_act_space: Discrete action space for environment actions (e.g., Discrete(5)).
        disc_mask_space: MultiBinary action space for sensor mask.
        hidden_sizes: List of hidden layer sizes for separate heads.
        shared_hidden_sizes: List of hidden layer sizes for shared layers.
        activation: Activation function name.
        weight_initialization_mode: Weight initialization mode.
    """

    def __init__(
            self,
            obs_space,
            disc_env_act_space,
            disc_mask_space,
            hidden_sizes,
            shared_hidden_sizes,
            activation='relu',
            weight_initialization_mode='kaiming_uniform',
    ):
        """Initialize the MultiHeadDiscreteActor with shared layers.

        """
        # Pass disc_env_act_space as act_space to base class
        super().__init__(obs_space, disc_env_act_space, hidden_sizes, activation, weight_initialization_mode)

        # Shared layers
        self.shared_net = build_mlp_network(
            sizes=[self._obs_dim, *shared_hidden_sizes],
            activation=activation,
            weight_initialization_mode=weight_initialization_mode,
            norm=True,
        )

        # Discrete environment action head (Categorical distribution)
        # disc_env_act_space.n gives number of discrete actions
        self.num_env_actions = disc_env_act_space.n
        self.discrete_env_net = build_mlp_network(
            sizes=[shared_hidden_sizes[-1], *hidden_sizes, self.num_env_actions],
            activation=activation,
            weight_initialization_mode=weight_initialization_mode,
            output_activation='identity',  # Raw logits for Categorical
            norm=True,
        )

        # Binary sensor mask head (Bernoulli distribution)
        self.num_sensors = disc_mask_space.shape[0]
        self.sensor_mask_net = build_mlp_network(
            sizes=[shared_hidden_sizes[-1], *hidden_sizes, self.num_sensors],
            activation=activation,
            weight_initialization_mode=weight_initialization_mode,
            output_activation='identity',  # Raw logits for Bernoulli
            norm=True,
        )

        self.disc_env_act_space = disc_env_act_space
        self.disc_mask_space = disc_mask_space
        self._current_env_dist = None
        self._current_mask_dist = None
        self._temperature = 1.0

    def _distribution(self, obs):
        """Generate and return distributions for discrete env action and sensor mask.

        Args:
            obs: Observation tensor.

        Returns:
            Tuple of (Categorical distribution for env action, Bernoulli distribution for mask).
        """
        shared_features = self.shared_net(obs)

        # Discrete environment action distribution (Categorical)
        env_logits = self.discrete_env_net(shared_features) / self._temperature
        env_dist = Categorical(logits=env_logits)

        # Sensor mask distribution (Bernoulli)
        mask_logits = self.sensor_mask_net(shared_features)
        mask_dist = Bernoulli(logits=mask_logits)

        return env_dist, mask_dist

    def forward(self, obs):
        """Generate and cache action distributions.

        Args:
            obs: Observation tensor.

        Returns:
            Tuple of (Categorical distribution, Bernoulli distribution).
        """
        self._current_env_dist, self._current_mask_dist = self._distribution(obs)
        self._after_inference = True
        return self._current_env_dist, self._current_mask_dist

    def predict(self, obs, deterministic=True):
        """Predict both discrete environment action and sensor mask.

        Args:
            obs: Observation tensor.
            deterministic: If True, use mode/argmax; if False, sample from distributions.

        Returns:
            Tuple of (env_action, sensor_mask) tensors.
        """
        self._current_env_dist, self._current_mask_dist = self._distribution(obs)

        # Predict discrete environment action
        if deterministic:
            env_action = self._current_env_dist.probs.argmax(dim=-1)
        else:
            env_action = self._current_env_dist.sample()

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

        # Split the action into environment action and sensor mask
        if isinstance(act, tuple):
            env_action = act[0]
            sensor_mask = act[1].squeeze(-1) if act[1].dim() > 1 else act[1]
        else:
            # Assume first element is env action (integer), rest is sensor mask
            env_action = act[:, 0].long()
            sensor_mask = act[:, 1:].squeeze(-1)

        # Compute log probabilities
        env_log_prob = self._current_env_dist.log_prob(env_action)
        mask_log_prob = self._current_mask_dist.log_prob(sensor_mask).sum(dim=-1)

        # Return as tuple for decoupled PPO (each head gets independent clipping)
        return (env_log_prob, mask_log_prob)

    @property
    def temperature(self) -> float:
        """Get the temperature parameter for Categorical softmax."""
        return self._temperature

    @temperature.setter
    def temperature(self, temp: float) -> None:
        """Set the temperature parameter."""
        assert temp > 0, "Temperature must be positive"
        self._temperature = temp