"""Random Mask Actor with Discrete Environment Actions.

This actor learns discrete environment actions (Categorical) but outputs
random modality masks, serving as a baseline to evaluate whether learning
modality selection provides value over random selection.
"""

import torch
from torch.distributions import Categorical, Bernoulli
from omnisafe.models.base import Actor
from omnisafe.utils.model import build_mlp_network


class RandomMaskActorDiscrete(Actor):
    """Actor with learned discrete environment actions and random modality masks.

    Architecture:
    - Environment action head: Learned Categorical distribution (discrete actions)
    - Mask head: Random Bernoulli sampling (no learning, zero gradient)

    This serves as a baseline to evaluate if learning modality selection
    provides value over random selection for discrete action tasks.
    """

    def __init__(
        self,
        obs_space,
        disc_act_space,
        disc_mask_space,
        hidden_sizes,
        shared_hidden_sizes,
        activation='relu',
        weight_initialization_mode='kaiming_uniform',
    ):
        """Initialize the RandomMaskActorDiscrete.

        Args:
            obs_space: Observation space
            disc_act_space: Discrete action space (environment actions)
            disc_mask_space: Discrete mask space (modality mask - not learned)
            hidden_sizes: Hidden layer sizes for action heads
            shared_hidden_sizes: Shared layer sizes
            activation: Activation function
            weight_initialization_mode: Weight initialization mode
        """
        super().__init__(obs_space, disc_act_space, hidden_sizes, activation, weight_initialization_mode)

        # Extract number of discrete actions and modalities
        if hasattr(disc_act_space, 'n'):
            self.num_actions = disc_act_space.n
        else:
            raise ValueError("Environment action space must be Discrete with attribute 'n'")

        self.num_modalities = disc_mask_space.shape[0]

        # Shared layers
        self.shared_net = build_mlp_network(
            sizes=[self._obs_dim, *shared_hidden_sizes],
            activation=activation,
            weight_initialization_mode=weight_initialization_mode,
            norm=True,
        )

        # Discrete environment action head (LEARNED)
        self.discrete_net = build_mlp_network(
            sizes=[shared_hidden_sizes[-1], *hidden_sizes, self.num_actions],
            activation=activation,
            weight_initialization_mode=weight_initialization_mode,
            output_activation='identity',  # Raw logits for Categorical
            norm=True,
        )

        # Temperature parameter for controlling exploration (higher = more exploration)
        self._temperature = 1.0

        # NO mask network - we just sample randomly
        # No parameters to learn for the mask head

        self.disc_action_space = disc_act_space
        self.disc_mask_space = disc_mask_space
        self._current_action_dist = None
        self._current_mask_dist = None

    def _distribution(self, obs):
        """Generate distribution for discrete environment actions only.

        Random masks are not part of the learned distribution - they're generated
        only during predict() for rollout purposes.

        Args:
            obs: Observation tensor (can be batched [B, obs_dim] or unbatched [obs_dim])

        Returns:
            action_dist: Learned Categorical distribution over environment actions
        """
        shared_features = self.shared_net(obs)

        # Discrete action distribution (LEARNED)
        logits = self.discrete_net(shared_features) / self._temperature
        action_dist = Categorical(logits=logits)

        return action_dist

    def forward(self, obs):
        """Generate and cache action distribution.

        Args:
            obs: Observation tensor

        Returns:
            tuple: (action_dist, None) - consistent with multi-head actors but mask is None
        """
        self._current_action_dist = self._distribution(obs)
        self._current_mask_dist = None  # Not used for training
        self._after_inference = True
        return (self._current_action_dist, None)  # Return tuple for consistency, mask is None

    def predict(self, obs, deterministic=False):
        """Predict environment actions and random masks.

        Args:
            obs: Observation tensor
            deterministic: If True, use argmax for discrete action

        Returns:
            disc_action: Sampled/argmax environment action
            binary_mask: Random binary mask (pure Bernoulli with p=0.5)
        """
        self._current_action_dist = self._distribution(obs)

        # Predict discrete action (LEARNED)
        if deterministic:
            disc_action = self._current_action_dist.probs.argmax(dim=-1)
        else:
            disc_action = self._current_action_dist.sample()

        # Generate random mask directly (NOT LEARNED - no distribution stored)
        # Always random, even if deterministic=True
        batch_shape = disc_action.shape
        binary_mask = torch.bernoulli(
            torch.full((*batch_shape, self.num_modalities), 0.5, device=obs.device)
        )

        self._after_inference = True
        return disc_action, binary_mask

    def log_prob(self, act):
        """Calculate log probability of environment actions only.

        Mask doesn't contribute to training (it's random, not learned).

        Args:
            act: Action tuple (discrete_action, discrete_mask) or concatenated tensor

        Returns:
            torch.Tensor: Log probability of environment action only (scalar)
        """
        assert self._current_action_dist is not None, \
            "Distribution not found. Call forward() or predict() before log_prob()."

        # Split the action into discrete action and mask components
        if isinstance(act, tuple):
            disc_action = act[0]
        else:
            # If concatenated, first element is the discrete action, rest is mask
            disc_action = act[:, 0].long()

        # Compute log probability - only environment action contributes
        action_log_prob = self._current_action_dist.log_prob(disc_action)

        return action_log_prob  # Scalar, not tuple

    @property
    def temperature(self):
        """Temperature parameter for softmax (controls exploration)."""
        return self._temperature

    @temperature.setter
    def temperature(self, temp: float):
        """Set temperature parameter."""
        assert temp > 0, "Temperature must be positive"
        self._temperature = temp