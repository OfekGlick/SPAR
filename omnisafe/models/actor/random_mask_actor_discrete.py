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
        """Generate distributions for discrete actions and random masks.

        Args:
            obs: Observation tensor (can be batched [B, obs_dim] or unbatched [obs_dim])

        Returns:
            action_dist: Learned Categorical distribution over environment actions
            mask_dist: Random Bernoulli(0.5) distribution over modality masks
        """
        shared_features = self.shared_net(obs)

        # Discrete action distribution (LEARNED)
        logits = self.discrete_net(shared_features) / self._temperature
        action_dist = Categorical(logits=logits)

        # Random mask distribution (NOT LEARNED)
        # Bernoulli with p=0.5 for each modality (uses logits=0)
        # Create random logits matching the batch dimension of shared_features
        # If batched: [B, hidden] -> [B, num_modalities]
        # If unbatched: [hidden] -> [num_modalities]
        random_logits = torch.zeros(*shared_features.shape[:-1], self.num_modalities, device=obs.device)
        mask_dist = Bernoulli(logits=random_logits)

        return action_dist, mask_dist

    def forward(self, obs):
        """Generate and cache action distributions.

        Args:
            obs: Observation tensor

        Returns:
            action_dist: Distribution over environment actions
            mask_dist: Distribution over modality masks (random)
        """
        self._current_action_dist, self._current_mask_dist = self._distribution(obs)
        self._after_inference = True
        return self._current_action_dist, self._current_mask_dist

    def predict(self, obs, deterministic=False):
        """Predict environment actions and random masks.

        Args:
            obs: Observation tensor
            deterministic: If True, use argmax for discrete action

        Returns:
            disc_action: Sampled/argmax environment action
            binary_mask: Random binary mask (pure Bernoulli with p=0.5)
        """
        self._current_action_dist, self._current_mask_dist = self._distribution(obs)

        # Predict discrete action (LEARNED)
        if deterministic:
            disc_action = self._current_action_dist.probs.argmax(dim=-1)
        else:
            disc_action = self._current_action_dist.sample()

        # Sample random mask (NOT LEARNED - always random, even if deterministic=True)
        # This is intentional: mask is never learned, so there's no "deterministic" mode
        binary_mask = self._current_mask_dist.sample()

        self._after_inference = True
        return disc_action, binary_mask

    def log_prob(self, act):
        """Calculate log probability of actions.

        For the environment action, we compute the actual log probability.
        For the mask, we return zero since it doesn't contribute to training.

        Args:
            act: Action tuple (discrete_action, discrete_mask)

        Returns:
            torch.Tensor: Total log probability (only env action contributes)
        """
        assert self._current_action_dist is not None and self._current_mask_dist is not None, \
            "Distributions not found. Call forward() or predict() before log_prob()."

        # Split the action into discrete action and mask components
        if isinstance(act, tuple):
            disc_action = act[0]
        else:
            # If concatenated, first element is the discrete action, rest is mask
            disc_action = act[:, 0].long()

        # Compute log probabilities
        # Environment action: actual log prob (LEARNED - affects gradient)
        action_log_prob = self._current_action_dist.log_prob(disc_action)

        return action_log_prob

    @property
    def temperature(self):
        """Temperature parameter for softmax (controls exploration)."""
        return self._temperature

    @temperature.setter
    def temperature(self, temp: float):
        """Set temperature parameter."""
        assert temp > 0, "Temperature must be positive"
        self._temperature = temp