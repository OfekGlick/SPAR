"""Random Mask Actor with Continuous Environment Actions.

This actor learns continuous environment actions (Gaussian) but outputs
random modality masks, serving as a baseline to evaluate whether learning
modality selection provides value over random selection.
"""

import torch
import torch.nn as nn
from torch.distributions import Normal, Bernoulli
from omnisafe.models.base import Actor
from omnisafe.utils.model import build_mlp_network


class RandomMaskActorContinuous(Actor):
    """Actor with learned continuous environment actions and random modality masks.

    Architecture:
    - Environment action head: Learned Gaussian distribution (continuous actions)
    - Mask head: Random Bernoulli sampling (no learning, zero gradient)

    This serves as a baseline to evaluate if learning modality selection
    provides value over random selection for continuous control tasks.
    """

    def __init__(
        self,
        obs_space,
        cont_act_space,
        disc_act_space,
        hidden_sizes,
        shared_hidden_sizes,
        activation='relu',
        weight_initialization_mode='kaiming_uniform',
    ):
        """Initialize the RandomMaskActorContinuous.

        Args:
            obs_space: Observation space
            cont_act_space: Continuous action space (environment actions)
            disc_act_space: Discrete action space (modality mask - not learned)
            hidden_sizes: Hidden layer sizes for action heads
            shared_hidden_sizes: Shared layer sizes
            activation: Activation function
            weight_initialization_mode: Weight initialization mode
        """
        super().__init__(obs_space, cont_act_space, hidden_sizes, activation, weight_initialization_mode)

        self.num_modalities = disc_act_space.shape[0]

        # Shared layers
        self.shared_net = build_mlp_network(
            sizes=[self._obs_dim, *shared_hidden_sizes],
            activation=activation,
            weight_initialization_mode=weight_initialization_mode,
            norm=True,
        )

        # Continuous environment action head (LEARNED)
        self.continuous_net = build_mlp_network(
            sizes=[shared_hidden_sizes[-1], *hidden_sizes, cont_act_space.shape[0]],
            activation=activation,
            weight_initialization_mode=weight_initialization_mode,
            output_activation='tanh',
            norm=True,
        )
        self.log_std = nn.Parameter(torch.zeros(cont_act_space.shape[0]), requires_grad=True)

        # NO discrete network for mask - we just sample randomly
        # No parameters to learn for the mask head

        self.cont_action_space = cont_act_space
        self.disc_action_space = disc_act_space
        self._current_cont_dist = None
        self._current_disc_dist = None

    def _distribution(self, obs):
        """Generate distributions for continuous actions and random masks.

        Args:
            obs: Observation tensor (can be batched [B, obs_dim] or unbatched [obs_dim])

        Returns:
            cont_dist: Learned Gaussian distribution over environment actions
            disc_dist: Random Bernoulli(0.5) distribution over modality masks
        """
        shared_features = self.shared_net(obs)

        # Continuous action distribution (LEARNED)
        cont_mean = self.continuous_net(shared_features)
        std = torch.exp(self.log_std)
        cont_dist = Normal(cont_mean, std)

        # Random mask distribution (NOT LEARNED)
        # Bernoulli with p=0.5 for each modality (uses logits=0)
        # Create random logits matching the batch dimension of shared_features
        # If batched: [B, hidden] -> [B, num_modalities]
        # If unbatched: [hidden] -> [num_modalities]
        random_logits = torch.zeros(*shared_features.shape[:-1], self.num_modalities, device=obs.device)
        disc_dist = Bernoulli(logits=random_logits)

        return cont_dist, disc_dist

    def forward(self, obs):
        """Generate and cache action distributions.

        Args:
            obs: Observation tensor

        Returns:
            cont_dist: Distribution over environment actions
            disc_dist: Distribution over modality masks (random)
        """
        self._current_cont_dist, self._current_disc_dist = self._distribution(obs)
        self._after_inference = True
        return self._current_cont_dist, self._current_disc_dist

    def predict(self, obs, deterministic=False):
        """Predict environment actions and random masks.

        Args:
            obs: Observation tensor
            deterministic: If True, use mean for continuous action

        Returns:
            cont_action: Sampled/mean environment action
            binary_mask: Random binary mask (pure Bernoulli with p=0.5)
        """
        self._current_cont_dist, self._current_disc_dist = self._distribution(obs)

        # Predict continuous action (LEARNED)
        cont_action = self._current_cont_dist.mean if deterministic else self._current_cont_dist.rsample()

        # Sample random mask (NOT LEARNED - always random, even if deterministic=True)
        # This is intentional: mask is never learned, so there's no "deterministic" mode
        binary_mask = self._current_disc_dist.sample()

        self._after_inference = True
        return cont_action, binary_mask

    def log_prob(self, act):
        """Calculate log probability of actions.

        For the environment action, we compute the actual log probability.
        For the mask, we return zero since it doesn't contribute to training.

        Args:
            act: Action tuple (continuous_action, discrete_mask)

        Returns:
            torch.Tensor: Total log probability (only env action contributes)
        """
        assert self._current_cont_dist is not None and self._current_disc_dist is not None, \
            "Distributions not found. Call forward() or predict() before log_prob()."

        # Split the action into continuous and discrete components
        if isinstance(act, tuple):
            cont_action = act[0]
            disc_action = act[1].squeeze(-1)
        else:
            cont_action = act[:, :self.cont_action_space.shape[0]]
            disc_action = act[:, self.cont_action_space.shape[0]:].squeeze(-1)

        # Compute log probabilities
        # Environment action: actual log prob (LEARNED - affects gradient)
        cont_log_prob = self._current_cont_dist.log_prob(cont_action).sum(dim=-1)

        return cont_log_prob

    @property
    def std(self):
        """Get the standard deviation of the continuous action distribution."""
        return torch.exp(self.log_std).detach().cpu().numpy()

    @std.setter
    def std(self, std: float):
        """Set the standard deviation of the continuous action distribution."""
        self.log_std.data.fill_(torch.log(torch.tensor(std, device=self.log_std.device)))