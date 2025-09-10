import torch
import torch.nn as nn
from torch.distributions import Normal, Bernoulli
from omnisafe.models.base import Actor
from omnisafe.utils.model import build_mlp_network


class MultiHeadActor(Actor):
    """MultiHeadActor with shared layers for continuous and discrete actions."""

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
        """Initialize the MultiHeadActor with shared layers."""
        super().__init__(obs_space, cont_act_space, hidden_sizes, activation, weight_initialization_mode)

        # Shared layers
        self.shared_net = build_mlp_network(
            sizes=[self._obs_dim, *shared_hidden_sizes],
            activation=activation,
            weight_initialization_mode=weight_initialization_mode,
        )

        # Continuous action head
        self.continuous_net = build_mlp_network(
            sizes=[shared_hidden_sizes[-1], *hidden_sizes, cont_act_space.shape[0]],
            activation=activation,
            weight_initialization_mode=weight_initialization_mode,
        )
        self.log_std = nn.Parameter(torch.zeros(cont_act_space.shape[0]))

        # Discrete action head
        self.discrete_net = build_mlp_network(
            sizes=[shared_hidden_sizes[-1], *hidden_sizes, disc_act_space.shape[0]],
            activation=activation,
            weight_initialization_mode=weight_initialization_mode,
        )

        self.cont_action_space = cont_act_space
        self.disc_action_space = disc_act_space
        self._current_cont_dist = None
        self._current_disc_dist = None

    def _distribution(self, obs):
        """Generate and return distributions for continuous and discrete actions."""
        shared_features = self.shared_net(obs)

        # Continuous action distribution
        cont_mean = self.continuous_net(shared_features)
        std = torch.exp(self.log_std)
        cont_dist = Normal(cont_mean, std)

        # Discrete action distribution
        logits = self.discrete_net(shared_features)
        disc_dist = Bernoulli(logits=logits)

        return cont_dist, disc_dist

    def forward(self, obs):
        """Generate and cache action distributions."""
        self._current_cont_dist, self._current_disc_dist = self._distribution(obs)
        self._after_inference = True
        return self._current_cont_dist, self._current_disc_dist

    def predict(self, obs, deterministic=False):
        """Predict both continuous and discrete actions and cache the distributions."""
        self._current_cont_dist, self._current_disc_dist = self._distribution(obs)

        # Predict continuous action
        cont_action = self._current_cont_dist.mean if deterministic else self._current_cont_dist.rsample()

        # Predict discrete action
        binary_action = self._current_disc_dist.probs.round() if deterministic else self._current_disc_dist.sample()
        self._after_inference = True
        return cont_action, binary_action

    def log_prob(self, act):
        """Calculate log probability of the given actions based on cached distributions.

        Args:
            act (torch.Tensor): Combined action tensor, where continuous actions are followed by discrete.

        Returns:
            torch.Tensor: Log probability of the actions.
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
        cont_log_prob = self._current_cont_dist.log_prob(cont_action).sum(dim=-1)
        disc_log_prob = self._current_disc_dist.log_prob(disc_action).sum(dim=-1)

        return cont_log_prob + disc_log_prob


    @property
    def std(self) -> float:
        """Get the standard deviation of the normal distribution."""
        return torch.exp(self.log_std).detach().cpu().numpy()


    @std.setter
    def std(self, std: float) -> None:
        """Set the standard deviation of the normal distribution."""
        self.log_std.data.fill_(torch.log(torch.tensor(std, device=self.log_std.device)))
