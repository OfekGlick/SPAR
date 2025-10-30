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
"""Implementation of ActorBuilder."""

from omnisafe.models.actor.gaussian_learning_actor import GaussianLearningActor
from omnisafe.models.actor.gaussian_sac_actor import GaussianSACActor
from omnisafe.models.actor.mlp_actor import MLPActor
from omnisafe.models.actor.perturbation_actor import PerturbationActor
from omnisafe.models.actor.vae_actor import VAE
from omnisafe.models.actor.categorical_learning_actor import CategoricalLearningActor
from omnisafe.models.base import Actor
from omnisafe.typing import Activation, ActorType, InitFunction, OmnisafeSpace
from omnisafe.models.actor.multiheadactor import MultiHeadActor
from omnisafe.models.actor.multihead_discrete_actor import MultiHeadDiscreteActor
from omnisafe.models.actor.multihead_double_actor import MultiHeadDoubleActor
from omnisafe.models.actor.multihead_double_actor_discrete import MultiHeadDoubleActorDiscrete
from omnisafe.models.actor.random_mask_actor_continuous import RandomMaskActorContinuous
from omnisafe.models.actor.random_mask_actor_discrete import RandomMaskActorDiscrete
from gymnasium import spaces

# pylint: disable-next=too-few-public-methods
class ActorBuilder:
    """Class for building actor networks.

    Args:
        obs_space (OmnisafeSpace): Observation space.
        act_space (OmnisafeSpace): Action space.
        hidden_sizes (list of int): List of hidden layer sizes.
        activation (Activation, optional): Activation function. Defaults to ``'relu'``.
        weight_initialization_mode (InitFunction, optional): Weight initialization mode. Defaults to
            ``'kaiming_uniform'``.
    """

    def __init__(
            self,
            obs_space: OmnisafeSpace,
            act_space: OmnisafeSpace,
            hidden_sizes: list[int],
            activation: Activation = 'relu',
            weight_initialization_mode: InitFunction = 'kaiming_uniform',
            env_act_space=None,
            mask_act_space=None,
    ) -> None:
        """Initialize an instance of :class:`ActorBuilder`."""
        self._obs_space: OmnisafeSpace = obs_space
        self._act_space: OmnisafeSpace = act_space
        self._weight_initialization_mode: InitFunction = weight_initialization_mode
        self._activation: Activation = activation
        self._hidden_sizes: list[int] = hidden_sizes
        self.env_act_space = env_act_space
        self.mask_act_space = mask_act_space

    # pylint: disable-next=too-many-return-statements
    def build_actor(
            self,
            actor_type: ActorType,
    ) -> Actor:
        """Build actor network.

        Currently, we support the following actor types:
            - ``gaussian_learning``: Gaussian actor with learnable standard deviation parameters.
            - ``gaussian_sac``: Gaussian actor with learnable standard deviation network.
            - ``mlp``: Multi-layer perceptron actor, used in ``DDPG`` and ``TD3``.

        Args:
            actor_type (ActorType): Type of actor network, e.g. ``gaussian_learning``.

        Returns:
            Actor network, ranging from GaussianLearningActor, GaussianSACActor to MLPActor.

        Raises:
            NotImplementedError: If the actor type is not implemented.
        """
        if actor_type == 'gaussian_learning':
            print("using GaussianLearningActor for continuous action space.")
            return GaussianLearningActor(
                self._obs_space,
                self._act_space,
                self._hidden_sizes,
                activation=self._activation,
                weight_initialization_mode=self._weight_initialization_mode,
            )
        if actor_type == 'gaussian_sac':
            return GaussianSACActor(
                self._obs_space,
                self._act_space,
                self._hidden_sizes,
                activation=self._activation,
                weight_initialization_mode=self._weight_initialization_mode,
            )
        if actor_type == 'mlp':
            return MLPActor(
                self._obs_space,
                self._act_space,
                self._hidden_sizes,
                activation=self._activation,
                weight_initialization_mode=self._weight_initialization_mode,
            )
        if actor_type == 'vae':
            return VAE(
                self._obs_space,
                self._act_space,
                self._hidden_sizes,
                activation=self._activation,
                weight_initialization_mode=self._weight_initialization_mode,
            )
        if actor_type == 'perturbation':
            return PerturbationActor(
                self._obs_space,
                self._act_space,
                self._hidden_sizes,
                activation=self._activation,
                weight_initialization_mode=self._weight_initialization_mode,
            )
        if actor_type == 'categorical_learning':
            print("using CategoricalLearningActor for discrete action space.")
            return CategoricalLearningActor(
                self._obs_space,
                self._act_space,
                self._hidden_sizes,
                activation=self._activation,
                weight_initialization_mode=self._weight_initialization_mode,
            )
        if actor_type == 'multihead':
            # Dynamically determine if we need continuous or discrete multihead actor
            # Check if action space is a Tuple
            if not isinstance(self._act_space, spaces.Tuple):
                raise ValueError(
                    f"Multihead actor requires Tuple action space, got {type(self._act_space)}"
                )

            # Use all hidden_sizes for shared network, empty head_hidden_sizes for direct projection
            # This matches the architecture of GaussianLearningActor/CategoricalLearningActor
            # Example: hidden_sizes=[64, 64]
            #   GaussianActor: obs → 64 → 64 → action
            #   MultiHeadActor: obs → 64 → 64 (shared) → cont_action & disc_action
            shared_hidden_sizes = self._hidden_sizes
            head_hidden_sizes = []  # Direct projection from shared features to outputs

            # Check if first element of Tuple is Box (continuous) or Discrete
            if isinstance(self._act_space[0], spaces.Box):
                # Continuous environment action + sensor mask
                print(f"Using MultiHeadActor with shared={shared_hidden_sizes}, direct projection heads")
                return MultiHeadActor(
                    obs_space=self._obs_space,
                    cont_act_space=self._act_space[0],
                    disc_act_space=self._act_space[1],
                    hidden_sizes=head_hidden_sizes,
                    activation=self._activation,
                    shared_hidden_sizes=shared_hidden_sizes,
                    weight_initialization_mode=self._weight_initialization_mode,
                )
            else:
                # Discrete environment action + sensor mask
                print(f"Using MultiHeadDiscreteActor with shared={shared_hidden_sizes}, direct projection heads")
                return MultiHeadDiscreteActor(
                    obs_space=self._obs_space,
                    disc_env_act_space=self._act_space[0],
                    disc_mask_space=self._act_space[1],
                    hidden_sizes=head_hidden_sizes,
                    activation=self._activation,
                    shared_hidden_sizes=shared_hidden_sizes,
                    weight_initialization_mode=self._weight_initialization_mode,
                )
        if actor_type == 'random_mask':
            # Random mask baseline: learns environment actions but uses random modality masks
            # Dynamically determine if we need continuous or discrete version
            if not isinstance(self._act_space, spaces.Tuple):
                raise ValueError(
                    f"Random mask actor requires Tuple action space, got {type(self._act_space)}"
                )

            # Use same architecture as multihead: all layers shared, direct projection heads
            shared_hidden_sizes = self._hidden_sizes
            head_hidden_sizes = []  # Direct projection from shared features to outputs

            # Check if first element of Tuple is Box (continuous) or Discrete
            if isinstance(self._act_space[0], spaces.Box):
                # Continuous environment action + random mask
                print(f"Using RandomMaskActorContinuous with shared={shared_hidden_sizes}, direct projection head")
                return RandomMaskActorContinuous(
                    obs_space=self._obs_space,
                    cont_act_space=self._act_space[0],
                    disc_act_space=self._act_space[1],
                    hidden_sizes=head_hidden_sizes,
                    activation=self._activation,
                    shared_hidden_sizes=shared_hidden_sizes,
                    weight_initialization_mode=self._weight_initialization_mode,
                )
            else:
                # Discrete environment action + random mask
                print(f"Using RandomMaskActorDiscrete with shared={shared_hidden_sizes}, direct projection head")
                return RandomMaskActorDiscrete(
                    obs_space=self._obs_space,
                    disc_act_space=self._act_space[0],
                    disc_mask_space=self._act_space[1],
                    hidden_sizes=head_hidden_sizes,
                    activation=self._activation,
                    shared_hidden_sizes=shared_hidden_sizes,
                    weight_initialization_mode=self._weight_initialization_mode,
                )
        if actor_type == 'multihead_double':
            # Double network architecture: completely independent networks for env and mask actions
            # Dynamically determine if we need continuous or discrete version
            if not isinstance(self._act_space, spaces.Tuple):
                raise ValueError(
                    f"Multihead double actor requires Tuple action space, got {type(self._act_space)}"
                )

            # Use all hidden_sizes for BOTH networks (independent architectures)
            env_hidden_sizes = self._hidden_sizes  # e.g., [64, 64] for environment action network
            mask_hidden_sizes = self._hidden_sizes  # e.g., [64, 64] for mask action network (can be different)

            # Check if first element of Tuple is Box (continuous) or Discrete
            if isinstance(self._act_space[0], spaces.Box):
                # Continuous environment action + sensor mask
                print(f"Using MultiHeadDoubleActor with env_net={env_hidden_sizes}, mask_net={mask_hidden_sizes}")
                return MultiHeadDoubleActor(
                    obs_space=self._obs_space,
                    cont_act_space=self._act_space[0],
                    disc_act_space=self._act_space[1],
                    env_hidden_sizes=env_hidden_sizes,
                    mask_hidden_sizes=mask_hidden_sizes,
                    activation=self._activation,
                    weight_initialization_mode=self._weight_initialization_mode,
                )
            else:
                # Discrete environment action + sensor mask
                print(f"Using MultiHeadDoubleActorDiscrete with env_net={env_hidden_sizes}, mask_net={mask_hidden_sizes}")
                return MultiHeadDoubleActorDiscrete(
                    obs_space=self._obs_space,
                    disc_env_act_space=self._act_space[0],
                    disc_mask_space=self._act_space[1],
                    env_hidden_sizes=env_hidden_sizes,
                    mask_hidden_sizes=mask_hidden_sizes,
                    activation=self._activation,
                    weight_initialization_mode=self._weight_initialization_mode,
                )
        raise NotImplementedError(
            f'Actor type {actor_type} is not implemented! '
            f'Available actor types are: gaussian_learning, gaussian_sac, mlp, vae, perturbation, '
            f'categorical_learning, multihead, random_mask, multihead_double.',
        )
