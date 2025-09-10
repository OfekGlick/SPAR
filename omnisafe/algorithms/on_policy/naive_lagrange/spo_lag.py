# SPDX-License-Identifier: Apache-2.0
"""Lagrangian version of Simple Policy Optimization (SPO-Lag)."""

import numpy as np
import torch

from omnisafe.algorithms import registry
from omnisafe.algorithms.on_policy.base.spo import SPO
from omnisafe.common.lagrange import Lagrange


@registry.register
class SPOLag(SPO):
    """SPO + Lagrange multiplier for cost constraints.

    Keeps SPO's policy loss (quadratic trust-region penalty on r_t)
    and mixes reward/cost advantages via a learned λ.
    """

    def _init(self) -> None:
        """Initialize base + Lagrange multiplier."""
        super()._init()
        self._lagrange: Lagrange = Lagrange(**self._cfgs.lagrange_cfgs)

    def _init_log(self) -> None:
        """Register same logs as base + λ stat."""
        super()._init_log()
        self._logger.register_key('Metrics/LagrangeMultiplier', min_and_max=True)

    def _update(self) -> None:
        r"""Update λ, then run the standard SPO updates.

        Uses the epoch-average cost as signal for λ.
        """
        Jc = self._logger.get_stats('Metrics/EpCost')[0]
        assert not np.isnan(Jc), 'cost for updating lagrange multiplier is nan'
        self._lagrange.update_lagrange_multiplier(Jc)

        super()._update()

        self._logger.store({'Metrics/LagrangeMultiplier': self._lagrange.lagrangian_multiplier})

    def _compute_adv_surrogate(self, adv_r: torch.Tensor, adv_c: torch.Tensor) -> torch.Tensor:
        r"""Combine reward/cost advantages for the actor loss.

        L = (A^R - λ A^C) / (1 + λ)
        """
        penalty = self._lagrange.lagrangian_multiplier.item()
        return (adv_r - penalty * adv_c) / (1.0 + penalty)
