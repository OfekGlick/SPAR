from __future__ import annotations

import torch

from omnisafe.algorithms import registry
from omnisafe.algorithms.on_policy.base.policy_gradient import PolicyGradient

@registry.register
class SPO(PolicyGradient):
    """Simple Policy Optimization (SPO).

    Uses a quadratic trust-region penalty instead of PPO's clipping:

      J(θ) = E[ r_t(θ) * Â_t  -  |Â_t|/(2ε) * (r_t(θ) - 1)^2 ]
      with r_t = π_θ(a|s) / π_old(a|s), minimize -J.

    Reuses algo_cfgs.clip as ε and PolicyGradient’s plumbing (KL early stop, critics, etc.).
    """

    def _loss_pi(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        logp: torch.Tensor,   # log π_old(a|s)
        adv: torch.Tensor,    # reward advantage (already standardized if configured)
    ) -> torch.Tensor:
        # Forward current policy
        distribution = self._actor_critic.actor(obs)
        logp_new = self._actor_critic.actor.log_prob(act)
        std = self._actor_critic.actor.std  # matches your PPO/PG usage

        # Ratio r_t
        ratio = torch.exp(logp_new - logp)

        # SPO penalty weight ε: reuse PPO's clip param
        eps = max(float(self._cfgs.algo_cfgs.clip), 1e-8)

        # SPO surrogate: r*A - |A|/(2ε)*(r-1)^2
        quad_penalty = (torch.abs(adv) / (2.0 * eps)) * (ratio - 1.0).pow(2)
        spo_obj = ratio * adv - quad_penalty
        loss = -spo_obj.mean()

        # Entropy bonus (same pattern as your code)
        try:
            entropy = sum(d.entropy().mean() for d in distribution)
            entropy_val = float(entropy.item())
            loss -= self._cfgs.algo_cfgs.entropy_coef * entropy
            self._logger.store(
                {
                    'Train/Entropy': entropy_val,
                    'Train/ContinuousEntropy': distribution[0].entropy().mean().item(),
                    'Train/DiscreteEntropy': distribution[1].entropy().mean().item(),
                    'Train/PolicyRatio': ratio,
                    'Train/PolicyStd': std,
                    'Loss/Loss_pi': loss.mean().item(),
                    # Optional extras (register if your Logger requires):
                    # 'Train/RatioDeviation': torch.mean(torch.abs(ratio - 1.0)).item(),
                    # 'Train/SPOPenalty': quad_penalty.mean().item(),
                },
            )
        except Exception:
            entropy = distribution.entropy().mean()
            entropy_val = float(entropy.item())
            loss -= self._cfgs.algo_cfgs.entropy_coef * entropy
            self._logger.store(
                {
                    'Train/Entropy': entropy_val,
                    'Train/PolicyRatio': ratio,
                    'Train/PolicyStd': std,
                    'Loss/Loss_pi': loss.mean().item(),
                    # Optional: 'Train/RatioDeviation': torch.mean(torch.abs(ratio - 1.0)).item(),
                    # Optional: 'Train/SPOPenalty': quad_penalty.mean().item(),
                },
            )

        return loss
