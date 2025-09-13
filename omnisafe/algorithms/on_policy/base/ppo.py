from __future__ import annotations

import torch

from omnisafe.algorithms import registry
from omnisafe.algorithms.on_policy.base.policy_gradient import PolicyGradient


@registry.register
class PPO(PolicyGradient):
    """The Proximal Policy Optimization (PPO) algorithm.

    References:
        - Title: Proximal Policy Optimization Algorithms
        - Authors: John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, Oleg Klimov.
        - URL: `PPO <https://arxiv.org/abs/1707.06347>`_
    """

    def _loss_pi(
            self,
            obs: torch.Tensor,
            act: torch.Tensor,
            logp: torch.Tensor,
            adv: torch.Tensor,
            unmasked_observation: torch.Tensor = None,
    ) -> torch.Tensor:
        r"""Computing pi/actor loss.

        In Proximal Policy Optimization, the loss is defined as:

        .. math::

            L^{CLIP} = \underset{s_t \sim \rho_{\theta}}{\mathbb{E}} \left[
                \min ( r_t A^{R}_{\pi_{\theta}} (s_t, a_t) , \text{clip} (r_t, 1 - \epsilon, 1 + \epsilon)
                A^{R}_{\pi_{\theta}} (s_t, a_t)
            \right]

        where :math:`r_t = \frac{\pi_{\theta}^{'} (a_t|s_t)}{\pi_{\theta} (a_t|s_t)}`,
        :math:`\epsilon` is the clip parameter, and :math:`A^{R}_{\pi_{\theta}} (s_t, a_t)` is the
        advantage.

        Args:
            obs (torch.Tensor): The ``observation`` sampled from buffer.
            act (torch.Tensor): The ``action`` sampled from buffer.
            logp (torch.Tensor): The ``log probability`` of action sampled from buffer.
            adv (torch.Tensor): The ``advantage`` processed. ``reward_advantage`` here.

        Returns:
            The loss of pi/actor.
        """
        # TODO: check why the current obs and the original obs are the same
        print(f"Action: {act}")
        distribution = self._actor_critic.actor(obs)
        logp_ = self._actor_critic.actor.log_prob(act)
        std = self._actor_critic.actor.std
        ratio = torch.exp(logp_ - logp)
        ratio_cliped = torch.clamp(
            ratio,
            1 - self._cfgs.algo_cfgs.clip,
            1 + self._cfgs.algo_cfgs.clip,
        )
        loss = -torch.min(ratio * adv, ratio_cliped * adv).mean()
        if isinstance(distribution, tuple):
            entropy = distribution[0].entropy().mean() + distribution[1].entropy().mean()
        else:
            entropy = distribution.entropy().mean()
        loss -= self._cfgs.algo_cfgs.entropy_coef * entropy
        # TODO: follow behaviour and make sure this doesn't break anything
        # >>> NEW: zero-barrier regularizer (disc head: Bernoulli over sensors)
        if isinstance(distribution, tuple) and hasattr(distribution[1], 'probs'):
            if self._cfgs.algo_cfgs.no_zero_act:
                p = distribution[1].probs.clamp(1e-6, 1 - 1e-6)  # [B, n_disc]
                zero_prob = (1.0 - p).prod(dim=-1)  # P(all zeros)
                p_atleast_one = (1.0 - zero_prob).clamp_min(self._cfgs.algo_cfgs.zero_barrier_eps)
                zero_barrier = -torch.log(p_atleast_one)  # >= 0; →0 when P(at least one)=1
                loss = loss + self._cfgs.algo_cfgs.zero_barrier_coef * zero_barrier.mean()
                # log a couple of diagnostics
                self._logger.store({'Train/ZeroProbMean': zero_prob.mean().item()})
                self._logger.store({'Reg/ZeroBarrier': zero_barrier.mean().item()})
            if self._cfgs.algo_cfgs.sd_regulizer and unmasked_observation is not None:
                # --- PPO substitute of Liu'17 auxiliary loss ---
                # Student: use the already-computed distribution on the *masked* obs (rollout view)
                mu_student = distribution[0].mean  # [B, n_cont]

                # Teacher: run the same actor on the *unmasked* obs (full sensors, mask bits = 1s).
                # By default we DETACH teacher so grads only shape the student head.
                dist_teacher = self._actor_critic.actor(unmasked_observation)
                mu_teacher = dist_teacher[0].mean.detach()

                # MSE between student and teacher continuous-action means
                aux_mse = (mu_student - mu_teacher).pow(2).mean()

                # Scale and add to PPO objective
                aux_coef = self._cfgs.algo_cfgs.sd_regulizer_coeff
                loss = loss + aux_coef * aux_mse

                # (optional) logging
                self._logger.store({'Loss/Aux': aux_mse.item()})
        entropy = entropy.item()
        try:
            self._logger.store(
                {
                    'Train/Entropy': entropy,
                    'Train/ContinuousEntropy': distribution[0].entropy().mean().item(),
                    'Train/DiscreteEntropy': distribution[1].entropy().mean().item(),
                    'Train/PolicyRatio': ratio,
                    'Train/PolicyStd': std,
                    'Loss/Loss_pi': loss.mean().item(),
                },
            )
        except TypeError:
            self._logger.store(
                {
                    'Train/Entropy': entropy,
                    'Train/PolicyRatio': ratio,
                    'Train/PolicyStd': std,
                    'Loss/Loss_pi': loss.mean().item(),
                },
            )
        return loss
