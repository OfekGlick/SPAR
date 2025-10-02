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
        distribution = self._actor_critic.actor(obs)
        logp_ = self._actor_critic.actor.log_prob(act)
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
        # TODO: track behaviour and make sure this doesn't break anything
        # >>> NEW: zero-barrier regularizer (disc head: Bernoulli over sensors)
        if isinstance(distribution, tuple) and hasattr(distribution[1], 'probs'):
            if self._cfgs.algo_cfgs.no_zero_act:
                p = distribution[1].probs.clamp(1e-6, 1 - 1e-6)  # [B, n_disc]
                zero_prob = (1.0 - p).prod(dim=-1)  # P(all zeros)
                p_atleast_one = (1.0 - zero_prob).clamp_min(self._cfgs.algo_cfgs.zero_barrier_eps)
                zero_barrier = -torch.log(p_atleast_one)  # >= 0; →0 when P(at least one)=1
                loss = loss - self._cfgs.algo_cfgs.zero_barrier_coef * zero_barrier.mean()
                # log a couple of diagnostics
                self._logger.store({'Train/ZeroProbMean': zero_prob.mean().item()})
                self._logger.store({'Reg/ZeroBarrier': zero_barrier.mean().item()})
            if self._cfgs.algo_cfgs.sd_regulizer and unmasked_observation is not None:
                # --- Sensor Dropout (SD) regularizer: student-teacher distillation ---
                # Student: use the already-computed distribution on the *masked* obs (rollout view)
                dist_student = distribution[0]

                # Teacher: run the same actor on the *unmasked* obs (full sensors, mask bits = 1s).
                # By default we DETACH teacher so grads only shape the student head.
                dist_teacher = self._actor_critic.actor(unmasked_observation)
                dist_teacher_env = dist_teacher[0]

                # Compute distillation loss based on distribution type
                from torch.distributions import Normal, Categorical

                if isinstance(dist_student, Normal):
                    # Continuous actions: MSE between means
                    mu_student = dist_student.mean  # [B, n_cont]
                    mu_teacher = dist_teacher_env.mean.detach()
                    aux_loss = (mu_student - mu_teacher).pow(2).mean()

                elif isinstance(dist_student, Categorical):
                    # Discrete actions: KL divergence KL(teacher || student)
                    # This encourages student to match teacher's action distribution
                    teacher_probs = dist_teacher_env.probs.detach()  # [B, n_actions]
                    student_logprobs = dist_student.logits - dist_student.logits.logsumexp(dim=-1, keepdim=True)

                    # KL(P||Q) = sum(P * log(P/Q)) = sum(P * (log(P) - log(Q)))
                    aux_loss = -(teacher_probs * student_logprobs).sum(dim=-1).mean()

                else:
                    raise NotImplementedError(
                        f"SD regularizer not implemented for distribution type: {type(dist_student)}"
                    )

                # Scale and add to PPO objective
                aux_coef = self._cfgs.algo_cfgs.sd_regulizer_coeff
                loss = loss - aux_coef * aux_loss

                # (optional) logging
                self._logger.store({'Loss/Aux': aux_loss.item()})
        entropy = entropy.item()
        if isinstance(distribution, tuple):
            try:
                self._logger.store(
                    {
                        'Train/Entropy': entropy,
                        'Train/ContinuousEntropy': distribution[0].entropy().mean().item(),
                        'Train/MaskEntropy': distribution[1].entropy().mean().item(),
                        'Train/PolicyRatio': ratio,
                        'Train/PolicyStd': self._actor_critic.actor.std,
                        'Loss/Loss_pi': loss.mean().item(),
                    },
                )
            except:
                self._logger.store(
                    {
                        'Train/Entropy': entropy,
                        'Train/DiscreteEntropy': distribution[0].entropy().mean().item(),
                        'Train/MaskEntropy': distribution[1].entropy().mean().item(),
                        'Train/PolicyRatio': ratio,
                        'Loss/Loss_pi': loss.mean().item(),
                    },
                )
        else:
            try:
                self._logger.store(
                    {
                        'Train/Entropy': entropy,
                        'Train/PolicyRatio': ratio,
                        'Train/PolicyStd': self._actor_critic.actor.std,
                        'Loss/Loss_pi': loss.mean().item(),
                    },
                )
            except:
                self._logger.store(
                    {
                        'Train/Entropy': entropy,
                        'Train/PolicyRatio': ratio,
                        'Loss/Loss_pi': loss.mean().item(),
                    },
                )
        return loss
