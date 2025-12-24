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
            logp: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
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

        For multi-head actors, separate ratios are computed for each head with independent clipping.

        Args:
            obs (torch.Tensor): The ``observation`` sampled from buffer.
            act (torch.Tensor): The ``action`` sampled from buffer.
            logp (torch.Tensor | tuple): The ``log probability`` (scalar or tuple for multi-head).
            adv (torch.Tensor): The ``advantage`` processed. ``reward_advantage`` here.

        Returns:
            The loss of pi/actor.
        """
        distribution = self._actor_critic.actor(obs)
        logp_current = self._actor_critic.actor.log_prob(act)

        # Dynamic type checking: handle both single-head and multi-head actors
        if isinstance(logp_current, tuple):
            # MULTI-HEAD: Decoupled ratios with independent clipping
            logp_env_current, logp_mask_current = logp_current
            logp_env_old, logp_mask_old = logp

            # Separate importance sampling ratios
            ratio_env = torch.exp(logp_env_current - logp_env_old)
            ratio_mask = torch.exp(logp_mask_current - logp_mask_old)

            # Independent clipping for each head
            clip = self._cfgs.algo_cfgs.clip
            ratio_env_clipped = torch.clamp(ratio_env, 1 - clip, 1 + clip)
            ratio_mask_clipped = torch.clamp(ratio_mask, 1 - clip, 1 + clip)

            # Use separate advantages if provided (for proper credit assignment in SPAR)
            # If adv is tuple: (adv_env, adv_mask) where env gets reward only, mask gets reward-cost
            # If adv is single tensor: use same advantage for both (backward compatibility)
            if isinstance(adv, tuple):
                adv_env, adv_mask = adv
            else:
                adv_env = adv_mask = adv

            # PPO loss for each head with appropriate advantages
            loss_env = -torch.min(ratio_env * adv_env, ratio_env_clipped * adv_env).mean()
            loss_mask = -torch.min(ratio_mask * adv_mask, ratio_mask_clipped * adv_mask).mean()

            # Weighted combination
            mask_weight = self._cfgs.algo_cfgs.get('mask_loss_weight', 0.1)
            loss = loss_env + mask_weight * loss_mask

            # Use combined ratio for logging (for backward compatibility)
            ratio = ratio_env  # Log env ratio as primary
        else:
            # SINGLE-HEAD: Standard PPO
            ratio = torch.exp(logp_current - logp)
            ratio_clipped = torch.clamp(ratio, 1 - self._cfgs.algo_cfgs.clip, 1 + self._cfgs.algo_cfgs.clip)
            loss = -torch.min(ratio * adv, ratio_clipped * adv).mean()
        if isinstance(distribution, tuple):
            # Multi-head actor: compute entropy for environment action distribution
            entropy = distribution[0].entropy().mean()
            # Add mask entropy only if mask distribution exists (not None for RandomMask)
            if distribution[1] is not None:
                mask_weight = self._cfgs.algo_cfgs.get('mask_loss_weight', 0.1)
                entropy = entropy + mask_weight * distribution[1].entropy().mean()
        else:
            entropy = distribution.entropy().mean()
        loss -= self._cfgs.algo_cfgs.entropy_coef * entropy
        if isinstance(distribution, tuple):
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
                loss = loss + aux_coef * aux_loss

                # (optional) logging
                self._logger.store({'Loss/Aux': aux_loss.item()})
        entropy = entropy.item()
        if isinstance(distribution, tuple):
            # Multi-head logging
            log_dict = {
                'Train/Entropy': entropy,
                'Train/PolicyRatio': ratio.mean().item() if isinstance(ratio, torch.Tensor) else ratio,
                'Loss/Loss_pi': loss.item(),
            }
            if distribution[1] is not None:
                log_dict['Train/MaskEntropy'] = distribution[1].entropy().mean().item()

            # Add decoupled ratio logging if multi-head
            if isinstance(logp_current, tuple):
                log_dict['Train/PolicyRatioEnv'] = ratio_env.mean().item()
                log_dict['Train/PolicyRatioMask'] = ratio_mask.mean().item()
                log_dict['Loss/Loss_pi_env'] = loss_env.item()
                log_dict['Loss/Loss_pi_mask'] = loss_mask.item()

            # Add distribution-specific metrics based on type
            from torch.distributions import Normal, Categorical

            if isinstance(distribution[0], Normal):
                log_dict['Train/ContinuousEntropy'] = distribution[0].entropy().mean().item()
                log_dict['Train/PolicyStd'] = self._actor_critic.actor.std
            elif isinstance(distribution[0], Categorical):
                log_dict['Train/DiscreteEntropy'] = distribution[0].entropy().mean().item()

            self._logger.store(log_dict)
        else:
            # Single-head logging with explicit type checking
            from torch.distributions import Normal, Categorical

            log_dict = {
                'Train/Entropy': entropy,
                'Train/PolicyRatio': ratio,
                'Loss/Loss_pi': loss.mean().item(),
            }

            # Add PolicyStd only for continuous actions
            if isinstance(distribution, Normal):
                log_dict['Train/PolicyStd'] = self._actor_critic.actor.std

            self._logger.store(log_dict)
        return loss
