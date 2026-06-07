from __future__ import annotations

from typing import Any

import itertools
import gymnasium
from packaging import version

import torch
import torch.nn as nn
import torch.nn.functional as F

from skrl import config, logger
from skrl.agents.torch import Agent
from skrl.memories.torch import Memory
from skrl.models.torch import Model
from skrl.resources.schedulers.torch import KLAdaptiveLR
from skrl.utils import ScopedTimer

from .sitt_cfg import SITT_CFG


def compute_gae(
    *,
    rewards: torch.Tensor,
    terminated: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    discount_factor: float = 0.99,
    lambda_coefficient: float = 0.95,
) -> torch.Tensor:
    """Compute the Generalized Advantage Estimator (GAE).

    :param rewards: Rewards obtained by the agent.
    :param terminated: Signals to indicate that episodes have ended.
    :param values: Values obtained by the agent.
    :param next_values: Next values obtained by the agent.
    :param discount_factor: Discount factor.
    :param lambda_coefficient: Lambda coefficient.

    :return: Generalized Advantage Estimator.
    """
    advantage = 0
    advantages = torch.zeros_like(rewards)
    not_terminated = terminated.logical_not()
    memory_size = rewards.shape[0]

    # advantages computation
    for i in reversed(range(memory_size)):
        next_values = values[i + 1] if i < memory_size - 1 else next_values
        advantage = (
            rewards[i]
            - values[i]
            + discount_factor * not_terminated[i] * (next_values + lambda_coefficient * advantage)
        )
        advantages[i] = advantage
    # returns computation
    returns = advantages + values
    # normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    return returns, advantages

def _register_backbone_hook(model: Model) -> tuple[list, nn.Module | None, Any | None]:
    """Register a forward pre-hook on the last module of model.net_container.

    The last module is the action head (LazyLinear → action_dim).
    Its INPUT is the backbone feature vector (256-d for our networks).

    Returns (storage_list, action_head_module, hook_handle).
    storage_list[0] will hold the captured feature tensor after each forward pass.
    """
    storage = [None]
    if not hasattr(model, "net_container"):
        return storage, None, None
    children = list(model.net_container.children())
    if len(children) < 2:
        return storage, None, None

    action_head = children[-1]  # the LazyLinear that maps features → actions

    def _hook(module, args):
        storage[0] = args[0]  # input to the action head = backbone features

    handle = action_head.register_forward_pre_hook(_hook)
    return storage, action_head, handle


class SITT(Agent):
    """Student-Informed Teacher Training (SITT).

    Full implementation of https://arxiv.org/pdf/2412.09149.

    Three networks, each output 256-d backbone features before an action head:
      Teacher  F_T  : kinematics → MLP [128, 256] → 256-d → action head
      Student  F_S  : frames     → CNN + dense    → 256-d → action head
      Proxy    F̂_S  : kinematics → MLP [128, 256] → 256-d → action head

    Per-update training:
      1. Teacher PPO  + proxy-alignment L1 penalty (pulls teacher features ≈ proxy).
      2. Alignment phase:
           proxy backbone → student features  (L1 feat + L1 via teacher head)
           student backbone → teacher features (L1 feat + L1 via teacher head)
    Reward shaping each step:  rewards -= kl_penalty_scale * KL(teacher ‖ proxy)
    """

    def __init__(
        self,
        *,
        models: dict[str, Model],
        memory: Memory | None = None,
        observation_space: gymnasium.Space | None = None,
        state_space: gymnasium.Space | None = None,
        action_space: gymnasium.Space | None = None,
        device: str | torch.device | None = None,
        cfg: SITT_CFG | dict = {},
    ) -> None:
        """Student Informed Teacher Training (SITT)

        https://arxiv.org/pdf/2412.09149

        :param models: Agent's models.
        :param memory: Memory to storage agent's data and environment transitions.
        :param observation_space: Observation space.
        :param state_space: State space.
        :param action_space: Action space.
        :param device: Data allocation and computation device. If not specified, the default device will be used.
        :param cfg: Agent's configuration.

        :raises KeyError: If a configuration key is missing.
        """
        self.cfg: SITT_CFG
        super().__init__(
            models=models,
            memory=memory,
            observation_space=observation_space,
            state_space=state_space,
            action_space=action_space,
            device=device,
            cfg=SITT_CFG(**cfg) if isinstance(cfg, dict) else cfg,
        )

        # ── models ────────────────────────────────────────────────────────
        self.teacher = self.models.get("teacher", None)
        self.student = self.models.get("student", None)
        self.proxy_student = self.models.get("proxy_student", None)
        self.value = self.models.get("value", None)

        for name in ("teacher", "student", "proxy_student", "value"):
            m = getattr(self, name)
            if m is not None:
                self.checkpoint_modules[name] = m

        # broadcast parameters in distributed runs
        if config.torch.is_distributed:
            logger.info("Broadcasting models' parameters")
            for m in (self.teacher, self.student, self.proxy_student, self.value):
                if m is not None:
                    m.broadcast_parameters()

        # ── AMP scaler ────────────────────────────────────────────────────
        self._device_type = torch.device(self.device).type
        if version.parse(torch.__version__) >= version.parse("2.4"):
            self.scaler = torch.amp.GradScaler(device=self._device_type, enabled=self.cfg.mixed_precision)
        else:
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.cfg.mixed_precision)

        # ── optimizers ────────────────────────────────────────────────────
        if self.teacher is not None and self.value is not None:
            if self.teacher is self.value:
                self.optimizer = torch.optim.Adam(self.teacher.parameters(), lr=self.cfg.learning_rate[0])
            else:
                self.optimizer = torch.optim.Adam(
                    itertools.chain(self.teacher.parameters(), self.value.parameters()),
                    lr=self.cfg.learning_rate[0],
                )
            self.checkpoint_modules["optimizer"] = self.optimizer
            self.scheduler = self.cfg.learning_rate_scheduler[0]
            if self.scheduler is not None:
                self.scheduler = self.cfg.learning_rate_scheduler[0](
                    self.optimizer, **self.cfg.learning_rate_scheduler_kwargs[0]
                )

        if self.student is not None:
            self.student_optimizer = torch.optim.Adam(
                self.student.parameters(), lr=self.cfg.student_learning_rate
            )
            self.checkpoint_modules["student_optimizer"] = self.student_optimizer
            self.student_scheduler = self.cfg.learning_rate_scheduler[1]
            if self.student_scheduler is not None:
                self.student_scheduler = self.cfg.learning_rate_scheduler[1](
                    self.student_optimizer, **self.cfg.learning_rate_scheduler_kwargs[1]
                )

        if self.proxy_student is not None:
            self.proxy_optimizer = torch.optim.Adam(
                self.proxy_student.parameters(), lr=self.cfg.student_learning_rate
            )
            self.checkpoint_modules["proxy_optimizer"] = self.proxy_optimizer
        else:
            self.proxy_optimizer = None

        # ── preprocessors ─────────────────────────────────────────────────
        if self.cfg.observation_preprocessor:
            self._observation_preprocessor = self.cfg.observation_preprocessor(
                **self.cfg.observation_preprocessor_kwargs
            )
            self.checkpoint_modules["observation_preprocessor"] = self._observation_preprocessor
        else:
            self._observation_preprocessor = self._empty_preprocessor
        if self.cfg.state_preprocessor:
            self._state_preprocessor = self.cfg.state_preprocessor(**self.cfg.state_preprocessor_kwargs)
            self.checkpoint_modules["state_preprocessor"] = self._state_preprocessor
        else:
            self._state_preprocessor = self._empty_preprocessor
        if self.cfg.value_preprocessor:
            self._value_preprocessor = self.cfg.value_preprocessor(**self.cfg.value_preprocessor_kwargs)
            self.checkpoint_modules["value_preprocessor"] = self._value_preprocessor
        else:
            self._value_preprocessor = self._empty_preprocessor

        # ── backbone feature hooks ─────────────────────────────────────────
        # Register a forward pre-hook on the last module of each net_container
        # (the action head, LazyLinear → num_actions).  Its INPUT is the 256-d
        # backbone feature vector used for feature-level alignment.
        self._teacher_feats, self._teacher_head, _th = _register_backbone_hook(self.teacher) if self.teacher else ([None], None, None)
        self._student_feats, self._student_head, _sh = _register_backbone_hook(self.student) if self.student else ([None], None, None)
        if self.proxy_student is not None:
            self._proxy_feats, self._proxy_head, _ph = _register_backbone_hook(self.proxy_student)
        else:
            self._proxy_feats, self._proxy_head, _ph = [None], None, None

        # Keep the teacher's action head as the canonical shared decoder
        # (used to project student/proxy features → action space for alignment)
        self._canonical_head = self._teacher_head  # nn.Module or None

    # ──────────────────────────────────────────────────────────────────────
    def init(self, *, trainer_cfg: dict[str, Any] | None = None) -> None:
        super().init(trainer_cfg=trainer_cfg)
        self.enable_models_training_mode(False)

        if self.memory is not None:
            self.memory.create_tensor(name="observations", size=self.observation_space, dtype=torch.float32)
            self.memory.create_tensor(name="states", size=self.state_space, dtype=torch.float32)
            self.memory.create_tensor(name="actions", size=self.action_space, dtype=torch.float32)
            self.memory.create_tensor(name="rewards", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="terminated", size=1, dtype=torch.bool)
            self.memory.create_tensor(name="log_prob", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="values", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="returns", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="advantages", size=1, dtype=torch.float32)
            self._tensors_names = ["observations", "states", "actions", "log_prob", "values", "returns", "advantages"]

        self._current_next_observations = None
        self._current_next_states = None
        self._current_log_prob = None
        self._current_values = None
        self._current_kl_div = None
        self._rollout = 0

    # ──────────────────────────────────────────────────────────────────────
    def act(
        self, observations: torch.Tensor, states: torch.Tensor | None, *, timestep: int, timesteps: int
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Process the environment's observations/states to make a decision (actions) using the main policy.

        :param observations: Environment observations.
        :param states: Environment states.
        :param timestep: Current timestep.
        :param timesteps: Number of timesteps.

        :return: Agent output. The first component is the expected action/value returned by the agent.
            The second component is a dictionary containing extra output values according to the model.
        """
        inputs = {
            "observations": self._observation_preprocessor(observations),
            "states": self._state_preprocessor(states),
        }

        if timestep < self.cfg.random_timesteps:
            return self.teacher.random_act(inputs, role="policy")

        with torch.autocast(device_type=self._device_type, enabled=self.cfg.mixed_precision):
            if self.training:
                teacher_actions, teacher_outputs = self.teacher.act(inputs, role="teacher")
                self._current_log_prob = teacher_outputs["log_prob"]

                # KL reward penalty: penalise teacher for actions proxy can't predict
                if (
                    self.proxy_student is not None
                    and self.cfg.kl_penalty_scale > 0
                    and timestep >= self.cfg.start_student_training_timestep
                ):
                    with torch.no_grad():
                        self.proxy_student.act(inputs, role="proxy")
                    teacher_dist = self.teacher.distribution()
                    proxy_dist = self.proxy_student.distribution()
                    self._current_kl_div = torch.distributions.kl_divergence(
                        teacher_dist, proxy_dist
                    ).sum(dim=-1)  # (num_envs,)
                else:
                    self._current_kl_div = None

                values, _ = self.value.act(inputs, role="value")
                self._current_values = self._value_preprocessor(values, inverse=True)
                return teacher_actions, teacher_outputs
            else:
                # Evaluation uses the student
                student_actions, student_outputs = self.student.act(inputs, role="student")
                return student_actions, student_outputs

    # ──────────────────────────────────────────────────────────────────────
    def record_transition(
        self,
        *,
        observations: torch.Tensor,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_observations: torch.Tensor,
        next_states: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
        infos: Any,
        timestep: int,
        timesteps: int,
    ) -> None:
        """Record an environment transition in memory.

        :param observations: Environment observations.
        :param states: Environment states.
        :param actions: Actions taken by the agent.
        :param rewards: Instant rewards achieved by the current actions.
        :param next_observations: Next environment observations.
        :param next_states: Next environment states.
        :param terminated: Signals that indicate episodes have terminated.
        :param truncated: Signals that indicate episodes have been truncated.
        :param infos: Additional information about the environment.
        :param timestep: Current timestep.
        :param timesteps: Number of timesteps.
        """
        super().record_transition(
            observations=observations,
            states=states,
            actions=actions,
            rewards=rewards,
            next_observations=next_observations,
            next_states=next_states,
            terminated=terminated,
            truncated=truncated,
            infos=infos,
            timestep=timestep,
            timesteps=timesteps,
        )

        if self.training:
            self._current_next_observations = next_observations
            self._current_next_states = next_states

            # reward shaping
            if self.cfg.rewards_shaper is not None:
                rewards = self.cfg.rewards_shaper(rewards, timestep, timesteps)

            # time-limit (truncation) bootstrapping
            if self.cfg.time_limit_bootstrap:
                rewards += self.cfg.discount_factor * self._current_values * truncated

            # Penalise teacher when its actions diverge from what proxy (≈ student) can predict
            if self._current_kl_div is not None and self.cfg.kl_penalty_scale > 0:
                rewards = rewards - self.cfg.kl_penalty_scale * self._current_kl_div.unsqueeze(-1)

            self.memory.add_samples(
                observations=observations,
                states=states,
                actions=actions,
                rewards=rewards,
                terminated=terminated,
                log_prob=self._current_log_prob,
                values=self._current_values,
            )

    def pre_interaction(self, *, timestep: int, timesteps: int) -> None:
        """Method called before the interaction with the environment.

        :param timestep: Current timestep.
        :param timesteps: Number of timesteps.
        """
        pass

    def post_interaction(self, *, timestep: int, timesteps: int) -> None:
        """Method called after the interaction with the environment.

        :param timestep: Current timestep.
        :param timesteps: Number of timesteps.
        """
        if self.training:
            self._rollout += 1
            if not self._rollout % self.cfg.rollouts and timestep >= self.cfg.learning_starts:
                with ScopedTimer() as timer:
                    self.enable_models_training_mode(True)
                    self.update(timestep=timestep, timesteps=timesteps)
                    self.enable_models_training_mode(False)
                    self.track_data("Stats / Algorithm update time (ms)", timer.elapsed_time_ms)
        super().post_interaction(timestep=timestep, timesteps=timesteps)

    # ──────────────────────────────────────────────────────────────────────
    def _get_features(self, feat_storage: list, fallback_dist_mean: torch.Tensor | None) -> torch.Tensor | None:
        """Return backbone features captured by the pre-hook, or fall back to action means."""
        feat = feat_storage[0]
        if feat is not None:
            return feat
        return fallback_dist_mean  # None or distribution.mean

    # ──────────────────────────────────────────────────────────────────────
    def update(self, *, timestep: int, timesteps: int) -> None:
        """Teacher PPO (always) + proxy / student alignment (after warmup)."""

        joint = timestep >= (self.cfg.start_student_training_timestep + self.cfg.rollouts)

        # ── 1. GAE ────────────────────────────────────────────────────────
        with torch.no_grad(), torch.autocast(device_type=self._device_type, enabled=self.cfg.mixed_precision):
            inputs_next = {
                "observations": self._observation_preprocessor(self._current_next_observations),
                "states": self._state_preprocessor(self._current_next_states),
            }
            self.value.enable_training_mode(False)
            last_values, _ = self.value.act(inputs_next, role="value")
            self.value.enable_training_mode(True)
            last_values = self._value_preprocessor(last_values, inverse=True)

        values = self.memory.get_tensor_by_name("values")
        returns, advantages = compute_gae(
            rewards=self.memory.get_tensor_by_name("rewards"),
            terminated=self.memory.get_tensor_by_name("terminated"),
            values=values,
            next_values=last_values,
            discount_factor=self.cfg.discount_factor,
            lambda_coefficient=self.cfg.gae_lambda,
        )
        self.memory.set_tensor_by_name("values", self._value_preprocessor(values, train=True))
        self.memory.set_tensor_by_name("returns", self._value_preprocessor(returns, train=True))
        self.memory.set_tensor_by_name("advantages", advantages)

        # ── 2. Teacher PPO ────────────────────────────────────────────────
        sampled = self.memory.sample_all(names=self._tensors_names, mini_batches=self.cfg.mini_batches)
        cum_pol = cum_ent = cum_val = cum_proxy_in_ppo = 0.0

        for epoch in range(self.cfg.learning_epochs):
            kl_divs = []

            for s_obs, s_states, s_acts, s_logp, s_vals, s_ret, s_adv in sampled:
                with torch.autocast(device_type=self._device_type, enabled=self.cfg.mixed_precision):
                    inputs = {
                        "observations": self._observation_preprocessor(s_obs),
                        "states": self._state_preprocessor(s_states),
                    }

                    _, out = self.teacher.act({**inputs, "taken_actions": s_acts}, role="teacher")
                    next_logp = out["log_prob"]

                    with torch.no_grad():
                        r = next_logp - s_logp
                        kl = ((torch.exp(r) - 1) - r).mean()
                        kl_divs.append(kl)

                    if self.cfg.kl_threshold and kl > self.cfg.kl_threshold:
                        break

                    ent_loss = (
                        -self.cfg.entropy_loss_scale * self.teacher.get_entropy(role="teacher").mean()
                        if self.cfg.entropy_loss_scale else 0
                    )

                    ratio = torch.exp(next_logp - s_logp)
                    pol_loss = -torch.min(
                        s_adv * ratio,
                        s_adv * torch.clip(ratio, 1 - self.cfg.ratio_clip, 1 + self.cfg.ratio_clip),
                    ).mean()

                    pv, _ = self.value.act(inputs, role="value")
                    if self.cfg.value_clip > 0:
                        pv = s_vals + torch.clip(pv - s_vals, -self.cfg.value_clip, self.cfg.value_clip)
                    val_loss = self.cfg.value_loss_scale * F.mse_loss(s_ret, pv)

                    loss = pol_loss + ent_loss + val_loss

                    # SITT: pull teacher toward proxy (which tracks student) – Eq. 8
                    if joint and self.proxy_student is not None and self.cfg.proxy_alignment_scale > 0:
                        teacher_feat = self._get_features(self._teacher_feats, self.teacher.distribution().mean)
                        with torch.no_grad():
                            self.proxy_student.act(inputs, role="proxy")
                        proxy_feat = self._get_features(self._proxy_feats, self.proxy_student.distribution().mean)
                        if teacher_feat is not None and proxy_feat is not None and teacher_feat.shape == proxy_feat.shape:
                            proxy_ppo_loss = F.l1_loss(teacher_feat, proxy_feat)
                        else:
                            proxy_ppo_loss = F.l1_loss(
                                self.teacher.distribution().mean, self.proxy_student.distribution().mean
                            )
                        loss = loss + self.cfg.proxy_alignment_scale * proxy_ppo_loss
                        cum_proxy_in_ppo += proxy_ppo_loss.item()

                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()

                if config.torch.is_distributed:
                    self.teacher.reduce_parameters()
                    if self.teacher is not self.value:
                        self.value.reduce_parameters()

                if self.cfg.grad_norm_clip > 0:
                    self.scaler.unscale_(self.optimizer)
                    params = self.teacher.parameters() if self.teacher is self.value else \
                             itertools.chain(self.teacher.parameters(), self.value.parameters())
                    nn.utils.clip_grad_norm_(params, self.cfg.grad_norm_clip)

                self.scaler.step(self.optimizer)
                self.scaler.update()

                cum_pol += pol_loss.item()
                cum_val += val_loss.item()
                if self.cfg.entropy_loss_scale:
                    cum_ent += ent_loss.item()

            if self.scheduler:
                if isinstance(self.scheduler, KLAdaptiveLR):
                    kl_t = torch.tensor(kl_divs, device=self.device).mean()
                    if config.torch.is_distributed:
                        torch.distributed.all_reduce(kl_t, op=torch.distributed.ReduceOp.SUM)
                        kl_t /= config.torch.world_size
                    self.scheduler.step(kl_t.item())
                else:
                    self.scheduler.step()

        n_t = self.cfg.learning_epochs * self.cfg.mini_batches
        self.track_data("Loss / Teacher Policy loss", cum_pol / n_t)
        self.track_data("Loss / Value loss", cum_val / n_t)
        if self.cfg.entropy_loss_scale:
            self.track_data("Loss / Entropy loss", cum_ent / n_t)
        if joint and self.proxy_student is not None:
            self.track_data("Loss / Proxy-in-PPO alignment", cum_proxy_in_ppo / n_t)
        self.track_data("Policy / Teacher StdDev", self.teacher.distribution().stddev.mean().item())
        if self.scheduler:
            self.track_data("Learning / LR", self.scheduler.get_last_lr()[0])

        if not joint or self.student is None:
            return

        # ── 3. Alignment phase ────────────────────────────────────────────
        # Teacher's action head (last module of net_container) is frozen during
        # alignment so gradient flows only through student / proxy backbones.
        canon = self._canonical_head  # nn.Module or None
        if canon is not None:
            for p in canon.parameters():
                p.requires_grad_(False)

        align_batches = self.memory.sample_all(
            names=self._tensors_names, mini_batches=self.cfg.alignment_mini_batches
        )
        cum_proxy_align = cum_student_align = 0.0

        for epoch in range(self.cfg.alignment_epochs):
            for s_obs, s_states, *_ in align_batches:
                with torch.autocast(device_type=self._device_type, enabled=self.cfg.mixed_precision):
                    inputs = {
                        "observations": self._observation_preprocessor(s_obs),
                        "states": self._state_preprocessor(s_states),
                    }

                    # ── teacher forward (no grad) ──────────────────────────
                    with torch.no_grad():
                        self.teacher.act(inputs, role="teacher")
                        t_feat = self._get_features(self._teacher_feats, self.teacher.distribution().mean)
                        if canon is not None and t_feat is not None:
                            t_act = canon(t_feat)
                        else:
                            t_act = self.teacher.distribution().mean

                    # ── proxy → student alignment ─────────────────────────
                    if self.proxy_student is not None:
                        self.proxy_student.act(inputs, role="proxy")
                        p_feat = self._get_features(self._proxy_feats, self.proxy_student.distribution().mean)

                        with torch.no_grad():
                            self.student.act(inputs, role="student")
                            s_feat_tgt = self._get_features(self._student_feats, self.student.distribution().mean)

                        if (p_feat is not None and s_feat_tgt is not None
                                and p_feat.shape == s_feat_tgt.shape):
                            proxy_loss = F.l1_loss(p_feat, s_feat_tgt.detach())
                            if canon is not None:
                                proxy_loss = proxy_loss + F.l1_loss(canon(p_feat), canon(s_feat_tgt).detach())
                        else:
                            proxy_loss = torch.distributions.kl_divergence(
                                self.student.distribution(), self.proxy_student.distribution()
                            ).mean()

                        self.proxy_optimizer.zero_grad()
                        self.scaler.scale(proxy_loss).backward(retain_graph=True)
                        if self.cfg.grad_norm_clip > 0:
                            self.scaler.unscale_(self.proxy_optimizer)
                            nn.utils.clip_grad_norm_(self.proxy_student.parameters(), self.cfg.grad_norm_clip)
                        self.scaler.step(self.proxy_optimizer)
                        self.scaler.update()
                        cum_proxy_align += proxy_loss.item()

                    # ── student → teacher alignment ───────────────────────
                    # Recompute student forward so the retained graph is fresh
                    self.student.act(inputs, role="student")
                    s_feat = self._get_features(self._student_feats, self.student.distribution().mean)

                    if s_feat is not None and t_feat is not None and s_feat.shape == t_feat.shape:
                        student_loss = F.l1_loss(s_feat, t_feat.detach())
                        if canon is not None:
                            student_loss = student_loss + F.l1_loss(canon(s_feat), t_act.detach())
                    else:
                        student_loss = torch.distributions.kl_divergence(
                            self.teacher.distribution(), self.student.distribution()
                        ).mean()

                    self.student_optimizer.zero_grad()
                    self.scaler.scale(student_loss).backward()
                    if self.cfg.grad_norm_clip > 0:
                        self.scaler.unscale_(self.student_optimizer)
                        nn.utils.clip_grad_norm_(self.student.parameters(), self.cfg.grad_norm_clip)
                    self.scaler.step(self.student_optimizer)
                    self.scaler.update()
                    cum_student_align += student_loss.item()

        if canon is not None:
            for p in canon.parameters():
                p.requires_grad_(True)

        n_a = self.cfg.alignment_epochs * self.cfg.alignment_mini_batches
        if self.proxy_student is not None:
            self.track_data("Loss / Proxy alignment", cum_proxy_align / n_a)
        self.track_data("Loss / Student alignment", cum_student_align / n_a)
        self.track_data("Policy / Student StdDev", self.student.distribution().stddev.mean().item())
