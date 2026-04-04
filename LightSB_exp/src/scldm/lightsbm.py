"""
Schrödinger Bridge Matching models for cell perturbation prediction.

LightSBM base class is adapted from:
  https://github.com/ngushchin/LightSBM  (ICML 2024)

ConditionedLightSBM extends LightSBM to accept per-sample perturbation
condition embeddings, enabling the bridge to be conditioned on the
perturbation identity (e.g. which gene was knocked out).
"""
import math

import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.distributions.independent import Independent
from torch.distributions.mixture_same_family import MixtureSameFamily
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal


class LightSBM(nn.Module):
    """
    Light Schrödinger Bridge Matching (LightSBM) from Gushchin et al., ICML 2024.

    Parameterises the terminal potential ψ(x) as a Gaussian mixture and learns
    it via the bridge-matching MSE loss.  Only the diagonal covariance case is
    fully tested here; the full covariance path (`is_diagonal=False`) is kept
    for completeness but is not required by `ConditionedLightSBM`.

    Args:
        dim:                  dimensionality of the latent space
        n_potentials:         number of GMM components (K)
        epsilon:              SB entropic regularisation ε
        is_diagonal:          if True use diagonal S; if False use full orthogonal × diagonal
        sampling_batch_size:  sub-batch size when running the forward bridge sampler
        S_diagonal_init:      initial value of diagonal S entries
        r_scale:              unused scaling kept for API compatibility
    """

    def __init__(
        self,
        dim: int = 2,
        n_potentials: int = 5,
        epsilon: float = 1.0,
        is_diagonal: bool = True,
        sampling_batch_size: int = 1,
        S_diagonal_init: float = 0.1,
        r_scale: float = 1.0,
    ):
        super().__init__()
        self.is_diagonal = is_diagonal
        self.dim = dim
        self.n_potentials = n_potentials
        self.register_buffer("epsilon", torch.tensor(epsilon))
        self.sampling_batch_size = sampling_batch_size

        self.log_alpha = nn.Parameter(torch.log(torch.ones(n_potentials) / n_potentials))
        self.r = nn.Parameter(torch.randn(n_potentials, dim))
        self.r_scale = nn.Parameter(torch.ones(n_potentials, 1))

        self.S_log_diagonal_matrix = nn.Parameter(
            torch.log(S_diagonal_init * torch.ones(n_potentials, dim))
        )
        if not is_diagonal:
            try:
                import geotorch
                self.S_rotation_matrix = nn.Parameter(
                    torch.randn(n_potentials, dim, dim)
                )
                geotorch.orthogonal(self, "S_rotation_matrix")
            except ImportError as e:
                raise ImportError("geotorch is required for is_diagonal=False") from e
        else:
            self.S_rotation_matrix = None

    def init_r_by_samples(self, samples: torch.Tensor) -> None:
        assert samples.shape[0] == self.r.shape[0]
        self.r.data = torch.clone(samples.to(self.r.device))

    def get_S(self) -> torch.Tensor:
        if self.is_diagonal:
            return torch.exp(self.S_log_diagonal_matrix)
        S_diag = torch.exp(self.S_log_diagonal_matrix)
        R = self.S_rotation_matrix
        return (R * S_diag[:, None, :]) @ R.permute(0, 2, 1)

    def get_r(self) -> torch.Tensor:
        return self.r

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Sample target z1 given source z0=x from the learned SB bridge."""
        S = self.get_S()
        r = self.get_r()
        epsilon = self.epsilon
        log_alpha = self.log_alpha

        samples = []
        B = x.shape[0]
        bs = self.sampling_batch_size
        n_iter = (B + bs - 1) // bs

        for i in range(n_iter):
            sub_x = x[bs * i : bs * (i + 1)]
            if self.is_diagonal:
                x_S_x = (sub_x[:, None, :] * S[None] * sub_x[:, None, :]).sum(-1)
                x_r = (sub_x[:, None, :] * r[None]).sum(-1)
                r_x = r[None] + S[None] * sub_x[:, None, :]
            else:
                x_S_x = (sub_x[:, None, None, :] @ (S[None] @ sub_x[:, None, :, None]))[:, :, 0, 0]
                x_r = (sub_x[:, None, :] * r[None]).sum(-1)
                r_x = r[None] + (S[None] @ sub_x[:, None, :, None])[:, :, :, 0]

            exp_arg = (x_S_x + 2 * x_r) / (2 * epsilon) + log_alpha[None]

            if self.is_diagonal:
                mix = Categorical(logits=exp_arg)
                comp = Independent(Normal(r_x, torch.sqrt(epsilon * S)[None].expand_as(r_x)), 1)
                gmm = MixtureSameFamily(mix, comp)
            else:
                mix = Categorical(logits=exp_arg)
                comp = MultivariateNormal(r_x, epsilon * S)
                gmm = MixtureSameFamily(mix, comp)

            samples.append(gmm.sample())

        return torch.cat(samples, dim=0)

    def get_drift(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Compute the bridge drift f(x_t, t) via autograd of the log-sum-exp potential."""
        x = torch.tensor(x.detach(), requires_grad=True, device=x.device)
        epsilon = self.epsilon
        r = self.get_r()

        S_diagonal = torch.exp(self.S_log_diagonal_matrix)  # (K, D)
        A_diagonal = (t / (epsilon * (1 - t)))[:, None, None] + 1.0 / (epsilon * S_diagonal)[None]
        # A_diagonal: (B, K, D)

        S_log_det = self.S_log_diagonal_matrix.sum(-1)  # (K,)
        A_log_det = torch.log(A_diagonal).sum(-1)       # (B, K)
        log_alpha = self.log_alpha                       # (K,)

        S_inv = 1.0 / S_diagonal  # (K, D)
        A_inv = 1.0 / A_diagonal  # (B, K, D)

        # c: (B, K, D)
        c = (
            (1.0 / (epsilon * (1 - t)))[:, None] * x
        )[:, None, :] + (r * S_inv / epsilon)[None]

        exp_arg = (
            log_alpha[None]
            - 0.5 * S_log_det[None]
            - 0.5 * A_log_det
            - 0.5 * ((r * S_inv * r) / epsilon).sum(-1)[None]
            + 0.5 * (c * A_inv * c).sum(-1)
        )  # (B, K)

        lse = torch.logsumexp(exp_arg, dim=-1)  # (B,)
        drift = (
            -x / (1 - t[:, None])
            + epsilon
            * torch.autograd.grad(
                lse, x, grad_outputs=torch.ones_like(lse), create_graph=True
            )[0]
        )
        return drift

    def sample_at_time_moment(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t = t.to(x.device)
        y = self(x)
        return t * y + (1 - t) * x + torch.sqrt(t * (1 - t) * self.epsilon) * torch.randn_like(x)

    def get_log_potential(self, x: torch.Tensor) -> torch.Tensor:
        S = self.get_S()
        r = self.get_r()
        mix = Categorical(logits=self.log_alpha)
        comp = Independent(Normal(r, torch.sqrt(self.epsilon * S)), 1)
        gmm = MixtureSameFamily(mix, comp)
        return gmm.log_prob(x) + torch.logsumexp(self.log_alpha, dim=-1)

    def get_log_C(self, x: torch.Tensor) -> torch.Tensor:
        S = self.get_S()
        r = self.get_r()
        epsilon = self.epsilon
        x_S_x = (x[:, None, :] * S[None] * x[:, None, :]).sum(-1)
        x_r = (x[:, None, :] * r[None]).sum(-1)
        exp_arg = (x_S_x + 2 * x_r) / (2 * epsilon) + self.log_alpha[None]
        return torch.logsumexp(exp_arg, dim=-1)

    def set_epsilon(self, new_epsilon: float) -> None:
        self.epsilon = torch.tensor(new_epsilon, device=self.epsilon.device)

    def sample_euler_maruyama(self, x: torch.Tensor, n_steps: int) -> torch.Tensor:
        """Generate full trajectory from x0 to x1 via Euler-Maruyama."""
        epsilon = self.epsilon
        t = torch.zeros(x.shape[0], device=x.device)
        dt = 1.0 / n_steps
        trajectory = [x]
        for _ in range(n_steps):
            x = (
                x
                + self.get_drift(x, t) * dt
                + math.sqrt(dt) * torch.sqrt(epsilon) * torch.randn_like(x)
            )
            t = t + dt
            trajectory.append(x)
        return torch.stack(trajectory, dim=1)


class ConditionedLightSBM(nn.Module):
    """
    Conditional extension of LightSBM for cell perturbation prediction.

    Supports a configurable number of condition labels (K types), each with its
    own embedding table. Per-sample embeddings are concatenated and fused before
    predicting offsets for bridge parameters.

    Args:
        dim:                latent space dimensionality D
        n_potentials:       number of GMM components K
        condition_label_order: ordered condition labels to embed (e.g.
                            ``["gene"]`` or ``["gene", "cell_line"]``)
        condition_vocab_sizes: mapping label -> vocabulary size
        n_conditions:       legacy single-label alias (equivalent to
                            ``condition_vocab_sizes={"gene": n_conditions}``)
        cond_embed_dim:     dimension of the condition embedding vector
        epsilon:            entropic regularisation ε
        sampling_batch_size: sub-batch size for the forward bridge sampler
        S_diagonal_init:    initial value of diagonal S entries
        cond_hidden_dim:    hidden size of the MLP that maps embedding → Δr / Δα
                            (set to None to use a single linear layer)
    """

    def __init__(
        self,
        dim: int,
        n_potentials: int,
        condition_label_order: list[str] | tuple[str, ...] | None = None,
        condition_vocab_sizes: dict[str, int] | None = None,
        n_conditions: int | None = None,
        cond_embed_dim: int = 64,
        epsilon: float = 1.0,
        sampling_batch_size: int = 64,
        S_diagonal_init: float = 0.1,
        cond_hidden_dim: int | None = None,
    ):
        super().__init__()
        self.dim = dim
        self.n_potentials = n_potentials
        self.register_buffer("epsilon", torch.tensor(epsilon))
        self.sampling_batch_size = sampling_batch_size

        if condition_vocab_sizes is None:
            if n_conditions is None:
                raise ValueError("Either condition_vocab_sizes or n_conditions must be provided.")
            condition_vocab_sizes = {"gene": n_conditions}
        if condition_label_order is None:
            condition_label_order = list(condition_vocab_sizes.keys())
        self.condition_label_order = tuple(condition_label_order)
        self.condition_vocab_sizes = {
            label: int(condition_vocab_sizes[label]) for label in self.condition_label_order
        }
        self.num_condition_labels = len(self.condition_label_order)
        if self.num_condition_labels == 0:
            raise ValueError("condition_label_order must include at least one label.")
        # Backward-compatible attribute name used in some tests/code paths.
        self.n_conditions = int(math.prod(self.condition_vocab_sizes.values()))

        # Shared base parameters
        self.base_log_alpha = nn.Parameter(
            torch.log(torch.ones(n_potentials) / n_potentials)
        )
        self.base_r = nn.Parameter(torch.randn(n_potentials, dim) * 0.01)
        self.S_log_diagonal_matrix = nn.Parameter(
            torch.log(S_diagonal_init * torch.ones(n_potentials, dim))
        )

        # One condition embedding table per label.
        self.condition_embeddings = nn.ModuleDict({
            label: nn.Embedding(vocab_size, cond_embed_dim)
            for label, vocab_size in self.condition_vocab_sizes.items()
        })
        for emb in self.condition_embeddings.values():
            nn.init.normal_(emb.weight, std=0.01)
        self.condition_fusion = nn.Linear(self.num_condition_labels * cond_embed_dim, cond_embed_dim)

        # Heads that produce per-sample offsets from the condition embedding
        if cond_hidden_dim is not None:
            self.cond_to_r = nn.Sequential(
                nn.Linear(cond_embed_dim, cond_hidden_dim),
                nn.SiLU(),
                nn.Linear(cond_hidden_dim, n_potentials * dim),
            )
            self.cond_to_log_alpha = nn.Sequential(
                nn.Linear(cond_embed_dim, cond_hidden_dim),
                nn.SiLU(),
                nn.Linear(cond_hidden_dim, n_potentials),
            )
        else:
            self.cond_to_r = nn.Linear(cond_embed_dim, n_potentials * dim)
            self.cond_to_log_alpha = nn.Linear(cond_embed_dim, n_potentials)

        # Initialise output heads to zero so the model starts as unconditional
        for layer in ([self.cond_to_r] if cond_hidden_dim is None else [self.cond_to_r[-1]]):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
        for layer in ([self.cond_to_log_alpha] if cond_hidden_dim is None else [self.cond_to_log_alpha[-1]]):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def init_r_by_samples(self, samples: torch.Tensor) -> None:
        """Initialise base_r from K representative samples (e.g. k-means centres)."""
        assert samples.shape == (self.n_potentials, self.dim), (
            f"Expected ({self.n_potentials}, {self.dim}), got {samples.shape}"
        )
        self.base_r.data = samples.to(self.base_r.device).clone()

    def _get_conditioned_params(
        self, condition_indices: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return per-sample effective parameters.

        Args:
            condition_indices: (B, K_labels) integer tensor of condition indices.
                               If K_labels==1, (B,) is also accepted.

        Returns:
            r_eff:        (B, K, D)
            log_alpha_eff:(B, K)
        """
        condition_indices = self._normalize_condition_indices(condition_indices)
        condition_embeds = []
        for idx, label in enumerate(self.condition_label_order):
            condition_embeds.append(self.condition_embeddings[label](condition_indices[:, idx]))
        e_c = self.condition_fusion(torch.cat(condition_embeds, dim=-1))
        delta_r = self.cond_to_r(e_c).view(-1, self.n_potentials, self.dim)  # (B, K, D)
        delta_log_alpha = self.cond_to_log_alpha(e_c)                         # (B, K)

        r_eff = self.base_r[None] + delta_r                                   # (B, K, D)
        log_alpha_eff = self.base_log_alpha[None] + delta_log_alpha           # (B, K)
        return r_eff, log_alpha_eff

    def _normalize_condition_indices(self, condition_indices: torch.Tensor) -> torch.Tensor:
        if condition_indices.dim() == 1:
            if self.num_condition_labels != 1:
                raise ValueError(
                    "Expected condition_indices with shape (B, K_labels) for multi-label conditioning."
                )
            condition_indices = condition_indices.unsqueeze(-1)
        if condition_indices.dim() != 2:
            raise ValueError(
                f"condition_indices must be rank-2 (B, K_labels), got shape={tuple(condition_indices.shape)}"
            )
        if condition_indices.shape[1] != self.num_condition_labels:
            raise ValueError(
                "condition_indices second dimension does not match condition_label_order: "
                f"got {condition_indices.shape[1]}, expected {self.num_condition_labels}."
            )
        return condition_indices.long()

    @torch.no_grad()
    def forward(self, x: torch.Tensor, condition_indices: torch.Tensor) -> torch.Tensor:
        """
        Sample target latent z1 ~ p(z1 | z0=x, condition) from the SB bridge.

        Args:
            x:             (B, D) source latent vectors
            condition_indices: (B, K_labels) condition indices

        Returns:
            samples: (B, D) sampled target latent vectors
        """
        S = torch.exp(self.S_log_diagonal_matrix)  # (K, D)
        epsilon = self.epsilon
        r_eff, log_alpha_eff = self._get_conditioned_params(condition_indices)
        # r_eff: (B, K, D), log_alpha_eff: (B, K)

        samples = []
        B = x.shape[0]
        bs = self.sampling_batch_size
        n_iter = (B + bs - 1) // bs

        for i in range(n_iter):
            sl = slice(bs * i, bs * (i + 1))
            sub_x = x[sl]               # (b, D)
            sub_r = r_eff[sl]           # (b, K, D)
            sub_la = log_alpha_eff[sl]  # (b, K)

            # x^T S x  →  (b, K)
            x_S_x = (sub_x[:, None, :] * S[None] * sub_x[:, None, :]).sum(-1)
            # x^T r_k   →  (b, K)
            x_r = (sub_x[:, None, :] * sub_r).sum(-1)
            # r_k + S * x  →  (b, K, D)
            r_x = sub_r + S[None] * sub_x[:, None, :]

            exp_arg = (x_S_x + 2 * x_r) / (2 * epsilon) + sub_la  # (b, K)

            mix = Categorical(logits=exp_arg)
            comp = Independent(
                Normal(r_x, torch.sqrt(epsilon * S)[None].expand_as(r_x)), 1
            )
            gmm = MixtureSameFamily(mix, comp)
            samples.append(gmm.sample())

        return torch.cat(samples, dim=0)

    def get_drift(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        condition_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the conditioned bridge drift f(x_t, t, c) via autograd.

        Args:
            x:             (B, D)  interpolated latent x_t
            t:             (B,)    interpolation times in [0, 1)
            condition_indices: (B, K_labels) condition indices

        Returns:
            drift: (B, D)
        """
        x = torch.tensor(x.detach(), requires_grad=True, device=x.device)

        epsilon = self.epsilon
        r_eff, log_alpha_eff = self._get_conditioned_params(condition_indices)
        # r_eff: (B, K, D),  log_alpha_eff: (B, K)

        S_diagonal = torch.exp(self.S_log_diagonal_matrix)  # (K, D)

        # A_diagonal: (B, K, D)
        A_diagonal = (
            (t / (epsilon * (1 - t)))[:, None, None]
            + 1.0 / (epsilon * S_diagonal)[None]
        )

        S_log_det = self.S_log_diagonal_matrix.sum(-1)  # (K,)
        A_log_det = torch.log(A_diagonal).sum(-1)       # (B, K)

        S_inv = 1.0 / S_diagonal  # (K, D)
        A_inv = 1.0 / A_diagonal  # (B, K, D)

        # c: (B, K, D)  —  uses per-sample r_eff
        c = (
            (1.0 / (epsilon * (1 - t)))[:, None, None] * x[:, None, :]
            + r_eff * S_inv[None] / epsilon
        )

        # r^T S^{-1} r / ε  →  (B, K)
        r_S_inv_r = (r_eff * S_inv[None] * r_eff).sum(-1) / epsilon

        exp_arg = (
            log_alpha_eff
            - 0.5 * S_log_det[None]
            - 0.5 * A_log_det
            - 0.5 * r_S_inv_r
            + 0.5 * (c * A_inv * c).sum(-1)
        )  # (B, K)

        lse = torch.logsumexp(exp_arg, dim=-1)  # (B,)
        drift = (
            -x / (1 - t[:, None])
            + epsilon
            * torch.autograd.grad(
                lse, x, grad_outputs=torch.ones_like(lse), create_graph=True
            )[0]
        )
        return drift

    def set_epsilon(self, new_epsilon: float) -> None:
        """Sync entropic regularisation with ``SchrodingerBridgeMatching.epsilon``."""
        self.epsilon = torch.tensor(new_epsilon, device=self.epsilon.device)

    def sample_at_time_moment(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        condition_indices: torch.Tensor,
    ) -> torch.Tensor:
        t = t.to(x.device)
        y = self(x, condition_indices)
        return t * y + (1 - t) * x + torch.sqrt(t * (1 - t) * self.epsilon) * torch.randn_like(x)

    def sample_euler_maruyama(
        self,
        x: torch.Tensor,
        condition_indices: torch.Tensor,
        n_steps: int = 100,
    ) -> torch.Tensor:
        """
        Generate trajectory from x0 → x1 via Euler-Maruyama integration.

        Returns a tensor of shape (B, n_steps+1, D) containing the full trajectory.
        The last time step (index n_steps) is the predicted perturbed latent.
        """
        epsilon = self.epsilon
        t = torch.zeros(x.shape[0], device=x.device)
        dt = 1.0 / n_steps
        trajectory = [x]
        for _ in range(n_steps):
            drift = self.get_drift(x, t, condition_indices)
            x = (
                x.detach()
                + drift.detach() * dt
                + math.sqrt(dt) * torch.sqrt(epsilon) * torch.randn_like(x)
            )
            t = (t + dt).clamp(max=1.0 - 1e-6)
            trajectory.append(x)
        return torch.stack(trajectory, dim=1)
