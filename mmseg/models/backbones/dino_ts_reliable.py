"""Reliability-guided fusion for DINO-TS.

New, self-contained modules.  Nothing here is wired into the backbone yet --
`dino_shared_vit.py` keeps using `AnchorExtraFusion` until the call site is
switched over, so adding this file cannot disturb a running job.

Why this file exists
--------------------
Measured on the Stage-2C EMA weights (2026-10-10):

    thermal_only forward (fusion bypassed, pure thermal)   51.67 mIoU
    rgb_missing  (RGB blacked out, fusion still runs)      52.12
    dedicated Thermal-only model                           57.67
    rgb_only forward (fusion bypassed, pure RGB)           56.93
    t_missing    (thermal blacked out, fusion still runs)  57.33
    dedicated RGB-only model                               59.71

Bypassing the fusion is 0.4 mIoU WORSE, so the fusion is not corrupting the
features.  The 6.00 / 2.78 gap is in the weights: the shared trunk and the
adapters were never trained to operate on a single modality.  Capability comes
from supervision, not from routing -- so these modules are the *enabling* half
of the fix, and the explicit missing-modality views + single-modality
distillation are the load-bearing half.

What the reliability weighting does buy: with a convex mixture
(alpha_rgb + alpha_t = 1), q_rgb -> 0 makes the anchor EXACTLY the thermal
token -- a clean single-modality representation, which is the distribution the
Thermal-only teacher was trained on.  Without it, a missing view produces a
mixture of real thermal and black RGB, and the distillation signal lands on a
representation that matches nothing.
"""
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ReliabilityHead(nn.Module):
    """Per-token reliability q in [0, 1] for ONE modality.

    Reads only that modality's own shallow tokens plus a modality embedding --
    never the other modality.  That is the point: a modality that has been
    deleted must not be able to borrow the other sensor's evidence and score
    itself reliable.

    Structure follows the design note: LayerNorm -> 1x1 down-projection ->
    3x3 depthwise (local context) -> concatenate this modality's global
    statistic -> small MLP with the modality embedding -> sigmoid.

    The output layer is zero-initialised, so q starts at exactly 0.5 for every
    token: the fusion then begins at an even mixture and has to learn its way
    out, rather than starting from an arbitrary prior.
    """

    def __init__(self, dim: int, hidden: int = 128, n_mod: int = 2,
                 kernel: int = 3):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Conv2d(dim, hidden, kernel_size=1)
        self.depthwise = nn.Conv2d(hidden, hidden, kernel_size=kernel,
                                   padding=kernel // 2, groups=hidden)
        self.mod_embed = nn.Parameter(torch.zeros(n_mod, hidden))
        self.mlp = nn.Sequential(
            nn.Linear(hidden * 2, hidden), nn.GELU(), nn.Linear(hidden, 1))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x: torch.Tensor, grid: Tuple[int, int],
                mod_id: int) -> torch.Tensor:
        """x: [B, N, D] shallow tokens of ONE modality, N == H*W.

        Returns q: [B, N] in [0, 1].
        """
        B, N, D = x.shape
        H, W = grid
        if N != H * W:
            raise ValueError(
                f'token count {N} does not match grid {H}x{W}={H * W}')

        h = self.norm(x).transpose(1, 2).reshape(B, D, H, W)
        h = self.proj(h)
        h = self.depthwise(h) + self.mod_embed[mod_id].view(1, -1, 1, 1)
        # This modality's own global statistic -- cheap, and it gives the head
        # a whole-image cue that a pure 3x3 receptive field cannot see.
        stat = h.mean(dim=(2, 3), keepdim=True).expand_as(h)
        h = torch.cat([h, stat], dim=1).permute(0, 2, 3, 1).reshape(B, N, -1)
        return torch.sigmoid(self.mlp(h)).squeeze(-1)


class PairScorer(nn.Module):
    """Bounded per-modality correction to the log-reliability prior.

    The reliability head deliberately sees one modality at a time, so it cannot
    know how the OTHER modality changes the decision.  This is where that is
    handled: it reads the pairwise descriptor and adds a bounded term to each
    modality's log-reliability before the softmax.

    Output layer zero-initialised so the fusion starts at the pure reliability
    prior.
    """

    def __init__(self, dim: int, hidden: Optional[int] = None,
                 in_dim: Optional[int] = None, rel_dim: int = 0):
        super().__init__()
        h = hidden or dim
        d_in = dim * 4 + 2 if in_dim is None else int(in_dim)
        self.mlp = nn.Sequential(
            nn.LayerNorm(d_in), nn.Linear(d_in, h), nn.GELU(),
            nn.Linear(h, 2))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
        self.rel_dim = int(rel_dim)
        if self.rel_dim:
            self.rel_proj = nn.Linear(self.rel_dim, h)
            nn.init.zeros_(self.rel_proj.weight)
            nn.init.zeros_(self.rel_proj.bias)

    def forward(self, phi: torch.Tensor,
                rel: Optional[torch.Tensor] = None) -> torch.Tensor:
        # rel is added AFTER the GELU: adding before it would multiply this
        # branch's gradient by GELU'(z), which vanishes wherever the token
        # path saturates.
        h = self.mlp[2](self.mlp[:2](phi))
        if self.rel_dim:
            if rel is None:
                raise ValueError(
                    f'PairScorer was built with rel_dim={self.rel_dim} but '
                    f'forward() got rel=None')
            h = h + self.rel_proj(rel)
        return self.mlp[3](h)  # [B, N, 2]


class InteractionResidual(nn.Module):
    """Delta_i: the cross-modal term a per-token weighting cannot express.

    Weighting can only pick a point on the segment between r and t.  A feature
    that exists in neither modality alone but is implied by the pair (the usual
    reason to do RGB-T at all) is off that segment, so it needs an additive
    branch.

    Down-project -> 3x3 depthwise (local) -> GELU -> project back to `dim`.
    Last layer zero-initialised, so the model starts exactly at the weighted
    main path and grows the interaction term only if supervision asks for it.
    """

    def __init__(self, dim: int, hidden: int = 128, kernel: int = 3,
                 in_dim: Optional[int] = None, rel_dim: int = 0):
        super().__init__()
        d_in = dim * 4 + 2 if in_dim is None else int(in_dim)
        self.norm = nn.LayerNorm(d_in)
        self.down = nn.Linear(d_in, hidden)
        self.depthwise = nn.Conv2d(hidden, hidden, kernel_size=kernel,
                                   padding=kernel // 2, groups=hidden)
        self.up = nn.Linear(hidden, dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)
        self.rel_dim = int(rel_dim)
        if self.rel_dim:
            self.rel_proj = nn.Linear(self.rel_dim, hidden)
            nn.init.zeros_(self.rel_proj.weight)
            nn.init.zeros_(self.rel_proj.bias)

    def forward(self, phi: torch.Tensor, grid: Tuple[int, int],
                rel: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, N, _ = phi.shape
        H, W = grid
        h = F.gelu(self.down(self.norm(phi)))
        if self.rel_dim:
            if rel is None:
                raise ValueError(
                    f'InteractionResidual was built with rel_dim='
                    f'{self.rel_dim} but forward() got rel=None')
            # After the GELU, so this branch's gradient does not inherit
            # GELU'(z) and die where the token path saturates.
            h = h + self.rel_proj(rel)
        h = h.transpose(1, 2).reshape(B, -1, H, W)
        h = F.gelu(self.depthwise(h))
        h = h.reshape(B, -1, N).transpose(1, 2)
        return self.up(h)  # [B, N, D]


class ChannelGate(nn.Module):
    """Per-token, per-CHANNEL gate conditioned on the reliability prior.

    g = sigmoid( f([x, q]) + logit(q) )        x: [B,N,D], q: [B,N] -> [B,N,D]

    Two properties are load-bearing:

    1. ``sigmoid(logit(q)) == q`` and the mlp's last layer is zero-initialised,
       so at step 0 the gate is EXACTLY the reliability that Stages 2A-2R were
       trained with.  Turning the option on cannot make a trained checkpoint
       behave differently until the mlp moves.
    2. q -> 0 sends logit(q) -> -inf, so a missing modality is gated to zero no
       matter what the mlp outputs.  The whole-modality-missing behaviour that
       Stage 2R establishes is therefore preserved by construction.

    Unlike the convex alpha (one scalar per token shared by all D channels),
    this gate can suppress the channels a corruption occupies and keep the
    channels that still carry signal.
    """

    def __init__(self, dim: int, hidden: int = 32, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.norm = nn.LayerNorm(dim + 1)
        self.mlp = nn.Sequential(nn.Linear(dim + 1, hidden), nn.GELU(),
                                 nn.Linear(hidden, dim))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x, q):
        """g = q * clamp(2*sigmoid(mlp([x,q])), max=1).

        Multiplicative suppression rather than an additive logit offset, for
        two structural guarantees the additive form could not give:

        * mlp == 0  ->  2*sigmoid(0) == 1  ->  g == q exactly at init.
        * q == 0    ->  g == 0 for ANY mlp output, so a missing modality can
          never be re-admitted by the gate.  With the additive form a large
          positive mlp output lifts sigmoid(logit(q) + h) off zero (measured
          0.13 at q=0.001), which would put the whole-modality-missing result
          back at the mercy of training.

        The gate can only SUPPRESS a channel relative to the reliability prior,
        never boost it: a low-q token is exactly the one that must not be
        trusted more than q says.
        """
        qc = q.clamp(self.eps, 1.0 - self.eps).unsqueeze(-1)
        h = self.mlp(self.norm(torch.cat([x, qc], dim=-1)))
        m = (2.0 * torch.sigmoid(h)).clamp(max=1.0)
        return qc * m


class ReliabilityFusion(nn.Module):
    """Anchor/Extra construction from the two shallow streams.

    Replaces AnchorExtraFusion.  Two things are load-bearing:

    1. The main path is a CONVEX mixture of the raw tokens,
       ``a0 = alpha_rgb * r + alpha_t * t`` with ``alpha_rgb + alpha_t = 1``.
       Because the weights come from ``softmax([log q_rgb + b_rgb,
       log q_t + b_t])``, q_rgb -> 0 drives alpha_rgb -> 0 and makes
       ``a0 == t`` exactly.  That is what lets a whole-modality-missing view
       produce a clean single-modality anchor for the distillation target to
       land on.

    2. No positional embeddings.  The previous fusion added a learned
       absolute ``extra_pos_embed`` (and the backbone added ``anchor_pos_embed``)
       which tied the model to one input resolution; RoPE is applied by the
       blocks and already carries position, so those tables were redundant.

    `gate_mode` selects how the interaction residual is gated, and is meant to
    be ablated rather than assumed:
      'availability' (default) -- shut the interaction off only when a modality
          is genuinely absent (q below `absent_thr`).  'Weak' must not disable
          it: two individually mediocre modalities that are jointly informative
          is exactly where cross-modal interaction earns its keep.
      'min' -- the design note's ``min(q_rgb, q_t)``.  Kept for the ablation;
          it conflates "absent" with "weak".
    """

    def __init__(self, dim: int, n_mod: int = 2, tau: float = 1.0,
                 gate_mode: str = 'availability', absent_thr: float = 0.1,
                 head_hidden: int = 128, additive_rel: bool = False,
                 rel_dim: int = 4, channel_gate: bool = False,
                 gate_hidden: int = 32):
        super().__init__()
        self.channel_gate = bool(channel_gate)
        if self.channel_gate:
            self.chan_r = ChannelGate(dim, hidden=gate_hidden)
            self.chan_t = ChannelGate(dim, hidden=gate_hidden)
        if gate_mode not in ('availability', 'min'):
            raise ValueError(f'unknown gate_mode {gate_mode!r}')
        self.dim = dim
        self.tau = float(tau)
        self.gate_mode = gate_mode
        self.absent_thr = float(absent_thr)
        # additive_rel=False reproduces the original fusion exactly, module for
        # module and key for key, so the frozen v2 config is untouched.
        self.additive_rel = bool(additive_rel)
        self.rel_dim = int(rel_dim) if self.additive_rel else 0
        d_in = dim * 4 if self.additive_rel else dim * 4 + 2

        self.reliability = ReliabilityHead(dim, hidden=head_hidden, n_mod=n_mod)
        self.pair_scorer = PairScorer(dim, in_dim=d_in, rel_dim=self.rel_dim)
        self.interaction = InteractionResidual(dim, hidden=head_hidden,
                                               in_dim=d_in,
                                               rel_dim=self.rel_dim)
        self.gate_mlp = nn.Sequential(
            nn.LayerNorm(d_in), nn.Linear(d_in, head_hidden),
            nn.GELU(), nn.Linear(head_hidden, 1))
        self.extra = nn.Sequential(
            nn.LayerNorm(d_in), nn.Linear(d_in, dim), nn.GELU(),
            nn.Linear(dim, dim))
        self.extra_type_embed = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.zeros_(self.extra[-1].weight)
        nn.init.zeros_(self.extra[-1].bias)
        # Created LAST so the v2 module-creation order (and therefore the RNG
        # stream that seeds everything after this module) is unchanged.
        if self.rel_dim:
            self.gate_rel = nn.Linear(self.rel_dim, head_hidden)
            self.extra_rel = nn.Linear(self.rel_dim, dim)
            for layer in (self.gate_rel, self.extra_rel):
                nn.init.zeros_(layer.weight)
                nn.init.zeros_(layer.bias)

    @staticmethod
    def _descriptor(r, t, q_rgb=None, q_t=None):
        base = [r, t, (r - t).abs(), r * t]
        if q_rgb is None:
            return torch.cat(base, dim=-1)
        return torch.cat(base + [q_rgb.unsqueeze(-1), q_t.unsqueeze(-1)],
                         dim=-1)

    @staticmethod
    def reliability_features(q_rgb: torch.Tensor, q_t: torch.Tensor,
                             alpha: Optional[torch.Tensor] = None
                             ) -> torch.Tensor:
        """[B, N, 4] reliability summary, every channel roughly in [-1, 1].

        Channels: 4*a_rgb*a_t (how BALANCED the mixture is), tanh(log-odds / 2)
        (which modality the mixture favours), and the two raw reliabilities.

        Balance carries the redundancy information an extra token is worth: an
        extra is worth keeping exactly where both modalities are trustworthy
        AND disagree, and 4*a_rgb*a_t is 0 at either extreme of the mixture.
        tanh bounds the log-odds, which are otherwise unbounded (alpha is a
        softmax of log q) and would dominate the channel.

        alpha=None uses the pure reliability prior softmax(log q), so this is
        computable BEFORE the pair scorer -- which is what lets the scorer see
        the reliability channels without a cycle.  The scale is deliberately
        O(1): the additive branch consuming this has fan-in 4 and is
        zero-initialised, so its weights need to travel only ~0.3 to match the
        token branch, against the ~0.55 an Adam schedule at this learning rate
        can move a weight over the whole run.
        """
        eps = 1e-6
        if alpha is None:
            alpha = F.softmax(
                torch.stack([torch.log(q_rgb + eps), torch.log(q_t + eps)],
                            dim=-1), dim=-1)
        a_r = alpha[..., 0].clamp_min(eps)
        a_t = alpha[..., 1].clamp_min(eps)
        balance = 4.0 * a_r * a_t
        direction = torch.tanh(0.5 * (torch.log(a_r) - torch.log(a_t)))
        return torch.stack([balance, direction, q_rgb, q_t], dim=-1)

    def _extra_logits(self, phi, rel=None):
        # GELU first, then the reliability offset: see PairScorer.forward.
        h = self.extra[2](self.extra[:2](phi))
        if self.rel_dim:
            h = h + self.extra_rel(rel)
        return self.extra[3](h)

    def _fuse_main(self, r, t, alpha, q_rgb, q_t):
        """Convex mixture (default) or independent per-channel gating."""
        if not self.channel_gate:
            return alpha[..., 0:1] * r + alpha[..., 1:2] * t
        return self.chan_r(r, q_rgb) * r + self.chan_t(t, q_t) * t

    def _gate_logit(self, phi, rel=None):
        h = self.gate_mlp[2](self.gate_mlp[:2](phi))
        if self.rel_dim:
            h = h + self.gate_rel(rel)
        return self.gate_mlp[3](h)

    def forward(self, r: torch.Tensor, t: torch.Tensor,
                grid: Tuple[int, int],
                q_rgb: Optional[torch.Tensor] = None,
                q_t: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                           torch.Tensor, torch.Tensor, torch.Tensor]:
        """r, t: [B, N, D] shallow RGB / thermal tokens at the fusion point.

        q_rgb / q_t may be supplied (e.g. to reuse a frozen head); otherwise
        they are computed here.  NOTE: callers training the fusion should pass
        DETACHED q so the fusion cannot improve its loss by editing the
        reliability scores -- q is trained by the reliability loss alone.

        Returns (anchors, extras, alpha, q_rgb, q_t, gate).
        """
        if q_rgb is None:
            q_rgb = self.reliability(r, grid, 0)
        if q_t is None:
            q_t = self.reliability(t, grid, 1)

        # Gradient isolation.  q is trained by the reliability loss ALONE; the
        # fusion must not be able to lower its own loss by editing q, or
        # "reliability" degenerates into whatever the segmentation objective
        # finds convenient.  So every use of q inside the fusion is detached.
        # The live q is still returned so the caller can supervise it.
        # (Detaching at the call site is not enough -- a caller-supplied q that
        # requires grad still picks up gradient from phi, the routing logits and
        # the gate unless it is detached here.)
        q_rgb_d, q_t_d = q_rgb.detach(), q_t.detach()
        eps = 1e-6

        if self.additive_rel:
            # Reliability travels on its own projection instead of being two
            # columns of a 4D+2-wide concat; see UtilityRouter's docstring for
            # why the concatenated version cannot work at this learning rate.
            phi = self._descriptor(r, t)
            rel_pre = self.reliability_features(q_rgb_d, q_t_d)
            bias = self.pair_scorer(phi, rel_pre)          # [B, N, 2]
            logits = torch.stack(
                [torch.log(q_rgb_d + eps) + bias[..., 0],
                 torch.log(q_t_d + eps) + bias[..., 1]], dim=-1)
            alpha = F.softmax(logits / self.tau, dim=-1)
            rel = self.reliability_features(q_rgb_d, q_t_d, alpha)
            main = self._fuse_main(r, t, alpha, q_rgb_d, q_t_d)
            delta = self.interaction(phi, grid, rel)
            gate = self._gate(q_rgb_d, q_t_d, phi, rel)
            anchors = main + gate * delta
            extras = ((r - t) + self._extra_logits(phi, rel)
                      + self.extra_type_embed)
            return anchors, extras, alpha, q_rgb, q_t, gate

        phi = self._descriptor(r, t, q_rgb_d, q_t_d)
        bias = self.pair_scorer(phi)                       # [B, N, 2]
        logits = torch.stack(
            [torch.log(q_rgb_d + eps) + bias[..., 0],
             torch.log(q_t_d + eps) + bias[..., 1]], dim=-1)
        alpha = F.softmax(logits / self.tau, dim=-1)       # [B, N, 2], sums to 1

        main = self._fuse_main(r, t, alpha, q_rgb_d, q_t_d)   # convex: q_rgb->0 => t
        delta = self.interaction(phi, grid)
        gate = self._gate(q_rgb_d, q_t_d, phi)
        anchors = main + gate * delta

        extras = (r - t) + self.extra(phi) + self.extra_type_embed
        return anchors, extras, alpha, q_rgb, q_t, gate

    def _gate(self, q_rgb, q_t, phi, rel=None):
        learned = torch.sigmoid(self._gate_logit(phi, rel))   # [B, N, 1]
        if self.gate_mode == 'min':
            return learned * torch.minimum(q_rgb, q_t).unsqueeze(-1)
        present = ((q_rgb > self.absent_thr) & (q_t > self.absent_thr))
        return learned * present.unsqueeze(-1).to(learned.dtype)
