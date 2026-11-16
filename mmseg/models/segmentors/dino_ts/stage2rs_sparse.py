"""Stage 2R with token sparsity switched on (the UtilityRouter is activated).

DinoTSReliable trains on the full N + N sequence -- ``forward_mode='dense'`` --
so the UtilityRouter, which DinoSharedViT constructs unconditionally and which
is therefore present in every Stage 2R checkpoint, is never called and never
receives a gradient.  This subclass is the SAME objective with routing on:
identical teachers, views, losses, schedule, seed and warm start.  The only
difference is that the extras are pruned before the nine deep blocks, so the
student trains in the regime inference actually runs in.

It is meant to be stage B of a two-stage run: stage A trains the same objective
dense (configs/dino_ts/stage2r_additive_mfnet.py), and this stage continues
from its checkpoint with the front end frozen, so the only new variable is the
sparsity itself.

Budget
------
``budget_schedule`` ramps the number of kept extras down to ``target_k`` over
``budget_warmup_epochs``.  Training sparse from epoch 0 with a randomly
initialised router would prune half the extras AT RANDOM at step 0 and keep
moving the kept set afterwards.  More importantly, through ``topk_gather`` a
PRUNED token's utility receives exactly zero gradient -- u reaches the graph
only via ``topk().indices`` (non-differentiable) and ``gather(u, idx)`` -- so
K=N, where the gather is a permutation and every extra is in idx, is the only
phase in which the router can learn a full ranking at all.

Evaluation always runs at ``target_k`` whatever the training budget, so every
validation point measures the deployment regime and the checkpoint selector
never compares a dense checkpoint against a sparse one.

Freezing
--------
``freeze_shallow`` stops gradients on the R shallow blocks and their adapters;
``freeze_reliability`` stops them on ReliabilityHead.  Freezing the head is
what makes q a FIXED function of the input, which keeps the router's input
stationary and stops the compression objective from redefining what
"reliable" means -- at the cost of q's calibration being locked to whatever
stage A produced.
"""
from typing import Optional

import torch

from mmseg.registry import MODELS
from .stage2r_reliable import DinoTSReliable


@MODELS.register_module()
class DinoTSReliableSparse(DinoTSReliable):

    def __init__(self, *args,
                 target_k: Optional[int] = None,
                 budget_schedule=None,
                 budget_ladder=None,
                 budget_warmup_epochs: int = 0,
                 gate_hard: bool = False,
                 freeze_shallow: bool = False,
                 freeze_reliability: bool = False,
                 **kwargs):
        # Eval mode.  DinoTSReliable's own setdefault('forward_mode', 'dense')
        # cannot override this because we set the key before calling super().
        kwargs.setdefault('forward_mode', 'sparse_hard')
        kwargs['target_k'] = target_k
        super().__init__(*args, **kwargs)
        self.target_k = target_k
        self.budget_schedule = list(budget_schedule or [])
        self.budget_ladder = [float(x) for x in (budget_ladder or [])]
        self.budget_warmup_epochs = int(budget_warmup_epochs)
        if any(not 0.0 <= x <= 1.0 for x in self.budget_ladder):
            raise ValueError('budget_ladder entries are fractions of N, in '
                             '[0, 1]')
        self.gate_hard = bool(gate_hard)
        self.freeze_shallow = bool(freeze_shallow)
        self.freeze_reliability = bool(freeze_reliability)
        if self.budget_schedule and self.budget_warmup_epochs <= 0:
            raise ValueError('budget_schedule needs budget_warmup_epochs > 0')
        self._apply_freeze()

    # -- freezing ----------------------------------------------------------
    def _apply_freeze(self):
        """Freeze the front end, and report exactly what that left trainable.

        Nothing else re-enables requires_grad: DinoTSReliable.init_weights only
        loads the frozen teachers and never touches the student's parameters,
        so freezing here is stable for the whole run.
        """
        bb = self.backbone
        frozen = []
        if self.freeze_shallow:
            for layer in range(bb.R):
                for p in bb.blocks[layer].parameters():
                    p.requires_grad = False
                frozen.append('blocks[%d]' % layer)
            for modality in bb.adapters:
                for layer, adapter in enumerate(bb.adapters[modality]):
                    if layer < bb.R:
                        for p in adapter.parameters():
                            p.requires_grad = False
            frozen.append('adapters[0..%d]' % (bb.R - 1))
        if self.freeze_reliability:
            for p in bb.fusion.reliability.parameters():
                p.requires_grad = False
            frozen.append('ReliabilityHead')
        total = sum(p.numel() for p in self.parameters())
        train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        self._freeze_report = dict(frozen=frozen, total=total, trainable=train)
        if frozen:
            print('[DinoTSReliableSparse] frozen: %s | trainable %.2fM / %.2fM '
                  'params (%.1f%% frozen)'
                  % (', '.join(frozen), train / 1e6, total / 1e6,
                     100.0 * (total - train) / max(total, 1)), flush=True)

    # -- budget ------------------------------------------------------------
    def _target_k(self, n: int) -> int:
        return n // 2 if self.target_k is None else self.target_k

    def _current_k(self, n: int) -> int:
        """Budget for this epoch: ramps down to target_k over the warmup.

        With no schedule this is just target_k.  A schedule turns it into
        multi-budget training, so one checkpoint serves several budgets at
        deployment; the kept sets are nested for free because every rung is a
        top-k over the same utility.
        """
        if self.target_k is None:
            return n // 2
        stages = self.budget_schedule
        if not stages or self.current_epoch >= self.budget_warmup_epochs:
            return self.target_k
        step = self.budget_warmup_epochs / (len(stages) + 1)
        idx = min(int(self.current_epoch / max(step, 1)), len(stages) - 1)
        return max(self.target_k, int(round(stages[idx] * n)))

    def _budget_for_step(self, n: int) -> int:
        """How many extras this step keeps.

        PER-STEP SAMPLING (budget_ladder) is the default when set, rather than
        the epoch curriculum, because it is strictly more informative: the
        model sees every budget on every epoch instead of living in one phase
        at a time, so a single checkpoint serves the whole range at deployment.

        The ladder must contain 1.0 (K = N).  Through topk_gather a PRUNED
        token's utility gets exactly zero gradient -- u reaches the graph only
        via topk().indices (non-differentiable) and gather(u, idx) -- so K = N,
        where the gather is a permutation and every extra is in idx, is the
        ONLY rung that gives the router full-support gradient.  Sampling it
        ~1/len(ladder) of the time is enough for it to learn a ranking.

        The first `budget_warmup_epochs` epochs stay dense regardless: at step
        0 the router is a random initialisation, and pruning half the extras at
        random before it has any signal is pure noise.

        Evaluation never comes through here -- extract_feat pins it to
        target_k so every validation measures the deployment regime.
        """
        if self.current_epoch < self.budget_warmup_epochs:
            return n
        if self.budget_ladder:
            idx = int(torch.randint(len(self.budget_ladder), (1,)).item())
            return max(0, min(n, int(round(self.budget_ladder[idx] * n))))
        return self._current_k(n)

    # -- forward -----------------------------------------------------------
    def extract_feat(self, inputs, mode=None, availability=None,
                     gate_hard=None):
        """Same as BaseDinoTS.extract_feat, plus the budget and the gate mode."""
        rgb, thermal = self._split(inputs)
        if mode is None:
            mode = self.forward_mode
        n = self.backbone.grid[0] * self.backbone.grid[1]
        if mode.startswith('sparse'):
            # Training ramps the budget down; EVALUATION is always at the
            # deployment budget.  Otherwise the epoch-5 validation would score
            # a still-dense model, and the checkpoint selector would be
            # comparing dense checkpoints against sparse ones.
            k = (self._budget_for_step(n) if self.training
                 else self._target_k(n))
        else:
            k = self.target_k
        out = self.backbone(
            rgb, thermal, mode=mode, target_k=k, soft_tau=self.soft_tau,
            gate_hard=(self.gate_hard if gate_hard is None else gate_hard),
            availability=availability)
        self._last_seq_len = out['seq_len']
        self._last_backbone_out = out
        return self._anchor_to_pyramid(out['anchor_map'])
