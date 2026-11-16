# Stage 3 (routing / sparsity) on top of the BEST Stage-2 result.
#
# Stage 2 is closed.  Everything tried on it came back indistinguishable:
#   additive reliability branch   full 58-case clean +0.37 / rgb_missing -0.48
#                                 against a +-2 single-run noise floor
#   +30 epochs (low LR)           robust_score +0.02 over 25 epochs
#   v2 +30 epochs (low LR)        60.51 / 59.70 / 60.26 against a 60.20 start
#   warm restart at 3e-5          destroyed the model (-1.61 at epoch 5)
# So v2 e60 stands: clean 62.01 / corrupted 59.69 / rgb_missing 55.92 /
# t_missing 57.99, and this arm starts from exactly that checkpoint.
#
# WHAT THIS ARM CHANGES, and nothing else:
#
#   SPARSITY  the UtilityRouter is switched on.  It is the one component that
#             has never received a gradient: DinoSharedViT builds it
#             unconditionally, so a random initialisation of it sits in every
#             Stage 2R checkpoint, and dense mode never calls it.  The extras
#             are pruned to target_k before the nine deep blocks, so the
#             student trains in the regime inference runs in.
#
#             The budget is SAMPLED PER STEP from a ladder
#             (0, 0.25, 0.5, 0.75, 1.0) x N rather than scheduled by epoch, so
#             one checkpoint serves the whole range at deployment instead of
#             being specialised to one rung.  1.0 must be in the ladder: it is
#             the only rung that gives the router full-support gradient, since
#             through topk_gather a pruned token's utility gets exactly zero
#             gradient (u reaches the graph only via topk().indices, which is
#             non-differentiable, and gather(u,idx)).  The first 5 epochs stay
#             dense so the random initialisation is not asked to prune before
#             it has any signal.
#
#             Evaluation always runs at target_k whatever the training budget
#             (DinoTSReliableSparse.extract_feat), so every validation measures
#             the deployment regime and the checkpoint selector never compares
#             a dense checkpoint against a sparse one.
#
#             The anchors are NEVER pruned -- anchor_map must reshape to
#             [B, D, H, W] with exactly N entries -- so target_k=600 gives a
#             sequence of 1200 + 600 = 1800 against the dense 2400: a 25%
#             reduction, not 50%.  Attention is quadratic in sequence length,
#             so the attention term falls 43.75%, but the full-model latency,
#             FLOPs and memory have to be measured, not inferred.
#
#   FREEZING  the R=3 shallow blocks and their per-modality adapters, plus
#             ReliabilityHead.  Freezing the head is the load-bearing part: it
#             makes q a FIXED function of the input, so the router's input is
#             stationary and the compression objective cannot redefine what
#             "reliable" means.  The cost is that q's calibration is locked to
#             what Stage 2 produced -- which is the point, since Stage 2 is
#             closed and this arm must not be able to drift it.
#
# This is the arm that answers the question Stage 2 could not: not "can the
# accuracy go higher" (it cannot) but "how much accuracy does a 25% shorter
# deep-block sequence cost".
#
# NOT additive: backbone.reliable_fusion keeps v2's concatenated descriptor
# (4D+2).  router_rel_dim=4 is set because the router's reliability input is
# the whole point of the routing arm; its rel_proj is zero-initialised, so the
# router starts reliability-blind and has to learn to use it.
_base_ = ['stage2r_reliable_mfnet.py']

load_from = '{{$V2_E60:/root/autodl-tmp/code/QWSEG/work_dirs/dino_ts/stage2r_reliable_v2/weight/stage2r_reliable_v2/best_robust_score_epoch_60.pth}}'

model = dict(
    type='DinoTSReliableSparse',
    backbone=dict(router_rel_dim=4),
    forward_mode='sparse_hard',
    # The EVALUATION budget.  Training samples from budget_ladder below; this
    # is the single rung every validation and every checkpoint-selection
    # decision is made at, so the selected checkpoint is the one that is best
    # in the middle of the range rather than best at an extreme.  The full
    # accuracy-vs-budget curve needs a separate sweep over the rungs.
    target_k=600,
    # PER-STEP SAMPLING: each training step draws one rung at random, so the
    # model sees every budget on every epoch instead of living in one phase at
    # a time.  1.0 (K = N, dense) MUST be in the ladder -- it is the only rung
    # that gives the router full-support gradient, because through
    # topk_gather a pruned token's utility receives exactly zero gradient.
    # 0.0 is the true floor: no extras at all, deep sequence = the 1200
    # anchors only, half the dense length.  It is not a degenerate model --
    # the cross-modal InteractionResidual sits INSIDE the anchor
    # (anchors = alpha_r*r + alpha_t*t + gate*delta), so the interaction
    # survives even when every extra is gone.
    budget_ladder=(0.0, 0.25, 0.5, 0.75, 1.0),
    # The first few epochs stay dense regardless: at step 0 the router is a
    # random initialisation and pruning half the extras at random before it
    # has any signal is pure noise.
    budget_warmup_epochs=5,
    soft_tau=1.0,
    freeze_shallow=True,
    freeze_reliability=True,
    # False -> train with the real Top-K gather, the exact deployment regime,
    # at the cost of top-k's gradient hole on the pruned positions.  True ->
    # soft_gate(hard=True): same 0/1 forward but the sequence keeps its full
    # length, so every token gets gradient.
    gate_hard=False)
