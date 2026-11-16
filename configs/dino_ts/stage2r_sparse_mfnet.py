# Stage 2R (the v2/e60 method) + a budget curriculum + the additive
# reliability branch.  Nothing else changes: same two single-modality
# teachers, same unimodal KD, same accuracy/normalised reliability targets,
# same views (0.6/0.2/0.2), same seed, same 60-epoch schedule, same Stage-2A
# warm start.
#
# Change 1 -- sparsity, WITH A CURRICULUM.  DinoTSReliable trains on the full
#   N + N sequence (forward_mode='dense'), so the UtilityRouter that
#   DinoSharedViT builds unconditionally is never called and has never
#   received a gradient -- it is a random initialisation sitting in every
#   Stage 2R checkpoint.  Switching it on at full strength from epoch 0 would
#   prune half the extras AT RANDOM at step 0, hand the nine deep blocks a
#   randomly-halved sequence, and then keep moving the kept SET as the router
#   changes.  So the budget ramps instead:
#
#       epochs  0-7   K=1200  = dense, nothing thrown away
#       epochs  8-14  K=900
#       epochs 15-60  K=600   = the deployment budget
#
#   The dense phase is not just a warm-up for the student -- it is the only
#   phase in which the router can learn a full ranking at all.  Through
#   topk_gather a PRUNED token's utility receives exactly zero gradient (u
#   reaches the graph only via topk().indices, which is non-differentiable,
#   and gather(u, idx)); at K=N the gather is a permutation, so every one of
#   the N extras is in idx and every one gets gradient.  Ramp first, prune
#   later.
#
#   Evaluation always runs at target_k regardless of the training budget (see
#   DinoTSReliableSparse.extract_feat), so every validation point measures the
#   deployment regime and the checkpoint selector never compares a dense
#   checkpoint against a sparse one.
#
#   NOTE the anchors are never pruned -- anchor_map must reshape to
#   [B, D, H, W] with exactly N entries -- so target_k=600 gives a sequence of
#   1200 + 600 = 1800 at deployment, a 25% reduction, not 50%.
#
# Change 2 -- the reliability channels get their own projection.  In v2 they
#   were two columns of a 4D+2 = 3074-wide descriptor, i.e. 0.065% of the
#   first Linear's pre-activation variance, and their weights would have had
#   to travel ~1.6 to matter.  Adam moves a weight at most lr per step and
#   this schedule's PolyLR budget is ~0.55 over the whole run, so they could
#   not get there.  additive_rel=True gives them a Linear(4, h) of their own,
#   zero-initialised, added AFTER the GELU (adding before it would multiply
#   this branch's gradient by GELU'(z), which vanishes exactly where the token
#   path saturates).  Zero-init makes step 0 identical to the reliability-blind
#   fusion, so the norm of those weights is a free diagnostic of how much the
#   branch ended up being used.
_base_ = ['stage2r_reliable_mfnet.py']

model = dict(
    type='DinoTSReliableSparse',
    backbone=dict(
        router_rel_dim=4,
        reliable_fusion=dict(additive_rel=True)),
    forward_mode='sparse_hard',
    target_k=600,
    budget_schedule=(1.0, 0.75, 0.5),
    budget_warmup_epochs=30,
    soft_tau=1.0,
    # False -> train with the real Top-K gather, the exact deployment regime,
    # at the cost of top-k's gradient hole on the pruned positions.  True ->
    # soft_gate(hard=True): same 0/1 forward but the sequence keeps its full
    # length, so every token gets gradient.
    gate_hard=False)
