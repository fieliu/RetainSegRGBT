# Stage 2R stage B: sparsity on top of stage A, with the front end frozen.
#
# Continues from stage A's checkpoint (the v2/e60 objective + the additive
# reliability branch).  Two things change, and nothing else:
#
#   FREEZE  the R=3 shallow blocks and their per-modality adapters, plus
#           ReliabilityHead.  Freezing the head is the load-bearing part: it
#           makes q a FIXED function of the input, so the router's input is
#           stationary and the compression objective cannot redefine what
#           "reliable" means.  The cost is that q's calibration is locked to
#           whatever stage A produced -- which is why stage A is a full
#           60-epoch run rather than a warm-up.
#
#   SPARSITY  the UtilityRouter is switched on.  It is the one component that
#           has never received a gradient: DinoSharedViT builds it
#           unconditionally, so it sits in every Stage 2R checkpoint as a
#           random initialisation, and dense mode never calls it.  Pruning to
#           target_k extras happens before the nine deep blocks, so the student
#           trains in the regime inference runs in.
#
#           The budget RAMPS rather than starting at 600, for two reasons.
#           Training sparse from epoch 0 with a random router would prune half
#           the extras AT RANDOM at step 0 and keep moving the kept set; and
#           through topk_gather a PRUNED token's utility gets exactly zero
#           gradient (u reaches the graph only via topk().indices, which is
#           non-differentiable, and gather(u, idx)), so K=N -- where the gather
#           is a permutation and every extra is in idx -- is the ONLY phase in
#           which the router can learn a full ranking at all.
#
#             epochs  0-7   K=1200  = dense, nothing thrown away
#             epochs  8-14  K=900
#             epochs 15-59  K=600   = the deployment budget
#
#           Evaluation always runs at target_k whatever the training budget
#           (see DinoTSReliableSparse.extract_feat), so every validation point
#           measures the deployment regime and the checkpoint selector never
#           compares a dense checkpoint against a sparse one.
#
#           NOTE the anchors are never pruned -- anchor_map must reshape to
#           [B, D, H, W] with exactly N entries -- so target_k=600 gives a
#           sequence of 1200 + 600 = 1800, a 25% reduction, not 50%.
_base_ = ['stage2r_additive_mfnet.py']

model = dict(
    type='DinoTSReliableSparse',
    forward_mode='sparse_hard',
    target_k=600,
    budget_schedule=(1.0, 0.75, 0.5),
    budget_warmup_epochs=30,
    soft_tau=1.0,
    freeze_shallow=True,
    freeze_reliability=True,
    # False -> train with the real Top-K gather, the exact deployment regime,
    # at the cost of top-k's gradient hole on the pruned positions.  True ->
    # soft_gate(hard=True): same 0/1 forward but the sequence keeps its full
    # length, so every token gets gradient.
    gate_hard=False)

# Stage A's accepted checkpoint.  Override with --cfg-options load_from=... to
# use best_mIoU instead of best_robust_score.
load_from = '{{$STAGE2R_A:/root/autodl-tmp/code/QWSEG/work_dirs/dino_ts/stage2r_additive/weight/stage2r_additive/best_robust_score_epoch_60.pth}}'
