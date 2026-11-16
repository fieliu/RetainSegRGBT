# Stage 2R stage A: the v2/e60 objective + the additive reliability branch.
#
# This is the arm that produced the best result so far (clean 62.01 /
# corrupted 59.69 / rgb_missing 55.92 / t_missing 57.99), with ONE change:
#
#   the reliability channels get their own projection instead of being two
#   columns of the 4D+2 = 3074-wide descriptor the fusion builds.  In that
#   concatenation they held 0.065% of the first Linear's pre-activation
#   variance, and their weights would have had to travel ~1.6 to matter --
#   against the ~0.55 an Adam schedule at this learning rate can move a weight
#   over the whole run, so they could not get there at all.  additive_rel=True
#   gives them a Linear(4, h) of their own, zero-initialised, added AFTER the
#   GELU (adding before it would multiply this branch's gradient by GELU'(z),
#   which vanishes exactly where the token path saturates).  Zero-init makes
#   step 0 identical to the reliability-blind fusion, so the norm of those
#   weights is a free diagnostic of how much the branch ended up being used.
#
# Everything else is inherited unchanged: accuracy targets, normalised, the
# explicit-deletion bypass OFF, the unimodal KD on the missing views, the two
# single-modality teachers, view_probs (0.6/0.2/0.2), the seed, the 60-epoch
# schedule and the Stage-2A warm start.  Dense throughout -- routing is stage
# B's job.
#
# router_rel_dim is set here too even though dense mode never calls the router:
# it keeps stage A's and stage B's architectures identical, so stage B's
# load_from is a clean same-shape load.
_base_ = ['stage2r_reliable_mfnet.py']

model = dict(
    backbone=dict(
        router_rel_dim=4,
        reliable_fusion=dict(additive_rel=True)))
