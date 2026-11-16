# Stage 2R v3 arm 1: ABSOLUTE + IoU reliability targets, deletion bypass.
#
# Baseline being compared against: Stage 2R v2 e60, full 58-case benchmark
#   clean 62.01 / corrupted 59.69 / rgb_missing 55.92 / t_missing 57.99
# produced by the code committed as 42a4026 ("restore the exact code that
# produced the Stage 2R v2 e60 result").  This config changes exactly two
# things about the training objective and nothing else -- same seed, same
# backbone/neck/head, same views, same teachers, same schedule -- so the only
# difference the comparison can see is the target definition.
#
# Change 1 -- quality_normalize=False (absolute targets)
#   v2 target:  y = metric(degraded) / metric(clean)      "capability RETAINED"
#   v3 target:  y = metric(degraded)                      "ABSOLUTE support"
#   alpha = softmax([log q^R, log q^T]) compares the two modalities AT THE SAME
#   TOKEN, so it needs "which one is more accurate here", not "which one lost
#   less".  The two disagree: RGB 0.90->0.45 and thermal 0.30->0.30 give
#   retained-fractions 0.50 vs 1.00 (picking thermal) but absolute scores 0.45
#   vs 0.30 (picking RGB, which is the right answer).  The clean teacher's
#   scores are still passed and still FILTER classes it never got right
#   (min_reference stays 0.1); they are just no longer the divisor.
#
# Change 2 -- quality_metric='iou' instead of 'accuracy'
#   accuracy is |pred n gt| / |gt| and is blind to the teacher over-painting a
#   class into neighbouring pixels; IoU is |pred n gt| / |pred u gt| and is not.
#   IoU <= accuracy always, so the min_reference gate is STRICTER here -- the
#   qcov/qtgt diagnostics below are the instrument for checking that this did
#   not quietly shrink the supervised token set.
#
# Change 3 -- quality_missing_min_deleted_frac=0.5 (explicit-deletion bypass)
#   A token whose GT is (mostly) inside an explicitly deleted region has no
#   clean reference and does not need one: its correct target is exactly 0,
#   because the modality carries no evidence there.  v2 dropped such tokens
#   from the loss whenever every one of their classes failed the reference
#   filter, so the modality was never taught to report q=0 exactly where it is
#   absent -- the one place the convex mixture needs it.  v2's diagnostics
#   report qresc (deleted tokens this rescues) so the effect can be measured
#   rather than assumed; qresc == 0 would mean it is a correctness fix only.
#
# NOT changed, deliberately: gate_mode stays 'availability' and the hard
# threshold stays at 0.1; the RobustSelectionHook keeps its 4 cases so the
# selected epoch stays comparable with e60's.
#
# The gate argument, CORRECTED (an earlier draft of this comment argued "you
# would need >=9 simultaneously-missed classes, and MFNet has only 8 object
# classes plus unlabeled". That is wrong twice over and is kept here only so
# nobody re-derives it.) The real reason the gate cannot close by mistake:
# on the CLEAN view teacher_logits IS reference_logits, so score == ref
# elementwise, every class that passes the filter has score >= min_reference,
# and the support-weighted mean of values all >= min_reference is itself
# >= min_reference -- structurally, not by counting classes. The
# 0.4844/0.3125/0.2266/0.1750 figures that argument was built on are
# DEGRADED-view numbers, where score and ref come from different teachers.
#
# WHAT THIS ARM ACTUALLY CHANGES (found by audit after the first draft, and
# bigger than "target semantics" suggests): in v2 the target collapsed to
# exactly 1.0 on every view where the modality's INPUT WAS UNTOUCHED, because
# score and ref are then the same computation on the same tensor. That is the
# whole clean view, and the surviving modality of every missing view. So v2's
# only informative target was the corrupted modality's ratio A/R, and
#     alpha_v2 = sigmoid(L - d)   vs   alpha_v3 = sigmoid(L)
# with L = log(A_R/A_T) and d = log(R_R/R_T). The v2 normalisation therefore
# SUBTRACTED a systematic bias toward whichever modality has the better clean
# teacher -- and the RGB teacher is the stronger one (59.71 vs 57.67), so v2
# was biased away from RGB. Removing d is the main effect of this arm, and it
# lands on the CLEAN path, which is where the 62.01 headline lives. Watch
# clean mIoU first; watch clean.q_rgb_bg/fg and clean.target_rgb_bg/fg to see
# the new token-dependence directly.
_base_ = ['stage2r_reliable_mfnet.py']

model = dict(
    quality_metric='iou',
    quality_normalize=False,
    quality_min_reference=0.1,
    quality_missing_min_deleted_frac=0.5)
