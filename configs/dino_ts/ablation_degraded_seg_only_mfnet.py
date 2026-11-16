# Ablation: degraded supervised segmentation ONLY -- no EMA teacher, no anchor
# consistency.
#
# This is the control that answers "what does Stage 2C actually contribute?".
# Stage 2C's objective is
#
#     L = L_seg(clean) + 1.0 * L_seg(degraded) + 1.0 * L_anchor
#                        ^^^^^^^^^^^^^^^^^^^   ^^^^^^^^^^^^
#                        plain augmented training    the doc's claimed contribution
#
# and its corrupted mIoU rose 49.30 -> 57.24 on the four proxy cases.  Nothing
# so far separates the two terms.  This config keeps everything else identical
# -- same warm start, same seed, same degradation policy, same 60-epoch
# schedule -- and drops only L_anchor (and with it the EMA teacher, which then
# has no target to serve).
#
# Implemented with DinoTSDense rather than DinoTSDenseEMA(lambda_anchor=0):
# with lambda_anchor = 0 the EMA teacher receives no gradient path at all, so
# the two are the same computation without the extra model copy.  The strong
# view is still drawn from the same shared policy by _make_degraded, so the
# degraded input distribution is unchanged.
#
# Reading the result:
#   ~59.3  -> L_anchor contributes almost nothing; Stage 2C is a training recipe
#   ~55    -> L_anchor is doing real work and needs its own explanation
_base_ = ['stage2a_dense_mfnet.py']

crop_size = (480, 640)

model = dict(
    type='DinoTSDense',
    forward_mode='dense',
    lambda_deg=1.0,          # supervised segmentation on the degraded view
    lambda_missing=0.0,      # missing-modality handled by the policy, not a
                             # separate view (same as Stage 2C)
    test_cfg=dict(mode='slide', crop_size=crop_size, stride=(320, 427)))

# Same warm start as Stage 2C -- without this the control would train from
# DINOv3 and the comparison would be meaningless.  stage2a_dense_mfnet.py leaves
# load_from = None because the other derived configs set it themselves.
# NOTE: mmengine's env-var regex cannot cross a line break.
load_from = '{{$STAGE2A_TEACHER:/root/autodl-tmp/code/QWSEG/work_dirs/dino_ts/stage2a_dense_teacher_e115_mIoU62.07.pth}}'

randomness = dict(seed=471605737, deterministic=False)

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=60, val_interval=5)

param_scheduler = [
    dict(type='LinearLR', start_factor=0.01, by_epoch=True, begin=0, end=1,
         convert_to_iter_based=True),
    dict(type='PolyLR', eta_min=1e-6, power=1.0, begin=1, end=60,
         by_epoch=True, convert_to_iter_based=True),
]

custom_hooks = [
    dict(type='EpochSyncHook'),
    dict(type='RobustSelectionHook', priority='NORMAL',
         cases=['gaussian_noise/3', 't_gaussian_noise/3',
                'rgb_missing/1', 't_missing/1'],
         seed=42, clean_weight=0.5),
]

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook', by_epoch=True, interval=5, max_keep_ckpts=1,
        save_last=True, save_best=['robust_score', 'mIoU'],
        rule=['greater', 'greater']),
    visualization=dict(type='SegVisualizationHook', draw=False))
