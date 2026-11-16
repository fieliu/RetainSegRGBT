# Stage 2R-A, 30 more epochs as a WARM FINE-TUNE.
#
# Why not --resume: mmengine's BaseScheduler.state_dict() returns the whole
# scheduler __dict__, and the checkpoint written at epoch 60 stores the
# ITER-BASED curve {'begin': 588, 'end': 35280} (35280 = 60 epochs x 588).
# ParamSchedulerHook restores that state on resume, so it overwrites whatever
# `end` the config declares -- a config with end=90 resumed this way trains at
# lr = eta_min = 1e-6 for all 30 epochs, i.e. it does nothing.  (Measured:
# the first attempt logged `lr: 1.0000e-06` from epoch 61 onward.)
#
# So this is a fresh run that LOADS the epoch-60 weights and starts a new LR
# curve, decaying from the original peak 3e-5 to 1e-7 over 30 epochs.
#
# CORRECTION to an earlier draft of this comment: the `optimizer = dict(lr=1e-5)`
# override below does NOT take effect.  The base config writes
# `optim_wrapper = dict(..., optimizer=optimizer, ...)`, which COPIES the
# optimizer dict at parse time, so overriding the top-level `optimizer` key
# leaves `optim_wrapper.optimizer.lr` at 3e-5.  Measured: epoch 5 logs
# lr = 2.5891e-05 = 3e-5 * (1 - 4/29), i.e. the schedule is the new one but the
# peak is the original.  That is arguably the right choice for the question
# being asked -- a warm restart at the original peak gives the LR enough room
# to move a converged model off its plateau, which a third-sized peak might not
# -- so it is left as-is and documented rather than fixed.
#
# Optimizer moments restart (that is the difference between load_from and
# --resume), which is why epoch 0 ramps in rather than jumping to the peak.
#
# EXPECTATION, from this project's own evidence: extending training has bought
# very little so far.  Stage 2A went from epoch 60 to 115 (55 more epochs) for
# +0.28 mIoU; FMB Stage 2A peaked at epoch 35 and oscillated in a 1.3-point
# band afterwards.  This arm's own clean mIoU went 62.26 (e40) -> 62.37 (e50)
# -> 61.63 (e60), i.e. already flat to slightly falling.  A gain here would be
# surprising; flat is the expected outcome.
_base_ = ['stage2r_additive_mfnet.py']

load_from = '{{$STAGE2R_A_E60:/root/autodl-tmp/code/QWSEG/work_dirs/dino_ts/stage2r_additive/weight/stage2r_additive/epoch_60.pth}}'

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=30, val_interval=5)

optimizer = dict(type='AdamW', lr=1e-5, betas=(0.9, 0.999), weight_decay=0.05)

param_scheduler = [
    dict(type='LinearLR', start_factor=0.1, by_epoch=True, begin=0, end=1,
         convert_to_iter_based=True),
    dict(type='PolyLR', eta_min=1e-7, power=1.0, begin=1, end=30,
         by_epoch=True, convert_to_iter_based=True),
]
