# Stage 2R-A with the schedule EXTENDED from 60 to 90 epochs.
#
# Resuming the 60-epoch run as-is would be a no-op: its PolyLR(end=60) has
# already reached eta_min=1e-6 at epoch 60, so 30 more epochs would train at a
# learning rate of essentially zero.  Moving `end` to 90 puts epoch 60 back at
# ~1.08e-5 and decays it to 1e-6 by epoch 90 -- a real extension.
#
# `end` is a constructor argument of PolyLR rather than scheduler state, so it
# takes effect on --resume even though the scheduler's step counters are
# restored from the checkpoint.
#
# EXPECTATION, from this project's own evidence: extending training has bought
# very little so far.  Stage 2A went from epoch 60 to 115 (55 more epochs) for
# +0.28 mIoU; FMB Stage 2A peaked at epoch 35 and oscillated in a 1.3-point
# band afterwards.  And this arm's own clean mIoU went 62.26 (e40) -> 62.37
# (e50) -> 61.63 (e60), i.e. it was already flat to slightly falling.  So a
# gain here would be surprising; a flat result is the expected outcome.
_base_ = ['stage2r_additive_mfnet.py']

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=90, val_interval=5)

param_scheduler = [
    dict(type='LinearLR', start_factor=0.01, by_epoch=True, begin=0, end=1,
         convert_to_iter_based=True),
    dict(type='PolyLR', eta_min=1e-6, power=1.0, begin=1, end=90,
         by_epoch=True, convert_to_iter_based=True),
]
