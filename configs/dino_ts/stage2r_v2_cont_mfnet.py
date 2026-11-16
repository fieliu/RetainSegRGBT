# v2 (the best result so far) trained 30 more epochs from its own epoch-60.
#
# This is the arm WITHOUT the additive reliability branch: _base_ is
# stage2r_reliable_mfnet.py directly, so the fusion keeps its original
# concatenated descriptor (4D+2) and reliability travels the path that
# produced clean 62.01 / corrupted 59.69 / rgb_missing 55.92 / t_missing 57.99.
#
# WHY THIS IS WORTH TRYING, unlike the additive arm: v2's LAST validation was
# its BEST one.  clean went 61.56 (e40) -> 61.80 -> 61.62 -> 61.47 -> 62.01
# (e60) and robust_score went 59.40 -> 59.63 -> 59.54 -> 59.45 -> 60.20.  The
# final +0.54 / +0.75 jump is inside the +-2 single-run noise, but it is the
# one arm in this project whose curve had not visibly turned over at the end.
# The additive arm peaked at e50 and fell; Stage 2A was flat from e60 to e115;
# FMB Stage 2A peaked at e35.
#
# SETUP, identical to the additive fine-tune so the two are comparable:
#   load_from    the accepted e60 checkpoint (weights only -- optimizer
#                moments restart, which is the difference from --resume)
#   lr           1e-5, a third of the original 3e-5 peak.  Measured on the
#                additive arm: a 3e-5 warm restart drops clean by 1.97 within
#                five epochs (it pushes the model back to its epoch-5 state),
#                while 1e-5 leaves the starting point intact (-0.01 on
#                robust_score at e5).  The question is whether a CONVERGED
#                model can be nudged higher, so the LR must not undo the
#                convergence.
#   schedule     plain epoch-based PolyLR 1e-5 -> 1e-7 over 30 epochs.  No
#                convert_to_iter_based: the iter-based conversion is one of
#                the things that went wrong on the --resume attempts.
#
# NOT --resume, deliberately.  mmengine's BaseScheduler.state_dict() stores
# the whole scheduler __dict__, so a resumed run restores the checkpoint's
# iter-based curve and ignores the config's `end` -- extending a schedule
# through --resume is impossible without editing the checkpoint, and the one
# attempt at that produced an lr 3.19x the configured base and destroyed the
# model.  load_from sidesteps the whole subsystem.
_base_ = ['stage2r_reliable_mfnet.py']

load_from = '{{$V2_E60:/root/autodl-tmp/code/QWSEG/work_dirs/dino_ts/stage2r_reliable_v2/weight/stage2r_reliable_v2/best_robust_score_epoch_60.pth}}'

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=30, val_interval=5)

optim_wrapper = dict(
    type='AmpOptimWrapper',
    loss_scale=dict(init_scale=128.0),
    accumulative_counts=1,
    optimizer=dict(type='AdamW', lr=1e-5, betas=(0.9, 0.999),
                   weight_decay=0.05),
    constructor='LayerDecayOptimizerConstructor',
    paramwise_cfg=dict(num_layers=12, layer_decay_rate=0.9),
    clip_grad=dict(max_norm=1.0))

param_scheduler = [
    dict(type='PolyLR', eta_min=1e-7, power=1.0, begin=0, end=30,
         by_epoch=True),
]
