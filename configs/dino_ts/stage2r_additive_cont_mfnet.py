# Stage 2R-A: 30 more epochs from the epoch-60 weights, at a LOW, VERIFIED LR.
#
# Two earlier attempts failed and both are worth recording:
#
#   1. `--resume` with PolyLR(end=90) trained at lr = eta_min = 1e-6 for 30
#      epochs, i.e. it did nothing.  mmengine's BaseScheduler.state_dict()
#      returns the whole scheduler __dict__, so the epoch-60 checkpoint stores
#      the ITER-BASED curve {'begin': 588, 'end': 35280} and
#      ParamSchedulerHook restores it, overwriting whatever `end` the config
#      declares.  A config with end=90 cannot be reached through --resume.
#
#   2. Stripping `param_schedulers` from a checkpoint COPY (keeping the
#      optimizer state) fixed the `end` problem but produced lr = 9.55e-05 --
#      3.19x the configured base lr of 3e-05, and identical across all 28
#      param groups even though their initial_lr values are the correct
#      layer-decay ladder (7.6e-06 .. 3e-05).  Whatever the scheduler computes
#      from, it is not the config's base lr.  Measured effect: the epoch-65
#      validation fell to clean 58.24 / rgb_missing 47.82 against the epoch-60
#      starting point of 61.63 / 55.63, so the run was stopped.
#
# So this arm does not use --resume at all.  It loads the weights, sets the
# learning rate explicitly on optim_wrapper.optimizer (the key that actually
# takes effect -- a top-level `optimizer = dict(lr=...)` override does NOT,
# because the base config copies that dict into optim_wrapper at parse time),
# and uses a plain epoch-based PolyLR so there is one less conversion to go
# wrong.  The LR is read back from the log before the run is allowed to
# continue.
#
# 1e-5 is a third of the original 3e-5 peak.  A warm restart at the full peak
# was already measured (the ft30 run) and it pushed the model back to its
# epoch-5 state -- clean fell 1.97 in five epochs.  The question here is
# whether a CONVERGED model can be nudged higher, so the LR has to be small
# enough not to undo the convergence.
_base_ = ['stage2r_additive_mfnet.py']

load_from = '{{$STAGE2R_A_E60:/root/autodl-tmp/code/QWSEG/work_dirs/dino_ts/stage2r_additive/weight/stage2r_additive/epoch_60_cont.pth}}'

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
