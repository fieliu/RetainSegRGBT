# Stage 2C on FMB: dense EMA robust training with degraded-view supervision.
#
# Same objective as the MFNet Stage 2C, on the FMB 13-class protocol:
#
#     L = L_seg(clean) + 1.0 * L_seg(degraded) + 1.0 * L_anchor
#
# Warm start from the FMB Stage-2A best checkpoint (13-class clean model), not
# from the MFNet weights: the class head and the 608x800 geometry differ.
_base_ = [
    '_base_dino_ts_m2f_fmb.py',
    '_base_dino_ts_data_fmb.py',
    '../_base_/default_runtime.py',
]

crop_size = (600, 800)

# NOTE: mmengine's env-var regex cannot cross a line break.
load_from = '{{$FMB_2A_BEST:/root/autodl-tmp/code/QWSEG/work_dirs/dino_ts/fmb_stage2a/weight/fmb_stage2a/best_mIoU_epoch_35.pth}}'

train_dataloader = dict(batch_size=2)

model = dict(
    type='DinoTSDenseEMA',
    data_preprocessor={{_base_.data_preprocessor}},
    backbone={{_base_.backbone}},
    neck={{_base_.neck}},
    decode_head={{_base_.decode_head}},
    forward_mode='dense',
    lambda_deg=1.0,
    lambda_missing=0.0,
    ema_momentum_base=0.996,
    ema_momentum_final=0.9999,
    total_epochs=60,
    lambda_anchor=1.0,
    lambda_global=0.0,
    degradation={{_base_.degradation_policy}},
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))

optimizer = dict(type='AdamW', lr=3e-5, betas=(0.9, 0.999), weight_decay=0.05)
optim_wrapper = dict(
    type='AmpOptimWrapper',
    loss_scale=dict(init_scale=128.0),
    accumulative_counts=1,
    optimizer=optimizer,
    constructor='LayerDecayOptimizerConstructor',
    paramwise_cfg=dict(num_layers=12, layer_decay_rate=0.9),
    clip_grad=dict(max_norm=1.0))

param_scheduler = [
    dict(type='LinearLR', start_factor=0.01, by_epoch=True, begin=0, end=1,
         convert_to_iter_based=True),
    dict(type='PolyLR', eta_min=1e-6, power=1.0, begin=1, end=60,
         by_epoch=True, convert_to_iter_based=True),
]

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=60, val_interval=5)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

randomness = dict(seed=471605737, deterministic=False)

custom_hooks = [
    dict(type='EpochSyncHook'),
    dict(type='EMAUpdateHook'),
    dict(type='RobustSelectionHook', priority='NORMAL',
         cases=['gaussian_noise/3', 't_gaussian_noise/3',
                'rgb_missing/1', 't_missing/1'],
         seed=42, clean_weight=0.5),
]

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=True),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook', by_epoch=True, interval=5, max_keep_ckpts=1,
        save_last=True, save_best=['robust_score', 'mIoU'],
        rule=['greater', 'greater']),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook', draw=False))

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
]
visualizer = dict(
    type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer')
