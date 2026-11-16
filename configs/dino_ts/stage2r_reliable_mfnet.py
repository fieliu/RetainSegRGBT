# Stage 2R: reliability-guided robust dense training on MFNet.
#
# Why this stage exists (measured 2026-10-10, docs/EXPERIMENT_01.md 2.7):
# running the Stage-2C weights as a PURE single-modality model gives 51.67 mIoU
# (thermal) / 56.93 (RGB) against dedicated baselines of 57.67 / 59.71, and
# bypassing the fusion is 0.4 mIoU WORSE than running it with one modality
# blacked out.  So the fusion is not the problem -- the shared trunk was simply
# never trained to operate on a single modality.  Whole-modality-missing is
# currently a 3.6%-per-direction byproduct of the corruption sampler with no
# dedicated loss, so this stage makes it an explicit, supervised view and adds
# distillation from the two single-modality baselines.
#
# The teachers are the DEDICATED single-modality baselines, not a clean
# multimodal teacher: matching a clean multimodal teacher on a missing view asks
# the student to recover information that is gone (docs/EXPERIMENT_01.md 3.2).
#
# NOTE: mmengine's env-var regex cannot cross a line break -- keep each
# {{$VAR:default}} on one line.
_base_ = [
    '_base_dino_ts_m2f.py',
    '_base_dino_ts_data.py',
    '../_base_/default_runtime.py',
]

crop_size = (480, 640)

_rgb_teacher_ckpt = '{{$RGB_TEACHER:/root/autodl-tmp/code/QWSEG/work_dirs/dino_ts/baseline_rgb_only/weight/baseline_rgb_only/best_mIoU_epoch_55.pth}}'
_thr_teacher_ckpt = '{{$THR_TEACHER:/root/autodl-tmp/code/QWSEG/work_dirs/dino_ts/baseline_thermal_only/weight/baseline_thermal_only/best_mIoU_epoch_85.pth}}'


def _backbone(fusion_type='anchor_extra', identity=False):
    """One explicit backbone block, shared by the student and both teachers.

    This is written out rather than using {{_base_.backbone}} on purpose.  A
    free-standing teacher dict never receives the `model`-level deep merge that
    the baseline configs rely on, and in the student's own `model = dict(...)`
    the base's `backbone` is still the literal placeholder string at merge time,
    so a nested dict REPLACES it instead of merging -- which silently drops
    `type`.  Spelling the block out removes both traps.

    identity=True disables both adapters.  An identity adapter has NO parameters
    at all (ModalityAdapter returns before building anything), so the teachers
    must be built that way or they would gain parameters the single-modality
    checkpoints do not contain.
    """
    bb = dict(
        type='DinoSharedViT',
        backbone_name='facebook/dinov3-vitb16-pretrain-lvd1689m',
        backbone_ckpt={{_base_.dinov3_checkpoint}},
        img_size={{_base_.dino_crop_size}},
        patch_size=16,
        embed_dims={{_base_.embed_dim}},
        depth=12,
        fusion_block=3,
        d_adapter=64,
        align_out_dim=256,
        thr_in_channels=3,
        freeze_vit=False,
        attention_backend='sdpa',
        with_cp=True,
        local_files_only=True,
        rgb_adapter_identity=identity,
        thermal_adapter_identity=identity)
    if fusion_type == 'reliable':
        bb['fusion_type'] = 'reliable'
        bb['reliable_fusion'] = dict(gate_mode='availability', head_hidden=128,
                                     tau=1.0, absent_thr=0.1)
    return bb


def _teacher_cfg(mode):
    return dict(
        type='DinoTSDense',
        data_preprocessor={{_base_.data_preprocessor}},
        backbone=_backbone(identity=True),
        neck={{_base_.neck}},
        decode_head={{_base_.decode_head}},
        forward_mode=mode,
        lambda_deg=0.0,
        lambda_missing=0.0,
        test_cfg=dict(mode='slide', crop_size=crop_size, stride=(320, 427)))


teacher_cfgs = [_teacher_cfg('rgb_only'), _teacher_cfg('thermal_only')]

model = dict(
    type='DinoTSReliable',
    data_preprocessor={{_base_.data_preprocessor}},
    backbone=_backbone(fusion_type='reliable'),
    neck={{_base_.neck}},
    decode_head={{_base_.decode_head}},
    forward_mode='dense',
    degradation={{_base_.degradation_policy}},
    teacher_cfgs=teacher_cfgs,
    teacher_ckpts=[_rgb_teacher_ckpt, _thr_teacher_ckpt],
    # Explicit modality-availability views: 60% ordinary corruption,
    # 20% RGB gone, 20% thermal gone.  Compare with the 3.6% per direction the
    # corruption sampler used to hand out.
    view_probs=(0.6, 0.2, 0.2),
    q_on_clean=True,
    lambda_v=1.0,
    lambda_q=1.0,
    # lambda_u scales a per-pixel KL against the NORMALISED teacher score map.
    # Measured magnitude on a warm-started model: ~0.34, against ~150 for the
    # extra view's segmentation terms -- so at 1.0 the distillation term is a
    # ~0.2% regulariser.  Raise it if the missing-modality numbers turn out to
    # need the teacher's help; the design intends it as an accelerator on top of
    # the (load-bearing) explicit views, not as the main signal.
    lambda_u=1.0,
    train_cfg=dict(),
    test_cfg=dict(mode='slide', crop_size=crop_size, stride=(320, 427)))

# Warm start from the Stage-2A dense teacher, as every other MFNet stage does.
# The fusion and the position table do not exist in that file, so they keep the
# designed initialisation (reliability head zero-init => q = 0.5 everywhere,
# PairScorer and interaction residual zero-init => the model starts exactly at
# an even mixture of the two modalities and grows from there).
load_from = '{{$STAGE2A_TEACHER:/root/autodl-tmp/code/QWSEG/work_dirs/dino_ts/stage2a_dense_teacher_e115_mIoU62.07.pth}}'

randomness = dict(seed=471605737, deterministic=False)

optimizer = dict(type='AdamW', lr=3e-5, betas=(0.9, 0.999), weight_decay=0.05)
optim_wrapper = dict(
    type='AmpOptimWrapper',
    loss_scale=dict(init_scale=128.0),
    accumulative_counts=1,
    optimizer=optimizer,
    constructor='LayerDecayOptimizerConstructor',
    paramwise_cfg=dict(num_layers=12, layer_decay_rate=0.9),
    clip_grad=dict(max_norm=1.0))

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=60, val_interval=5)
# _base_dino_ts_data.py sets batch_size=1 and every other MFNet stage overrides
# it to 2 (588 iters/epoch).  Inheriting the base directly would silently run
# 1176 iters/epoch -- a different effective batch from every number this stage
# will be compared against, and roughly double the wall clock.
train_dataloader = dict(batch_size=2)
# Runner-level loop configs.  These are NOT the same thing as model.test_cfg
# (which sets the inference mode); every stage config declares all three, and
# mmengine requires val_dataloader / val_cfg / val_evaluator to be all present
# or all absent.
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

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
