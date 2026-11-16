# Stage 2R + clean-bimodal dense KD.
#
# The distillation target changes role.  In the v2/v3 arms the two
# single-modality teachers did double duty: they supplied the reliability
# target AND the KL target on the missing views.  Here they supply ONLY the
# reliability target, and a single CLEAN BIMODAL teacher supplies a dense
# per-pixel KD target for EVERY view -- clean, ordinary degradation, and
# whole-modality missing alike.
#
# The teacher always runs on the CLEAN input, so its logits are a function of
# the image alone and are computed once per step and shared across views.  On
# the clean view the student and the teacher start identical (the student is
# warm-started from this very checkpoint), so that term begins at 0 and acts as
# a "do not forget" regulariser; on the degraded and missing views it is the
# usual invariance distillation, asking the student to predict what a model
# that can see both modalities would say.
#
# Everything else is inherited unchanged from the v3 arm, so the comparison
# against stage2r_v3_iou_abs_mfnet.py isolates the KD change.
_base_ = ['stage2r_v3_iou_abs_mfnet.py']

crop_size = (480, 640)

# The Stage 2A dense teacher: the strongest clean bimodal model on disk
# (clean mIoU 62.07), and the checkpoint the student itself warm-starts from.
_bimodal_ckpt = '{{$STAGE2A_TEACHER:/root/autodl-tmp/code/QWSEG/work_dirs/dino_ts/stage2a_dense_teacher_e115_mIoU62.07.pth}}'

# Spelled out rather than reusing the base's _backbone() helper: a function
# defined in a base config is not in scope in the child, and a free-standing
# teacher dict never receives the model-level deep merge that the baseline
# configs rely on.  identity=False because Stage 2A trained WITH its adapters.
_bimodal_backbone = dict(
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
    rgb_adapter_identity=False,
    thermal_adapter_identity=False)

model = dict(
    kd_source='bimodal_clean',
    lambda_kd=1.0,
    # Unused once kd_source is 'bimodal_clean'; zeroed so the log cannot show a
    # term that is not being computed.
    lambda_u=0.0,
    bimodal_ckpt=_bimodal_ckpt,
    bimodal_cfg=dict(
        type='DinoTSDense',
        data_preprocessor={{_base_.data_preprocessor}},
        backbone=_bimodal_backbone,
        neck={{_base_.neck}},
        decode_head={{_base_.decode_head}},
        forward_mode='dense',
        lambda_deg=0.0,
        lambda_missing=0.0,
        test_cfg=dict(mode='slide', crop_size=crop_size,
                      stride=(320, 427))))
