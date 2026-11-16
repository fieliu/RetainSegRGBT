"""Stage 2R: reliability-guided robust dense training.

Motivation (measured 2026-10-10, see docs/EXPERIMENT_01.md 2.7)
--------------------------------------------------------------
Running the Stage-2C EMA weights as a PURE single-modality model gives
51.67 mIoU (thermal) / 56.93 (RGB), against dedicated single-modality
baselines of 57.67 / 59.71.  Bypassing the fusion is 0.4 mIoU *worse* than
running it with one modality blacked out, so the fusion is not corrupting the
features -- the 6.00 / 2.78 gap is in the weights.  The shared trunk and the
adapters were never trained to operate on one modality.

Whole-modality-missing is currently only a 3.6%-per-direction byproduct of the
corruption sampler (7.2% overall) and carries no dedicated loss: DinoTSDenseEMA
overrides loss() and never calls _make_missing, so its lambda_missing branch is
dead code.  Capability comes from supervision, so this stage makes
modality-availability an EXPLICIT view with its own probability and its own
supervision:

    L = L_seg(clean)
      + lambda_v * L_seg(view)            view ~ 60% corrupt / 20% rgb_missing
                                                / 20% t_missing
      + lambda_q * L_quality              q supervised by the frozen
                                          single-modality teachers
      + lambda_u * L_uniKD                missing views distilled from the
                                          matching single-modality teacher

Two frozen single-modality teachers are used (NOT the clean multimodal
teacher).  Matching a clean multimodal teacher on a missing view asks the
student to recover information that no longer exists; the ablation in
docs/EXPERIMENT_01.md 3.2 shows it is worth little even on corrupted views.
"""
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.registry import MODELS

from .stage2a_dense import DinoTSDense
from .reliability_losses import (quality_regression, token_quality_targets,
                                 unimodal_distillation)

# Branch ids for the explicit view sampler.
VIEW_CORRUPT, VIEW_RGB_MISSING, VIEW_T_MISSING = 0, 1, 2


@MODELS.register_module()
class DinoTSReliable(DinoTSDense):
    """Dense training with explicit modality-availability views.

    Args:
        teacher_cfgs: two configs, [RGB-only, Thermal-only].  Each is built as a
            DinoTSDense in the matching forward_mode and frozen.
        teacher_ckpts: matching checkpoint paths.
        view_probs: (corrupt, rgb_missing, t_missing) for the extra view.
        q_on_clean: also supervise the reliability head on the clean view.
            Costs two extra teacher forwards per step; turning it off halves
            the teacher bill but leaves the "both modalities reliable" regime
            unsupervised, which is what q is supposed to contrast against.
    """

    def __init__(self, *args,
                 teacher_cfgs: Optional[List[dict]] = None,
                 teacher_ckpts: Optional[List[str]] = None,
                 view_probs=(0.6, 0.2, 0.2),
                 q_on_clean: bool = True,
                 lambda_v: float = 1.0,
                 lambda_q: float = 1.0,
                 lambda_u: float = 1.0,
                 lambda_feat: float = 0.0,
                 quality_target_mode: str = 'class_balanced',
                 quality_reduction: str = 'class_balanced',
                 quality_min_class_pixels: float = 4.0,
                 quality_metric: str = 'accuracy',
                 quality_normalize: bool = True,
                 quality_min_reference: float = 0.1,
                 kd_reduction: str = 'class_balanced',
                 kd_teacher_filter: str = 'correct',
                 **kwargs):
        kwargs.setdefault('forward_mode', 'dense')
        super().__init__(*args, **kwargs)

        if not teacher_cfgs or len(teacher_cfgs) != 2:
            raise ValueError('teacher_cfgs must hold [rgb_only, thermal_only]')
        if not teacher_ckpts or len(teacher_ckpts) != 2:
            raise ValueError('teacher_ckpts must hold two paths')
        probs = torch.tensor([float(p) for p in view_probs], dtype=torch.float64)
        if (probs.numel() != 3 or not torch.isfinite(probs).all()
                or (probs < 0).any() or probs.sum() <= 0):
            raise ValueError('view_probs must be three non-negative numbers')
        self.register_buffer('view_probs', (probs / probs.sum()).float())

        self.teachers = nn.ModuleList()
        self.teacher_ckpts = tuple(teacher_ckpts)
        self._teachers_initialized = False
        for cfg in teacher_cfgs:
            t = MODELS.build(cfg)
            t.eval()
            for p in t.parameters():
                p.requires_grad = False
            self.teachers.append(t)

        self.q_on_clean = q_on_clean
        self.lambda_v = lambda_v
        self.lambda_q = lambda_q
        self.lambda_u = lambda_u
        self.lambda_feat = float(lambda_feat)
        self.ignore_index = 255
        if quality_target_mode not in ('class_balanced', 'pixel_accuracy'):
            raise ValueError('Invalid quality_target_mode')
        if any(x not in ('class_balanced', 'pixel')
               for x in (quality_reduction, kd_reduction)):
            raise ValueError('Invalid loss reduction')
        if kd_teacher_filter not in ('correct', 'none'):
            raise ValueError('Invalid kd_teacher_filter')
        if quality_min_class_pixels <= 0:
            raise ValueError('quality_min_class_pixels must be positive')
        if quality_metric not in ('accuracy', 'iou'):
            raise ValueError('quality_metric must be accuracy or iou')
        if not 0.0 <= quality_min_reference < 1.0:
            raise ValueError('quality_min_reference must be in [0, 1)')
        self.quality_target_mode = quality_target_mode
        self.quality_reduction = quality_reduction
        self.quality_min_class_pixels = quality_min_class_pixels
        self.quality_metric = quality_metric
        self.quality_normalize = quality_normalize
        self.quality_min_reference = quality_min_reference
        self.kd_reduction = kd_reduction
        self.kd_teacher_filter = kd_teacher_filter

    def init_weights(self):
        if self._is_init and self._teachers_initialized:
            return
        super().init_weights()
        # Load AFTER initialization, which otherwise can reset teacher heads.
        for t, ckpt in zip(self.teachers, self.teacher_ckpts):
            t.init_weights()
            state = torch.load(ckpt, map_location='cpu', weights_only=False)
            sd = state.get('state_dict', state)
            sd = {(k[7:] if k.startswith('module.') else k): v
                  for k, v in sd.items()}
            if any(k.startswith('ema.') for k in sd):
                sd = {k: v for k, v in sd.items() if not k.startswith('ema.')}
            missing, unexpected = t.load_state_dict(sd, strict=False)
            if missing or unexpected:
                raise RuntimeError('teacher %s: %d missing / %d unexpected keys'
                                   % (ckpt, len(missing), len(unexpected)))
            t.eval()
        self._teachers_initialized = True

    def train(self, mode=True):
        super().train(mode)
        for t in self.teachers:
            t.eval()  # no_grad/disabled parameter gradients do not freeze BN.
        return self

    def state_dict(self, *args, **kwargs):
        """Drop the frozen teachers.

        They are frozen, byte-identical at every epoch, and reloaded from
        their own files by init_weights(), so serialising them into every
        checkpoint costs ~1 GB per file for nothing.
        """
        sd = super().state_dict(*args, **kwargs)
        for key in [k for k in sd if k.startswith('teachers.')]:
            del sd[key]
        return sd

    # -- explicit views -----------------------------------------------------
    def _missing_view(self, inputs, data_samples, which: str):
        """Black out one whole modality.  Raw black pixels, as at test time."""
        from rgbt_c.protocol import operation, recipe
        rgb, thermal = self._split(inputs)
        shapes = [ds.img_shape for ds in data_samples]
        spec = recipe([operation(which, 1, 0)])
        r, t, _, _ = self.degrader.apply(rgb, thermal, [spec] * rgb.shape[0],
                                         valid_shapes=shapes)
        return torch.cat([r, t], dim=1)

    def _build_extra_view(self, inputs, data_samples):
        from rgbt_c.protocol import operation, recipe, spatial_mask
        branch = int(torch.multinomial(self.view_probs, 1).item())
        rgb, thermal = self._split(inputs)
        shapes = [ds.img_shape for ds in data_samples]
        if branch == VIEW_CORRUPT:
            specs = self.degrader.sample(inputs.shape[0])
        else:
            which = 'rgb_missing' if branch == VIEW_RGB_MISSING else 't_missing'
            specs = [recipe([operation(which, 1, 0)]) for _ in data_samples]
        r, t, _, _ = self.degrader.apply(rgb, thermal, specs, valid_shapes=shapes)
        # Supervision metadata only: never supplied to extract_feat/fusion.
        observation = inputs.new_ones((inputs.shape[0], 2, *inputs.shape[-2:]))
        for b, (spec, shape) in enumerate(zip(specs, shapes)):
            h, w = shape[:2]
            for op in spec['operations']:
                if op['name'] in ('rgb_missing', 't_missing'):
                    m = 0 if op['name'] == 'rgb_missing' else 1
                    mask = torch.from_numpy(spatial_mask((h, w), op)[..., 0])
                    observation[b, m, :h, :w] *= 1 - mask.to(observation)
        return torch.cat([r, t], dim=1), branch, observation

    # -- teacher targets ----------------------------------------------------
    @torch.no_grad()
    def _teacher_logits(self, inputs, data_samples, indices=(0, 1)):
        """Per-pixel class distributions from both frozen teachers.

        Returns a list of [B, C, h, w] tensors, one per teacher, computed on
        the SAME (possibly degraded) inputs the student sees.  Mask2FormerHead
        .predict returns seg_logits, not argmax, so this is the per-pixel
        distribution the design asks for rather than a hard pseudo-label.
        """
        img_metas = [dict(ds.metainfo, batch_input_shape=inputs.shape[-2:])
                     for ds in data_samples]
        out = [None, None]
        for index in indices:
            t = self.teachers[index]
            t.eval()
            feats = t.extract_feat(inputs)
            out[index] = t.decode_head.predict(feats, img_metas, t.test_cfg)
        return out

    def _token_correctness(self, seg_logits, gt, patch: int):
        """Legacy accuracy proxy for ablations; black guesses need not score 0."""
        return token_quality_targets(
            seg_logits, gt, patch, target_mode='pixel_accuracy',
            ignore_index=self.ignore_index)[0]

    # -- losses -------------------------------------------------------------
    def _view_losses(self, inputs, data_samples, branch, prefix,
                     observation=None, quality_scale=1.0,
                     reference_logits=None):
        """Segmentation loss + uniKD for one extra view.

        The student forward is run ONCE here and its features are reused for
        both the segmentation loss and the distillation logits -- calling
        _seg_loss and then extract_feat again would silently double the cost of
        every step.
        """
        losses = {}
        feats = self.extract_feat(inputs)
        backbone_out = self._last_backbone_out
        for k, v in self.decode_head.loss(feats, data_samples,
                                          self.train_cfg).items():
            losses[f'{prefix}.decode.{k}'] = self.lambda_v * v
        needs_kd = branch != VIEW_CORRUPT and self.lambda_u > 0
        teacher_idx = 1 if branch == VIEW_RGB_MISSING else 0
        if self.lambda_q > 0 or needs_kd:
            indices = (0, 1) if self.lambda_q > 0 else (teacher_idx,)
            teacher_logits = self._teacher_logits(inputs, data_samples, indices)
        else:
            teacher_logits = [None, None]
        if needs_kd:
            # Distil from the teacher that was trained on exactly the input that
            # SURVIVES.  With RGB gone only thermal remains, so the target is
            # the Thermal-only teacher (index 1); with thermal gone it is the
            # RGB-only teacher (index 0).  Getting this backwards silently
            # distils from a teacher staring at a black frame.
            t_logits = teacher_logits[teacher_idx]
            s_logits = self.decode_head.predict(
                feats, [dict(ds.metainfo, batch_input_shape=inputs.shape[-2:])
                        for ds in data_samples], self.test_cfg)

            # NORMALISED per-pixel class distribution, as the design specifies.
            # Mask2FormerHead.predict returns score-like maps -- einsum of
            # softmax class weights and sigmoid masks -- in [0, 1], NOT logits.
            # Applying softmax(./T) to them leaves student and teacher both
            # near-uniform and washes the signal out: measured KL 0.009-0.013
            # for T in {1.0, 0.5, 0.2} against 0.337 for the normalised form, a
            # 27x difference.  Hence no temperature here.
            #
            # reduction='batchmean' would also be wrong: its "batch size" is
            # dim 0, so on [B, C, H, W] it sums over C*H*W too and comes out
            # ~3 orders of magnitude too large (measured 5373), which would let
            # this term swamp the segmentation loss.
            gt = torch.stack([ds.gt_sem_seg.data.squeeze(0).long()
                              for ds in data_samples])
            kd, diagnostics = unimodal_distillation(
                s_logits, t_logits, gt, reduction=self.kd_reduction,
                teacher_filter=self.kd_teacher_filter,
                ignore_index=self.ignore_index)
            losses['loss_unikd'] = self.lambda_u * kd
            losses.update(diagnostics)

            if self.lambda_feat > 0:
                # Feature-level distillation on the missing view: align the
                # student's fused anchor map with the surviving teacher's
                # anchor map.  The teacher ran extract_feat() inside
                # _teacher_logits and left its anchor map on _last_backbone_out.
                t_anchor = self.teachers[teacher_idx]._last_backbone_out[
                    'anchor_map']
                s_anchor = backbone_out['anchor_map']
                feat_kd = (F.normalize(s_anchor, dim=1)
                           - F.normalize(t_anchor, dim=1)).pow(2).mean()
                losses['loss_featkd'] = self.lambda_feat * feat_kd
        if self.lambda_q > 0:
            losses.update(self._quality_loss(
                inputs, data_samples, backbone_out, teacher_logits,
                observation, prefix=prefix, scale=quality_scale,
                reference_logits=reference_logits))
        return losses

    def _quality_loss(self, inputs, data_samples, backbone_out,
                      teacher_logits=None, observation=None, prefix='quality',
                      scale=1.0, reference_logits=None):
        q_rgb, q_t = backbone_out.get('q_rgb'), backbone_out.get('q_t')
        if q_rgb is None or q_t is None:
            raise RuntimeError('Quality loss needs fusion_type="reliable" outputs')
        gt = torch.stack(
            [ds.gt_sem_seg.data.squeeze(0).long() for ds in data_samples])
        patch = self.backbone.patch_size
        t_logits = (self._teacher_logits(inputs, data_samples)
                    if teacher_logits is None else teacher_logits)
        if reference_logits is None and self.quality_normalize:
            raise RuntimeError(
                'quality_normalize needs the CLEAN teacher scores as the '
                'denominator; pass reference_logits')
        values, diagnostics = [], {}
        for m, (q, name) in enumerate(((q_rgb, 'rgb'), (q_t, 't'))):
            obs = None if observation is None else observation[:, m]
            target, fractions, valid = token_quality_targets(
                t_logits[m], gt, patch, observation=obs,
                reference_scores=(reference_logits[m]
                                  if self.quality_normalize else None),
                target_mode=self.quality_target_mode,
                metric=self.quality_metric,
                min_class_pixels=self.quality_min_class_pixels,
                min_reference=self.quality_min_reference,
                ignore_index=self.ignore_index)
            values.append(quality_regression(q, target, fractions, valid,
                                             self.quality_reduction))
            for group, membership in (('bg', fractions[:, 0]),
                                      ('fg', fractions[:, 1:].sum(1))):
                mass = membership * valid
                den = mass.sum().clamp_min(1e-12)
                diagnostics[f'{prefix}.q_{name}_{group}'] = (
                    (q.detach() * mass).sum() / den)
                diagnostics[f'{prefix}.target_{name}_{group}'] = (
                    (target * mass).sum() / den)
        diagnostics[f'{prefix}.loss_quality'] = (
            self.lambda_q * scale * 0.5 * sum(values))
        return diagnostics

    def loss(self, inputs, data_samples):
        losses = dict()
        if not self._teachers_initialized:
            raise RuntimeError('Call init_weights() before Stage 2R training')
        quality_scale = 0.5 if self.q_on_clean else 1.0
        # clean view: GT supervision, and the reliability head's anchor regime
        clean_out = self.extract_feat(inputs)
        losses.update({f'clean.{k}': v for k, v in
                       self.decode_head.loss(clean_out, data_samples,
                                             self.train_cfg).items()})

        # Teacher scores on the CLEAN input, computed ONCE and shared: they are
        # both the clean view's own target and the denominator of the extra
        # view's normalised target.  Running the teachers again inside
        # _quality_loss would double this part of the step for nothing.
        need_clean_teachers = self.lambda_q > 0 and (self.q_on_clean
                                                     or self.quality_normalize)
        clean_t_logits = (self._teacher_logits(inputs, data_samples)
                          if need_clean_teachers else None)

        if self.q_on_clean and self.lambda_q > 0:
            losses.update(self._quality_loss(
                inputs, data_samples, self._last_backbone_out,
                teacher_logits=clean_t_logits, prefix='clean',
                scale=quality_scale, reference_logits=clean_t_logits))

        view_inputs, branch, observation = self._build_extra_view(inputs, data_samples)
        losses.update(self._view_losses(view_inputs, data_samples, branch, 'view',
                                        observation, quality_scale,
                                        reference_logits=clean_t_logits))
        return losses
