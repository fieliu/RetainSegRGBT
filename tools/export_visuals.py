"""Export everything the figures need, for one (dataset, scene, case).

Emits, per case:
  metadata.json  rgb_clean/thermal_clean/rgb_observed/thermal_observed (.npy)
  damage_mask.npy  gt.npy  pred.npy
  quality.npz    q_rgb, q_t, alpha, gate, grid
  features.npz   F_R, F_T, F_fused (anchor), F_deep

Case spec:
  clean | rgb_missing | t_missing
  id:<rgb|t>:<operator>:<severity>:<global|scatter>
  ood:<rgb|t>:<operator>:<severity>:<global|scatter>

Usage:
  python export_visuals.py --config CFG --checkpoint CKPT --dataset pst900 \
      --image 58_bag21_rect_rgb_frame0000000440.png --case rgb_missing \
      --out_dir /root/exports/pst900/scene01/rgb_missing
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, '/root/autodl-tmp/code/QWSEG/seg/mmsegmentation-main-rgbt')
sys.path.insert(0, '/root/autodl-tmp/code/QWSEG')

from mmengine.config import Config
from mmengine.dataset import pseudo_collate
from mmengine.registry import init_default_scope
from mmengine.runner.checkpoint import load_checkpoint
from mmseg.registry import MODELS, TRANSFORMS

from rgbt_c.protocol import (HELDOUT_CORRUPTIONS, MISSING, RGB_CORRUPTIONS,
                             TRAIN_CORRUPTIONS, apply_pair, operation, recipe,
                             stable_seed)


def parse_case(spec):
    """-> dict or None (clean) or str (missing)."""
    if spec == 'clean':
        return None
    if spec in MISSING:
        return spec
    parts = spec.split(':')
    if len(parts) != 5:
        raise ValueError('case must be clean | missing | tag:mod:op:sev:scope')
    tag, mod, op, sev, scope = parts
    if scope not in ('global', 'local', 'local_soft', 'scatter'):
        raise ValueError('unknown scope %r' % scope)
    pool = TRAIN_CORRUPTIONS if tag == 'id' else HELDOUT_CORRUPTIONS
    if op not in pool:
        raise ValueError('%s is not in the %s pool' % (op, tag))
    if (op in RGB_CORRUPTIONS) != (mod == 'rgb'):
        raise ValueError('operator %s does not belong to modality %s'
                         % (op, mod))
    return dict(op=op, sev=int(sev), scope=scope, mod=mod)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--dataset', required=True)
    p.add_argument('--image', required=True, help='file name inside the split')
    p.add_argument('--case', required=True)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--out_dir', required=True)
    a = p.parse_args()

    cfg = Config.fromfile(a.config)
    init_default_scope('mmseg')
    cfg.model.pop('pretrained', None)
    model = MODELS.build(cfg.model)
    load_checkpoint(model, a.checkpoint, map_location='cpu', strict=False)
    model = model.cuda().eval()

    ds_cfg = cfg.val_dataloader.dataset
    root = ds_cfg.data_root
    pipe = ds_cfg.pipeline
    pre = [TRANSFORMS.build(t) for t in pipe[:-1]]
    pack = TRANSFORMS.build(pipe[-1])
    ann_cfg = next(t for t in pipe if t['type'] == 'LoadAnnotations')

    rgb_dir = os.path.join(root, ds_cfg.data_prefix['img_path'])
    lab_dir = os.path.join(root, ds_cfg.data_prefix['seg_map_path'])
    results = dict(img_path=os.path.join(rgb_dir, a.image),
                   seg_map_path=os.path.join(lab_dir, a.image))
    results['reduce_zero_label'] = ann_cfg.get('reduce_zero_label', False)
    results['seg_fields'] = []
    for fn in pre:
        results = fn(results)
    img_clean = results['img']
    gt = np.asarray(results.get('gt_seg_map'))

    case = parse_case(a.case)

    # ---- observed input -------------------------------------------------
    thermal_clean = img_clean[..., 3:4]
    if case is None:
        img_obs = img_clean
        damage = np.zeros(img_clean.shape[:2], np.uint8)
    else:
        if isinstance(case, str):
            spec = recipe([operation(case, 1,
                                     stable_seed(a.seed, 'viz', a.image, case))])
        else:
            spec = recipe([operation(
                case['op'], case['sev'],
                stable_seed(a.seed, 'viz', a.image, a.case, case['op']),
                scope=case['scope'])])
        rgb2, t2 = apply_pair(np.ascontiguousarray(img_clean[..., :3]),
                              np.ascontiguousarray(thermal_clean), spec)
        img_obs = np.concatenate([rgb2, np.repeat(t2, 3, axis=2)], axis=2)
        damage = (np.abs(img_obs.astype(np.int16)
                         - img_clean.astype(np.int16)).sum(2) > 0
                  ).astype(np.uint8)

    # ---- hooks ----------------------------------------------------------
    cap = {}
    bb = model.backbone

    def hook_backbone(m, args, out):
        if isinstance(out, dict):
            for k in ('alpha', 'q_rgb', 'q_t', 'gate', 'anchor_map', 'grid'):
                v = out.get(k)
                cap[k] = v.detach().float().cpu().numpy() if torch.is_tensor(v) else v

    def hook_fusion(m, args, kwargs):
        cap['F_R'] = args[0].detach().float().cpu().numpy()
        cap['F_T'] = args[1].detach().float().cpu().numpy()

    def hook_norm(m, args, out):
        cap['F_deep'] = out.detach().float().cpu().numpy()

    handles = [bb.register_forward_hook(hook_backbone)]
    fusion = getattr(bb, 'fusion', None)
    if fusion is not None:
        handles.append(fusion.register_forward_pre_hook(hook_fusion,
                                                        with_kwargs=True))
    for name in ('final_norm',):
        mod = getattr(bb, name, None)
        if mod is not None:
            handles.append(mod.register_forward_hook(hook_norm))

    # ---- forward --------------------------------------------------------
    r = dict(results)
    r['img'] = img_obs
    r = pack(r)
    with torch.no_grad():
        out = model.test_step(pseudo_collate([r]))
    for h in handles:
        h.remove()

    logits = out[0].seg_logits.data
    logits = F.interpolate(logits[None].float(), size=gt.shape[-2:],
                           mode='bilinear', align_corners=False)
    pred = logits.argmax(1)[0].cpu().numpy().astype(np.uint8)

    # ---- save -----------------------------------------------------------
    os.makedirs(a.out_dir, exist_ok=True)
    np.save(os.path.join(a.out_dir, 'rgb_clean.npy'), img_clean[..., :3])
    np.save(os.path.join(a.out_dir, 'thermal_clean.npy'), thermal_clean)
    np.save(os.path.join(a.out_dir, 'rgb_observed.npy'), img_obs[..., :3])
    np.save(os.path.join(a.out_dir, 'thermal_observed.npy'), img_obs[..., 3:4])
    np.save(os.path.join(a.out_dir, 'damage_mask.npy'), damage)
    np.save(os.path.join(a.out_dir, 'gt.npy'), gt)
    np.save(os.path.join(a.out_dir, 'pred.npy'), pred)

    q = {k: cap[k] for k in ('q_rgb', 'q_t', 'alpha', 'gate') if k in cap}
    np.savez_compressed(os.path.join(a.out_dir, 'quality.npz'),
                        grid=np.asarray(cap.get('grid', (0, 0))), **q)
    f = {k: cap[k] for k in ('F_R', 'F_T', 'F_deep') if k in cap}
    if 'anchor_map' in cap:
        f['F_fused'] = cap['anchor_map']
    np.savez_compressed(os.path.join(a.out_dir, 'features.npz'), **f)

    meta = dict(dataset=a.dataset, image=a.image, case=a.case, seed=a.seed,
                config=a.config, checkpoint=a.checkpoint,
                input_size=list(img_obs.shape[:2]),
                grid=cap.get('grid'), has_quality=sorted(q), has_feature=sorted(f))
    with open(os.path.join(a.out_dir, 'metadata.json'), 'w') as fh:
        json.dump(meta, fh, indent=1)
    print(json.dumps(meta, default=str), flush=True)


if __name__ == '__main__':
    main()
