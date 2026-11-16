"""RGBT-C v4 benchmark for our mmseg segmentor.

Protocol
  levels : severity 2/3/5
  scopes : global (whole frame), scatter (16x16 patch grid)
  targets: rgb only, t only      (single-modality; the imbalance axis)
  pools  : id  = TRAIN_CORRUPTIONS   (in-distribution)
           ood = HELDOUT_CORRUPTIONS (never trained on)
  plus   : clean, rgb_missing, t_missing

Every (image, case) draws its own deterministic seed, so the recipe is
reproducible and independent of evaluation order.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, '/root/autodl-tmp/code/QWSEG')

from mmengine.config import Config
from mmengine.dataset import pseudo_collate
from mmengine.registry import init_default_scope
from mmseg.registry import MODELS
from mmengine.runner.checkpoint import load_checkpoint
from mmseg.registry import DATASETS, TRANSFORMS

from rgbt_c.protocol import (HELDOUT_CORRUPTIONS, MISSING, RGB_CORRUPTIONS,
                             TRAIN_CORRUPTIONS, apply_pair, operation, recipe,
                             stable_seed)




def build_cases(levels, scopes, with_ood, mode='sample'):
    """sample   : one operator drawn per (scope, level, pool)  -> 15 cases
    enumerate: every operator evaluated separately           -> 69 cases
    """
    cases = [('clean', None), ('rgb_missing', 'rgb_missing'),
             ('t_missing', 't_missing')]
    seen = [c for c in TRAIN_CORRUPTIONS if c not in MISSING]
    groups = [('id', seen)]
    if with_ood:
        groups.append(('ood', list(HELDOUT_CORRUPTIONS)))
    for tag, pool in groups:
        for scope in scopes:
            for lv in levels:
                if mode == 'enumerate':
                    for op in pool:
                        mod = 'rgb' if op in RGB_CORRUPTIONS else 't'
                        cid = '%s/%s/%d/%s/%s' % (tag, scope, lv, mod, op)
                        cases.append((cid, dict(op=op, sev=lv, scope=scope,
                                                mod=mod)))
                else:
                    cid = '%s/%s/%d' % (tag, scope, lv)
                    cases.append((cid, dict(draw=pool, sev=lv, scope=scope)))
    return cases


def resolve_spec(spec, image_id, cid, seed):
    """Turn a draw spec into a concrete (op, severity, scope)."""
    if not isinstance(spec, dict) or 'draw' not in spec:
        return spec
    pool = spec['draw']
    rng = np.random.default_rng(stable_seed(seed, 'v4draw', image_id, cid))
    op = str(pool[int(rng.integers(0, len(pool)))])
    return dict(op=op, sev=spec['sev'], scope=spec['scope'],
                mod='rgb' if op in RGB_CORRUPTIONS else 't')


def make_recipe(spec, image_id, cid, seed):
    if isinstance(spec, str) and spec in MISSING:
        return recipe([operation(spec, 1,
                                 stable_seed(seed, 'v4', image_id, cid, spec))])
    return recipe([operation(spec['op'], spec['sev'],
                             stable_seed(seed, 'v4', image_id, cid, spec['op']),
                             scope=spec['scope'])])


def corrupt_image(img6, spec, image_id, cid, seed):
    """img6: HxWx6 (RGB+thermal repeated) or HxWx4 (RGB+thermal)."""
    ch = img6.shape[2]
    rgb = np.ascontiguousarray(img6[..., :3])
    thermal = np.ascontiguousarray(img6[..., 3:4])
    rgb2, t2 = apply_pair(rgb, thermal, make_recipe(spec, image_id, cid, seed))
    if ch == 4:
        return np.concatenate([rgb2, t2], axis=2)
    return np.concatenate([rgb2, np.repeat(t2, 3, axis=2)], axis=2)


def add_hist(hist, pred, gt):
    n = hist.shape[0]
    pred = np.asarray(pred).reshape(-1)
    gt = np.asarray(gt).reshape(-1)
    mask = (gt >= 0) & (gt < n)
    inds = n * gt[mask].astype(np.int64) + pred[mask]
    hist += np.bincount(inds, minlength=n ** 2).reshape(n, n)


def mious(hist, ignore=()):
    inter = np.diag(hist).astype(np.float64)
    union = (hist.sum(1) + hist.sum(0) - np.diag(hist)).astype(np.float64)
    valid = union > 0
    for c in ignore:
        if c < len(valid):
            valid[c] = False
    return inter[valid] / union[valid]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--levels', default='2,3,5')
    p.add_argument('--scopes', default='global,scatter')
    p.add_argument('--ood', action='store_true')
    p.add_argument('--mode', default='sample', choices=['sample', 'enumerate'])
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--limit', type=int, default=None)
    p.add_argument('--num-classes', type=int, default=None,
                   help='defaults to model.num_classes')
    p.add_argument('--ignore', default='',
                   help='comma list of class ids excluded from mIoU')
    p.add_argument('--out', required=True)
    a = p.parse_args()

    levels = [int(x) for x in a.levels.split(',')]
    scopes = a.scopes.split(',')
    ignore = [int(x) for x in a.ignore.split(',') if x != '']
    cases = build_cases(levels, scopes, a.ood, a.mode)

    cfg = Config.fromfile(a.config)
    init_default_scope('mmseg')
    num_classes = a.num_classes or cfg.model.get('num_classes', 5)
    # NOTE: mmseg's init_model() force-injects `pretrained=None` into the
    # segmentor config, which our custom segmentors reject. Build directly.
    cfg.model.pop('pretrained', None)
    model = MODELS.build(cfg.model)
    load_checkpoint(model, a.checkpoint, map_location='cpu', strict=False)
    model = model.cuda().eval()
    ds_cfg = cfg.val_dataloader.dataset
    root = ds_cfg.data_root
    rgb_dir = os.path.join(root, ds_cfg.data_prefix['img_path'])
    lab_dir = os.path.join(root, ds_cfg.data_prefix['seg_map_path'])
    if 'ann_file' in ds_cfg:                 # list-based split (MFNet)
        suffix = ds_cfg.get('img_suffix', '.png')
        with open(os.path.join(root, ds_cfg['ann_file'])) as f:
            names = [ln.strip() for ln in f if ln.strip()]
        data_list = [dict(img_path=os.path.join(rgb_dir, n + suffix),
                          seg_map_path=os.path.join(lab_dir, n + suffix))
                     for n in names]
    else:                                    # directory-based split
        data_list = [dict(img_path=os.path.join(rgb_dir, n),
                          seg_map_path=os.path.join(lab_dir, n))
                     for n in sorted(os.listdir(rgb_dir)) if n.endswith('.png')]
    if a.limit:
        data_list = data_list[:a.limit]
    pipe = ds_cfg.pipeline
    # Run every transform in the order the config declares (datasets differ:
    # PST900 load/ann/resize, MFNet load/resize/ann, FMB adds RemapLabels),
    # then inject the corruption immediately before PackSegInputs.
    pre = [TRANSFORMS.build(t) for t in pipe[:-1]]
    pack = TRANSFORMS.build(pipe[-1])
    ann_cfg = next(t for t in pipe if t['type'] == 'LoadAnnotations')

    hists = {cid: np.zeros((num_classes, num_classes)) for cid, _ in cases}
    print('images %d  cases/image %d  total %d' %
          (len(data_list), len(cases), len(data_list) * len(cases)), flush=True)

    for i, info in enumerate(data_list):
        image_id = str(info.get('img_path', i)).split('/')[-1]
        results = dict(info)
        results['reduce_zero_label'] = ann_cfg.get(
            'reduce_zero_label', ds_cfg.get('reduce_zero_label', False))
        results['seg_fields'] = []
        # Every declared transform, in order; corruption goes in just before
        # packing so it hits the network input grid the baseline also sees.
        for fn in pre:
            results = fn(results)
        base_img = results['img']
        for cid, spec in cases:
            r = dict(results)
            spec_i = resolve_spec(spec, image_id, cid, a.seed)
            r['img'] = base_img if spec_i is None else corrupt_image(
                base_img, spec_i, image_id, cid, a.seed)
            r = pack(r)
            data = pseudo_collate([r])
            with torch.no_grad():
                out = model.test_step(data)
            gt = r['data_samples'].gt_sem_seg.data
            logits = out[0].seg_logits.data
            logits = F.interpolate(logits[None].float(), size=gt.shape[-2:],
                                   mode='bilinear', align_corners=False)
            pred = logits.argmax(1)[0]
            add_hist(hists[cid], pred.cpu().numpy(), gt.cpu().numpy())
        if (i + 1) % 20 == 0:
            print('  %d/%d images' % (i + 1, len(data_list)), flush=True)

    per_case = {cid: float(np.mean(mious(h, ignore))) for cid, h in hists.items()}
    summary = {'clean': per_case['clean'],
               'rgb_missing': per_case['rgb_missing'],
               't_missing': per_case['t_missing']}
    for tag in ('id', 'ood'):
        sel = [c for c in per_case if c.startswith(tag + '/')]
        if not sel:
            continue
        for scope in scopes:
            for lv in levels:
                pooled = '%s/%s/%d' % (tag, scope, lv)
                if pooled in per_case:            # sample mode
                    summary[pooled] = per_case[pooled]
                    continue
                for mod in ('rgb', 't'):          # enumerate mode
                    sub = [c for c in sel
                           if ('/%s/%s/%d/%s/' % (tag, scope, lv, mod)) in c]
                    if sub:
                        summary['%s/%s/%d/%s' % (tag, scope, lv, mod)] = \
                            float(np.mean([per_case[c] for c in sub]))
        summary['%s/mean' % tag] = float(np.mean([per_case[c] for c in sel]))
    summary['per_case'] = per_case
    summary['args'] = dict(config=a.config, checkpoint=a.checkpoint,
                           levels=levels, scopes=scopes, seed=a.seed,
                           limit=a.limit, n_images=len(data_list))
    with open(a.out, 'w') as f:
        json.dump(summary, f, indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k != 'per_case'},
                     indent=1), flush=True)


if __name__ == '__main__':
    main()
