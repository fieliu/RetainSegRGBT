"""RGBT-C v4 benchmark for PEAFusion (same cases/seeds as bench_v4.py).

Usage:
  python eval_pea_v4.py --config-file configs/MFdataset/swin_v2/swin_v2_base.yaml \
      --checkpoint /root/autodl-tmp/pretrain/peafusion/peafusion_base_MFNet.ckpt \
      --num-classes 9 --levels 2,3,5 --scopes global,scatter --ood \
      --out /root/v4_mf_pea.json
"""
import argparse
import json
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, '/root/autodl-tmp/code/PEAFusion')
sys.path.insert(0, '/root/autodl-tmp/code/QWSEG')

from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config

from models import MODELS
from models.config import add_peafusion_config
from models.mask2former import add_maskformer2_config
from dataloaders import build_dataset

from rgbt_c.protocol import (HELDOUT_CORRUPTIONS, MISSING, RGB_CORRUPTIONS,
                             TRAIN_CORRUPTIONS, apply_pair, operation, recipe,
                             stable_seed)

p = argparse.ArgumentParser()
p.add_argument('--config-file', required=True)
p.add_argument('--data-dir', required=True)
p.add_argument('--checkpoint', required=True)
p.add_argument('--num-classes', type=int, required=True)
p.add_argument('--levels', default='2,3,5')
p.add_argument('--scopes', default='global,scatter')
p.add_argument('--ood', action='store_true')
p.add_argument('--mode', default='sample', choices=['sample', 'enumerate'])
p.add_argument('--seed', type=int, default=42)
p.add_argument('--fmb_remap', action='store_true')
p.add_argument('--limit', type=int, default=None)
p.add_argument('--out', required=True)
a = p.parse_args()

LEVELS = [int(x) for x in a.levels.split(',')]
SCOPES = a.scopes.split(',')


def build_cases():
    cases = [('clean', None), ('rgb_missing', 'rgb_missing'),
             ('t_missing', 't_missing')]
    groups = [('id', [c for c in TRAIN_CORRUPTIONS if c not in MISSING])]
    if a.ood:
        groups.append(('ood', list(HELDOUT_CORRUPTIONS)))
    for tag, pool in groups:
        for sc in SCOPES:
            for lv in LEVELS:
                if a.mode == 'enumerate':
                    for op in pool:
                        mod = 'rgb' if op in RGB_CORRUPTIONS else 't'
                        cases.append(('%s/%s/%d/%s/%s' % (tag, sc, lv, mod, op),
                                      dict(op=op, sev=lv, scope=sc, mod=mod)))
                else:
                    cases.append(('%s/%s/%d' % (tag, sc, lv),
                                  dict(draw=pool, sev=lv, scope=sc)))
    return cases


def resolve(spec, image_id, cid):
    if not isinstance(spec, dict) or 'draw' not in spec:
        return spec
    pool = spec['draw']
    rng = np.random.default_rng(stable_seed(a.seed, 'v4draw', image_id, cid))
    return dict(op=str(pool[int(rng.integers(0, len(pool)))]),
                sev=spec['sev'], scope=spec['scope'])


def make_recipe(spec, image_id, cid):
    if isinstance(spec, str) and spec in MISSING:
        return recipe([operation(spec, 1,
                                 stable_seed(a.seed, 'v4', image_id, cid, spec))])
    return recipe([operation(spec['op'], spec['sev'],
                             stable_seed(a.seed, 'v4', image_id, cid, spec['op']),
                             scope=spec['scope'])])


def to_ours(a):
    """FMB raw ids -> our 13-class protocol (raw 13/Bicycle dropped)."""
    out = np.full(a.shape, 255, np.uint8)
    m = (a >= 1) & (a <= 12)
    out[m] = a[m] - 1
    out[a == 14] = 12
    return out


def add_hist(hist, pred, gt, n):
    pr = np.asarray(pred).reshape(-1)
    g = np.asarray(gt).reshape(-1)
    if a.fmb_remap:
        pr = to_ours(pr.astype(np.uint8)).astype(np.int64)
        g = to_ours(g.astype(np.uint8)).astype(np.int64)
    # Predictions outside the evaluated 13 classes (Bicycle = raw 13) are
    # dropped rather than counted; this is conservative w.r.t. our model.
    m = (g >= 0) & (g < n) & (pr >= 0) & (pr < n)
    hist += np.bincount(n * g[m].astype(np.int64) + pr[m],
                        minlength=n * n).reshape(n, n)


def mious(h):
    inter = np.diag(h).astype(np.float64)
    union = (h.sum(1) + h.sum(0) - np.diag(h)).astype(np.float64)
    v = union > 0
    return inter[v] / union[v]


cfg = get_cfg()
add_deeplab_config(cfg)
add_maskformer2_config(cfg)
add_peafusion_config(cfg)
cfg.merge_from_file(a.config_file)
cfg.DATASETS.DIR = a.data_dir
cfg.MODEL.BACKBONE.PRETRAINED = ''   # full weights come from the checkpoint
cfg.freeze()

ds = build_dataset(cfg)['test']
model = MODELS.build(name=cfg.MODEL.META_ARCHITECTURE, option=cfg)
ck = torch.load(a.checkpoint, weights_only=False)
model.load_state_dict(ck['state_dict'])
model = model.cuda().eval()

N = a.num_classes
cases = build_cases()
hists = {c: np.zeros((N, N)) for c, _ in cases}
n_img = len(ds) if a.limit is None else min(a.limit, len(ds))
print('images', n_img, 'cases/image', len(cases), flush=True)

for i in range(n_img):
    s = ds[i]
    img = s['image'].float()                     # C,H,W  (0-255)
    gt = s['sem_seg_gt'].cpu().numpy()
    name = str(s['name'])
    rgb = img[:3].permute(1, 2, 0).cpu().numpy().clip(0, 255).astype(np.uint8)
    thr = img[3:4].permute(1, 2, 0).cpu().numpy().clip(0, 255).astype(np.uint8)
    for cid, spec in cases:
        sp = resolve(spec, name, cid)
        if sp is None:
            new = img
        else:
            r2, t2 = apply_pair(np.ascontiguousarray(rgb),
                                np.ascontiguousarray(thr),
                                make_recipe(sp, name, cid))
            new = torch.from_numpy(
                np.concatenate([r2, t2], axis=2).transpose(2, 0, 1).copy()
            ).float().cuda()
        sample = {k: (v.cuda() if torch.is_tensor(v) else v)
                  for k, v in s.items() if k != 'image'}
        sample['image'] = new                       # C,H,W
        if 'sem_seg_gt' in sample:
            sample['sem_seg_gt'] = sample['sem_seg_gt'].long()
        with torch.no_grad():
            out = model.network([sample])
        # inference may return (results, attn), a list of dicts, a dict or a
        # tensor depending on the branch; normalise to a C,H,W tensor.
        while isinstance(out, (tuple, list)):
            out = out[0]
        if isinstance(out, dict):
            out = out.get('sem_seg', list(out.values())[0])
        if out.dim() == 4:
            out = out[0]
        out = F.interpolate(out[None].float(), size=gt.shape[-2:],
                            mode='bilinear', align_corners=False)
        add_hist(hists[cid], out.argmax(1)[0].cpu().numpy(), gt, N)
    if (i + 1) % 25 == 0:
        print('  %d/%d' % (i + 1, n_img), flush=True)

per_case = {c: float(np.mean(mious(h))) for c, h in hists.items()}
summary = dict(clean=per_case['clean'],
               rgb_missing=per_case['rgb_missing'],
               t_missing=per_case['t_missing'])
for tag in ('id', 'ood'):
    sel = [c for c in per_case if c.startswith(tag + '/')]
    if not sel:
        continue
    for sc in SCOPES:
        for lv in LEVELS:
            pooled = '%s/%s/%d' % (tag, sc, lv)
            if pooled in per_case:
                summary[pooled] = per_case[pooled]
    summary['%s/mean' % tag] = float(np.mean([per_case[c] for c in sel]))
summary['per_case'] = per_case
summary['args'] = dict(config=a.config_file, checkpoint=a.checkpoint,
                       n_images=n_img, seed=a.seed, levels=LEVELS,
                       scopes=SCOPES)
with open(a.out, 'w') as f:
    json.dump(summary, f, indent=1)
print(json.dumps({k: round(v, 2) for k, v in summary.items()
                  if k not in ('per_case', 'args')}, indent=1))
