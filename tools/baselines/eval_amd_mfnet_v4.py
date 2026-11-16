"""AMDANet / MFNet benchmark, RGBT-C v4 protocol (mirror of bench_v4.py).

levels : 2/3/5
scopes : global, scatter
targets: rgb only, t only
pools  : id (TRAIN_CORRUPTIONS), ood (HELDOUT_CORRUPTIONS)
plus   : clean, rgb_missing, t_missing

Case ids and per-(image, case) seeds are identical to bench_v4.py, so the
two models see exactly the same degraded pixels.
"""
import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch

sys.path.insert(0, '/root/autodl-tmp/code/QWSEG')
from rgbt_c.protocol import (HELDOUT_CORRUPTIONS, MISSING, RGB_CORRUPTIONS,
                             TRAIN_CORRUPTIONS, apply_pair, operation, recipe,
                             stable_seed)

from dataloader.dataloader import get_val_loader
from dataloader.RGBXDataset import RGBXDataset
from models.model import AMDANet
from utils.metric import hist_info

LEVELS = [2, 3, 5]
SCOPES = ['global', 'scatter']

p = argparse.ArgumentParser()
p.add_argument('--dataset_path', default='/root/autodl-tmp/data/MFNet_split')
p.add_argument('--checkpoint', required=True)
p.add_argument('--limit', type=int, default=None)
p.add_argument('--seed', type=int, default=42)
p.add_argument('--ood', action='store_true')
p.add_argument('--mode', default='sample', choices=['sample', 'enumerate'])
p.add_argument('--input-size', default=None)
p.add_argument('--out', required=True)
a = p.parse_args()

cfg = SimpleNamespace(
    dataset_path=a.dataset_path, rgb_folder='Visible', x_folder='Infrared',
    label_folder='Label', rgb_format='.png', x_format='.png',
    label_format='.png', x_is_single_channel=True, gt_transform=False,
    class_names=['unlabeled', 'car', 'person', 'bike', 'curve', 'car_stop',
                 'guardrail', 'color_cone', 'bump'],
    num_classes=9, decoder_embed_dim=512, bn_eps=0.01, bn_momentum=0.01,
    pretrained_backbone=None, num_workers=4, image_height=480,
    image_width=640, eval_size=[480, 640], train_scale_array=None,
    batch_size=1)
engine = SimpleNamespace(distributed=False)
loader = get_val_loader(cfg, engine, RGBXDataset)[0]

model = AMDANet(cfg=cfg, criterion=None)
ck = torch.load(a.checkpoint, map_location='cpu')
model.load_state_dict(ck.get('model', ck), strict=False)
model = model.cuda().eval()


def build_cases(with_ood, mode):
    cases = [('clean', None), ('rgb_missing', 'rgb_missing'),
             ('t_missing', 't_missing')]
    groups = [('id', [c for c in TRAIN_CORRUPTIONS if c not in MISSING])]
    if with_ood:
        groups.append(('ood', list(HELDOUT_CORRUPTIONS)))
    for tag, pool in groups:
        for scope in SCOPES:
            for lv in LEVELS:
                if mode == 'enumerate':
                    for op in pool:
                        mod = 'rgb' if op in RGB_CORRUPTIONS else 't'
                        cases.append(
                            ('%s/%s/%d/%s/%s' % (tag, scope, lv, mod, op),
                             dict(op=op, sev=lv, scope=scope, mod=mod)))
                else:
                    cases.append(('%s/%s/%d' % (tag, scope, lv),
                                  dict(draw=pool, sev=lv, scope=scope)))
    return cases


def resolve_spec(spec, image_id, cid, seed):
    if not isinstance(spec, dict) or 'draw' not in spec:
        return spec
    pool = spec['draw']
    rng = np.random.default_rng(stable_seed(seed, 'v4draw', image_id, cid))
    op = str(pool[int(rng.integers(0, len(pool)))])
    return dict(op=op, sev=spec['sev'], scope=spec['scope'])


CASES = build_cases(a.ood, a.mode)


def make_recipe(spec, image_id, cid, seed):
    if isinstance(spec, str) and spec in MISSING:
        return recipe([operation(spec, 1,
                                 stable_seed(seed, 'v4', image_id, cid, spec))])
    return recipe([operation(spec['op'], spec['sev'],
                             stable_seed(seed, 'v4', image_id, cid, spec['op']),
                             scope=spec['scope'])])


def corrupt(rgb, thermal, spec, image_id, cid):
    rgb2, t2 = apply_pair(rgb, thermal,
                          make_recipe(spec, image_id, cid, a.seed))
    return rgb2, t2


def forward(xr, xt):
    with torch.no_grad():
        score, _, _, _ = model(xr, xt)
    return torch.exp(score[0]).permute(1, 2, 0)


def score_from(score, label):
    pred = cv2.resize(score.cpu().numpy(),
                      (cfg.eval_size[1], cfg.eval_size[0]),
                      interpolation=cv2.INTER_LINEAR).argmax(2)
    gt = cv2.resize(label, (cfg.eval_size[1], cfg.eval_size[0]),
                    interpolation=cv2.INTER_NEAREST)
    return pred, gt


def mious(h):
    vals = []
    for c in range(h.shape[0]):
        inter = h[c, c]
        union = h[c, :].sum() + h[:, c].sum() - inter
        if union > 0:
            vals.append(inter / union)
    return float(np.mean(vals))


hists = {cid: np.zeros((cfg.num_classes, cfg.num_classes))
         for cid, _ in CASES}
n = len(loader) if a.limit is None else min(a.limit, len(loader))
print('images', n, 'cases/image', len(CASES), flush=True)

with torch.no_grad():
    for idx, data in enumerate(loader):
        if idx >= n:
            break
        img = data['data'].cuda(non_blocking=True)
        modal = data['modal_x'].cuda(non_blocking=True)
        name = str(data['fn'][0]).split('/')[-1]
        label = data['label'].squeeze().cpu().numpy()

        rgb = np.rint(img[0].permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255)
        rgb = rgb.astype(np.uint8)
        thr = np.rint(modal[0, :1].permute(1, 2, 0).cpu().numpy() * 255)
        thr = thr.clip(0, 255).astype(np.uint8)
        h, w = rgb.shape[:2]

        def run(r8, t8):
            xr = torch.from_numpy(
                r8.transpose(2, 0, 1)[None].copy()).float().cuda() / 255.0
            x1 = torch.from_numpy(
                t8.transpose(2, 0, 1)[None].copy()).float().cuda() / 255.0
            xt = x1.repeat(1, 3, 1, 1) if modal.shape[1] == 3 else x1
            return hist_info(cfg.num_classes,
                             *score_from(forward(xr, xt), label))[0]

        hists['clean'] += hist_info(cfg.num_classes,
                                    *score_from(forward(img, modal),
                                                label))[0]
        for cid in ('rgb_missing', 't_missing'):
            r8, t8 = corrupt(rgb, thr, cid, name, cid)
            hists[cid] += run(r8, t8)
        for cid, spec in CASES[3:]:
            r8, t8 = corrupt(rgb, thr, resolve_spec(spec, name, cid, a.seed),
                             name, cid)
            hists[cid] += run(r8, t8)
        if (idx + 1) % 20 == 0:
            print('  %d/%d images' % (idx + 1, n), flush=True)

per_case = {cid: mious(h) for cid, h in hists.items()}
summary = {'clean': per_case['clean'],
           'rgb_missing': per_case['rgb_missing'],
           't_missing': per_case['t_missing']}
for tag in ('id', 'ood'):
    sel = [c for c in per_case if c.startswith(tag + '/')]
    if not sel:
        continue
    for scope in SCOPES:
        for lv in LEVELS:
            for mod in ('rgb', 't'):
                sub = [c for c in sel
                       if ('/%s/%s/%d/%s/' % (tag, scope, lv, mod)) in c]
                if sub:
                    summary['%s/%s/%d/%s' % (tag, scope, lv, mod)] = \
                        float(np.mean([per_case[c] for c in sub]))
    summary['%s/mean' % tag] = float(np.mean([per_case[c] for c in sel]))
summary['per_case'] = per_case
summary['args'] = dict(checkpoint=a.checkpoint, n_images=n, seed=a.seed,
                       levels=LEVELS, scopes=SCOPES, ood=a.ood)
Path(a.out).write_text(json.dumps(summary, indent=1))
print(json.dumps({k: v for k, v in summary.items() if k != 'per_case'},
                 indent=1))
