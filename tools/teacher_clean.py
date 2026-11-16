"""Clean-only mIoU for a single-modality teacher.

A single-modality teacher only has a meaningful number on its OWN modality
with clean input, so this runs the validation split once and reports mIoU.

Usage:
  python teacher_clean.py --config CFG --checkpoint CK --num-classes 13
"""
import argparse
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

p = argparse.ArgumentParser()
p.add_argument('--config', required=True)
p.add_argument('--checkpoint', required=True)
p.add_argument('--num-classes', type=int, required=True)
p.add_argument('--limit', type=int, default=None)
p.add_argument('--out', required=True)
a = p.parse_args()

cfg = Config.fromfile(a.config)
init_default_scope('mmseg')
cfg.model.pop('pretrained', None)
model = MODELS.build(cfg.model)
load_checkpoint(model, a.checkpoint, map_location='cpu', strict=False)
model = model.cuda().eval()

ds_cfg = cfg.val_dataloader.dataset
pipe = ds_cfg.pipeline
pre = [TRANSFORMS.build(t) for t in pipe[:-1]]
pack = TRANSFORMS.build(pipe[-1])
ann = next(t for t in pipe if t['type'] == 'LoadAnnotations')

root = ds_cfg.data_root
img_dir = '{}/{}'.format(root, ds_cfg.data_prefix['img_path'])
lab_dir = '{}/{}'.format(root, ds_cfg.data_prefix['seg_map_path'])
import os
names = sorted(n for n in os.listdir(img_dir) if n.endswith('.png'))
if a.limit:
    names = names[:a.limit]

n = a.num_classes
hist = np.zeros((n, n))
for i, name in enumerate(names):
    res = dict(img_path='{}/{}'.format(img_dir, name),
               seg_map_path='{}/{}'.format(lab_dir, name))
    res['reduce_zero_label'] = ann.get('reduce_zero_label', False)
    res['seg_fields'] = []
    for fn in pre:
        res = fn(res)
    r = pack(res)
    with torch.no_grad():
        out = model.test_step(pseudo_collate([r]))
    gt = r['data_samples'].gt_sem_seg.data
    lg = F.interpolate(out[0].seg_logits.data[None].float(),
                       size=gt.shape[-2:], mode='bilinear', align_corners=False)
    pr = lg.argmax(1)[0].reshape(-1).cpu().numpy()
    g = gt.reshape(-1).cpu().numpy()
    m = (g >= 0) & (g < n)
    hist += np.bincount(n * g[m].astype(np.int64) + pr[m],
                        minlength=n * n).reshape(n, n)
    if (i + 1) % 50 == 0:
        print('  %d/%d' % (i + 1, len(names)), flush=True)

inter = np.diag(hist).astype(np.float64)
union = (hist.sum(1) + hist.sum(0) - np.diag(hist)).astype(np.float64)
v = union > 0
miou = float(np.mean(inter[v] / union[v]))
print('clean mIoU = %.2f  (n=%d)' % (miou * 100, len(names)))
import json
json.dump(dict(clean=miou, n_images=len(names), config=a.config,
               checkpoint=a.checkpoint), open(a.out, 'w'), indent=1)
