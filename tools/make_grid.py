"""Build a multi-row qualitative grid.

Rows are export directories (one per condition); columns are
RGB clean | RGB input | Thermal | GT | [AMDANet] | Ours | q_R | alpha_R.

Usage:
  python make_grid.py --n_classes 13 --amd_shift \
      --row clean:/root/exports/fmb/clean \
      --row "rgb scatter L3:/root/exports/fmb/rgb_scatter3" ... --out fig.png
"""
import argparse
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

PALETTE = {
    5: ['#000000', '#e6194b', '#3cb44b', '#4363d8', '#f58231'],
    9: ['#000000', '#e6194b', '#3cb44b', '#4363d8', '#f58231',
        '#911eb4', '#42d4f4', '#f032e6', '#bfef45'],
    15: ['#000000', '#e6194b', '#3cb44b', '#4363d8', '#f58231',
         '#911eb4', '#42d4f4', '#f032e6', '#bfef45', '#fabed4',
         '#469990', '#dcbeff', '#9a6324', '#fffac8', '#800000'],
}


def _fallback(n):
    base = plt.get_cmap('tab20').colors + plt.get_cmap('tab20b').colors
    out = ['#000000'] + ['#%02x%02x%02x' % tuple(int(255 * v) for v in c[:3])
                         for c in base]
    while len(out) < n:
        out += out[1:]
    return out[:n]


def colourise(idx, n):
    names = PALETTE.get(n) or _fallback(n)
    pal = np.array([[int(c[i:i + 2], 16) for i in (1, 3, 5)]
                    for c in names], np.uint8)
    out = np.zeros(idx.shape + (3,), np.uint8)
    ok = (idx >= 0) & (idx < n)
    out[ok] = pal[idx[ok]]
    return out


def shift_amd(idx):
    # raw 1..12 -> 0..11, raw 14 (Pole) -> 12, raw 13 (Bicycle) -> ignore
    out = np.full(idx.shape, 255, np.uint8)
    m = (idx >= 1) & (idx <= 12)
    out[m] = idx[m] - 1
    out[idx == 14] = 12
    return out


def upsample(a, shape):
    """nearest upsample a 2-D array to shape."""
    h, w = a.shape
    H, W = shape
    yi = (np.arange(H) * h // H).clip(0, h - 1)
    xi = (np.arange(W) * w // W).clip(0, w - 1)
    return a[yi][:, xi]


p = argparse.ArgumentParser()
p.add_argument('--row', action='append', required=True,
               help='label:dir')
p.add_argument('--n_classes', type=int, required=True)
p.add_argument('--amd_shift', action='store_true')
p.add_argument('--out', required=True)
p.add_argument('--width', type=float, default=2.1)
a = p.parse_args()

rows = []
for spec in a.row:
    label, d = spec.split(':', 1)
    rgb = np.load(os.path.join(d, 'rgb_observed.npy'))
    rgbc = np.load(os.path.join(d, 'rgb_clean.npy'))
    thr = np.load(os.path.join(d, 'thermal_observed.npy'))[..., 0]
    gt = np.load(os.path.join(d, 'gt.npy'))
    ours = np.load(os.path.join(d, 'pred.npy'))
    ap = os.path.join(d, 'pred_amdanet.npy')
    amd = np.load(ap) if os.path.exists(ap) else None
    if amd is not None and a.amd_shift:
        amd = shift_amd(amd)
    qpath = os.path.join(d, 'quality.npz')
    qr = qt = None
    if os.path.exists(qpath):
        z = np.load(qpath)
        qr = z['q_rgb'].reshape(z['grid'])
        qt = z['q_t'].reshape(z['grid'])
    rows.append(dict(label=label, rgb=rgb, rgbc=rgbc, thr=thr, gt=gt,
                     ours=ours, amd=amd, qr=qr, qt=qt))

has_amd = any(r['amd'] is not None for r in rows)
has_q = any(r['qr'] is not None for r in rows)
cols = ['RGB', 'Thermal', 'GT']
if has_amd:
    cols.append('AMDANet')
cols.append('Ours')
if has_q:
    cols += ['q (RGB)', 'q (Thermal)']

fig, axes = plt.subplots(len(rows), len(cols),
                         figsize=(a.width * len(cols), 1.55 * len(rows)))
if len(rows) == 1:
    axes = axes[None]
for i, r in enumerate(rows):
    panels = [r['rgb'], r['thr'], colourise(r['gt'], a.n_classes)]
    if has_amd:
        panels.append(colourise(r['amd'], a.n_classes)
                      if r['amd'] is not None
                      else np.full(r['gt'].shape + (3,), 255, np.uint8))
    panels.append(colourise(r['ours'], a.n_classes))
    if has_q:
        for key in ('qr', 'qt'):
            panels.append(np.full(r['gt'].shape, np.nan) if r[key] is None
                          else upsample(r[key], r['gt'].shape))
    for j, pnl in enumerate(panels):
        ax = axes[i, j]
        if cols[j].startswith('q ('):
            ax.imshow(pnl, cmap='turbo', vmin=0.0, vmax=1.0)
        else:
            ax.imshow(pnl, cmap='gray' if pnl.ndim == 2 else None)
        if i == 0:
            ax.set_title(cols[j], fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        if j == 0:
            ax.set_ylabel(r['label'], fontsize=9)
fig.tight_layout(pad=0.3)

if has_q:
    q0 = cols.index('q (RGB)')
    sm = matplotlib.cm.ScalarMappable(
        cmap='turbo', norm=matplotlib.colors.Normalize(0.0, 1.0))
    cb = fig.colorbar(sm, ax=axes[:, q0:], orientation='horizontal',
                      fraction=0.03, pad=0.02, aspect=45)
    cb.set_ticks([0.0, 0.5, 1.0])
    cb.ax.tick_params(labelsize=8)
    cb.set_label('reliability q  (0 = unreliable, 1 = reliable)', fontsize=8)

fig.savefig(a.out, dpi=170, bbox_inches='tight')
print('wrote', a.out)
