# Ablation: Stage 2C without whole-modality-missing training.
#
# Identical to stage2c_full_ema_mfnet.py except missing_prob is 0, so the
# degradation policy never zeroes an entire modality.  Everything else -- warm
# start, seed, schedule, EMA teacher, L_anchor -- is unchanged.
#
# Why this matters: rgb_missing improved the most of any case under Stage 2C
# (40.50 -> 52.12, +11.62), and "zero an entire sensor" is the one thing in the
# degradation policy that has no counterpart in ordinary image augmentation.
# If most of that +11.62 survives with missing_prob=0, the gain came from the
# generic corruptions; if it collapses, whole-modality dropout is the real
# contribution and is a cleaner story than the anchor consistency.
#
# Note that the multimodal model's thermal-only path is currently BELOW a
# dedicated Thermal-only model (52.12 vs 55.66 at epoch 35), so this ablation
# also bears on whether the missing-modality objective is closing that gap at
# all.
_base_ = ['stage2c_full_ema_mfnet.py']

model = dict(
    degradation=dict(missing_prob=0.0))
