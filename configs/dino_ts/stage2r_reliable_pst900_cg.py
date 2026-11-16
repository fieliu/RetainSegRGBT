# Stage 2R on PST900 with per-channel reliability gating.
#
# The only change against stage2r_reliable_pst900.py is the fusion form:
#
#   convex   anchor = a_r * r + a_t * t                  (a_r + a_t = 1)
#   gated    anchor = g_r(x_r, q_r) * r + g_t(x_t, q_t) * t   (independent)
#
# with g = sigmoid(mlp([x, q]) + logit(q)).  Zero-init on the mlp's last layer
# makes the gated form start EXACTLY at the trained convex behaviour (g == q),
# so any change is attributable to what the gate learns rather than to the
# re-parameterisation.  Degradation policy is untouched, so this isolates the
# fusion form.
_base_ = ['stage2r_reliable_pst900.py']

model = dict(backbone=dict(reliable_fusion=dict(channel_gate=True)))
