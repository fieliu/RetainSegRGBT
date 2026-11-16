"""Pure-PyTorch targets/reductions for reliability and unimodal distillation.

Class-balanced targets are task-support scores, not calibrated pixel-accuracy
probabilities or local IoU. Observation masks are training targets only.
"""
import torch
import torch.nn.functional as F


def valid_labels(gt, num_classes, ignore_index=255):
    return (gt != ignore_index) & (gt >= 0) & (gt < num_classes)


@torch.no_grad()
def token_quality_targets(scores, gt, patch, observation=None,
                          reference_scores=None, target_mode='class_balanced',
                          metric='accuracy', min_class_pixels=4.0,
                          min_reference=0.1, ignore_index=255):
    """Return targets [B,N], class fractions [B,C,N], and valid tokens.

    Per-class score inside a patch, with small class fragments downweighted by
    min(n_c / min_class_pixels, 1).  Explicitly deleted pixels provide zero
    evidence even when a black-image teacher guesses their label.

    ``metric`` picks the per-class score: 'accuracy' (|pred n gt| / |gt|) or
    'iou' (|pred n gt| / |pred u gt|, which additionally penalises the teacher
    over-painting a class).

    ``reference_scores`` switches on NORMALISATION:

        y = metric(degraded input) / metric(clean input)

    Without it the target is the raw score on whatever input was passed, and it
    is confounded by task difficulty: a hard region (small object, thin pole)
    scores low for the teacher whether or not it was degraded, so y mixes "this
    region is hard" with "this modality lost capability".  Dividing by the same
    teacher's score on the CLEAN input removes the difficulty factor -- a hard
    region has a low numerator AND a low denominator, so the ratio still says
    only how much of this modality's capability survived.

    The denominator must be the same teacher on the clean input, not the fused
    model: this is a per-modality statement.

    Classes whose clean-input score is below ``min_reference`` are dropped
    (support 0) -- the ratio is meaningless where the teacher never worked.
    """
    if target_mode not in ('class_balanced', 'pixel_accuracy'):
        raise ValueError('Unknown quality target mode: %s' % target_mode)
    if metric not in ('accuracy', 'iou'):
        raise ValueError('Unknown quality metric: %s' % metric)
    if patch <= 0 or min_class_pixels <= 0:
        raise ValueError('Patch and min_class_pixels must be positive')
    if scores.ndim != 4 or gt.shape != scores.shape[:1] + scores.shape[2:]:
        raise ValueError('Teacher scores and GT must share a pixel grid')
    if gt.shape[-2] % patch or gt.shape[-1] % patch:
        raise ValueError('GT sides must be divisible by the patch size')
    if reference_scores is not None and reference_scores.shape != scores.shape:
        raise ValueError('Reference scores must match the teacher score shape')
    valid = valid_labels(gt, scores.shape[1], ignore_index)
    evidence = valid.float()
    if observation is not None:
        if observation.shape != gt.shape:
            raise ValueError('Observation mask must have shape [B,H,W]')
        evidence = evidence * observation.float().clamp(0, 1)

    def pool(x):
        return F.avg_pool2d(x.unsqueeze(1).float(), patch, patch).flatten(1)

    mass = torch.stack([pool((gt == c) & valid)
                        for c in range(scores.shape[1])], dim=1)
    total = mass.sum(1)
    fractions = mass / total.unsqueeze(1).clamp_min(1e-12)

    def per_class_score(score_map, ev):
        """[B,C,N] per-class accuracy or IoU inside each patch.

        `ev` is passed in rather than closed over: the observation mask says
        "THIS VIEW deleted these pixels", which is true of the degraded input
        but NOT of the clean reference.  Masking the reference too would drive
        its score to 0, drop the class via min_reference, and silently exclude
        the token from the loss -- so a deleted modality would never be trained
        towards 0, it would just stop being supervised at all.
        """
        pred = score_map.argmax(1)
        hit = (pred == gt).float() * ev
        num, den = [], []
        for c in range(score_map.shape[1]):
            members = (gt == c) & valid
            inter = pool(hit * members)
            # The denominator counts ALL valid pixels of the class, so masking
            # the evidence (deleted modality) drives the score to 0 instead of
            # amplifying whichever few pixels survived.
            if metric == 'iou':
                den.append(pool(members)
                           + pool(((pred == c) & valid).float()) - inter)
            else:
                den.append(pool(members))
            num.append(inter)
        return (torch.stack(num, 1) / torch.stack(den, 1).clamp_min(1e-12))

    score = per_class_score(scores, evidence)
    if reference_scores is None:
        support_ok = torch.ones_like(score)
    else:
        ref = per_class_score(reference_scores, valid.float())
        support_ok = (ref >= min_reference).float()
        # A degraded run can beat the clean one by chance; the ratio is a
        # fraction of capability, so cap it at 1.
        score = (score / ref.clamp_min(1e-12)).clamp(0, 1)

    if target_mode == 'pixel_accuracy':
        pred = scores.argmax(1)
        target = (pool(((pred == gt).float() * evidence))
                  / total.clamp_min(1e-12))
    else:
        support = ((mass * (patch * patch) / min_class_pixels).clamp(max=1)
                   * support_ok)
        target = (score * support).sum(1) / support.sum(1).clamp_min(1e-12)
    # A token whose every class was dropped carries no supervision.
    valid_tokens = (total > 0) & (support_ok.sum(1) > 0)
    return target.clamp(0, 1), fractions, valid_tokens


def quality_regression(q, target, fractions, valid_tokens,
                       reduction='class_balanced'):
    """Balance errors across GT classes, using fractional patch membership."""
    if q.shape != target.shape or valid_tokens.shape != q.shape:
        raise ValueError('Quality prediction/target shapes differ')
    error = (q.float() - target.float()).square()
    if reduction == 'pixel':
        return (error * valid_tokens).sum() / valid_tokens.sum().clamp_min(1)
    if reduction != 'class_balanced':
        raise ValueError('Unknown quality reduction: %s' % reduction)
    class_mass = fractions.float() * valid_tokens.unsqueeze(1)
    count = class_mass.sum(dim=(0, 2))
    loss_c = ((class_mass * error.unsqueeze(1)).sum(dim=(0, 2))
              / count.clamp_min(1e-12))
    present = count > 0
    return (loss_c * present).sum() / present.sum().clamp_min(1)


def normalize_class_scores(scores, eps=1e-6):
    """Mask2Former class-score maps are nonnegative scores, not logits."""
    values = scores.float().clamp_min(eps)
    return values / values.sum(1, keepdim=True)


def unimodal_distillation(student_scores, teacher_scores, gt,
                          reduction='class_balanced', teacher_filter='correct',
                          ignore_index=255):
    """GT-stratified KL; exclude padding and optionally incorrect teachers.

    Each class denominator counts all its valid GT pixels. Filtering wrong
    teacher predictions therefore reduces its contribution rather than
    amplifying a handful of surviving pixels. Absent classes are excluded.
    """
    if student_scores.shape != teacher_scores.shape:
        raise ValueError('Student and teacher score-map shapes differ')
    if gt.shape != student_scores.shape[:1] + student_scores.shape[2:]:
        raise ValueError('Distillation and GT grids differ')
    if reduction not in ('class_balanced', 'pixel'):
        raise ValueError('Unknown KD reduction: %s' % reduction)
    if teacher_filter not in ('correct', 'none'):
        raise ValueError('Unknown teacher filter: %s' % teacher_filter)
    p_s = normalize_class_scores(student_scores)
    p_t = normalize_class_scores(teacher_scores.detach())
    kl = F.kl_div(p_s.log(), p_t, reduction='none').sum(1)
    valid = valid_labels(gt, p_s.shape[1], ignore_index)
    correct = teacher_scores.argmax(1) == gt
    active = valid & correct if teacher_filter == 'correct' else valid
    losses, counts = [], []
    for c in range(p_s.shape[1]):
        members = valid & (gt == c)
        count = members.sum()
        losses.append((kl * members * active).sum() / count.clamp_min(1))
        counts.append(count)
    loss_c, count_c = torch.stack(losses), torch.stack(counts)
    if reduction == 'class_balanced':
        present = count_c > 0
        loss = (loss_c * present).sum() / present.sum().clamp_min(1)
    else:
        loss = (kl * active).sum() / valid.sum().clamp_min(1)
    diagnostics = dict(kd_valid_pixels=valid.sum().detach().float(),
                       kd_teacher_correct_fraction=(valid & correct).sum().float()
                       / valid.sum().clamp_min(1))
    diagnostics.update({'kd_class_%d' % c: v.detach()
                        for c, v in enumerate(loss_c)})
    return loss, diagnostics
