import math

import torch
import torch.distributed as dist
import torch.nn.functional as F


class HardTokenSelector:
    def __init__(self, momentum=0.99, scale=0.8):
        self.momentum = momentum
        self.scale = scale
        self.mean = 1.0
        self.variance = 0.0
        self.pending = None

    @property
    def threshold(self):
        return self.scale * (self.mean + math.sqrt(max(0.0, self.variance)))

    @torch.no_grad()
    def __call__(self, target_logp, draft_logp, answer_mask, update=True):
        ratio = (target_logp.double() - draft_logp.detach().double()).exp()
        valid = answer_mask.bool()
        if not torch.isfinite(ratio[valid]).all():
            raise ValueError('Non-finite draft-target probability ratio')
        values = ratio[valid]
        moments = torch.stack((values.sum(), values.square().sum(),
                               values.new_tensor(values.numel())))
        self.pending = moments if self.pending is None else self.pending + moments
        if update:
            moments = self.pending
            self.pending = None
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(moments)
            if moments[2] > 0:
                mean = (moments[0] / moments[2]).item()
                variance = max(0.0, (moments[1] / moments[2]).item() - mean * mean)
                self.mean = self.momentum * self.mean + (1 - self.momentum) * mean
                self.variance = self.momentum * self.variance + (1 - self.momentum) * variance
        return valid & (ratio > self.threshold)

    def state_dict(self):
        return dict(mean=self.mean, variance=self.variance)

    def load_state_dict(self, state):
        self.mean = float(state['mean'])
        self.variance = float(state['variance'])
        self.pending = None


def vision_critical_mask(dependence, answer_mask, gamma=0.1):
    if not 0 < gamma <= 1:
        raise ValueError('gamma must be in (0, 1]')
    squeeze = dependence.ndim == 1
    scores = dependence.unsqueeze(0) if squeeze else dependence
    valid = answer_mask.unsqueeze(0) if squeeze else answer_mask
    selected = torch.zeros_like(valid, dtype=torch.bool)
    for row in range(scores.shape[0]):
        positions = valid[row].bool().nonzero(as_tuple=True)[0]
        if positions.numel() == 0:
            continue
        values = scores[row, positions]
        if not torch.isfinite(values).all():
            raise ValueError('Non-finite visual dependence scores')
        count = max(1, math.floor(positions.numel() * gamma))
        chosen = values.topk(count).indices
        selected[row, positions[chosen]] = True
    return selected[0] if squeeze else selected


def token_logprobs(logits, token_ids):
    return F.log_softmax(logits.float(), dim=-1).gather(
        -1, token_ids.long().unsqueeze(-1)).squeeze(-1)


def critical_mask(hard_mask, visual_mask, answer_mask):
    return (hard_mask.bool() | visual_mask.bool()) & answer_mask.bool()


def selected_denominator(weights):
    count = weights.sum().detach().clone()
    world_size = 1
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(count)
        world_size = dist.get_world_size()
    return (count + 1e-8) / world_size


def nll_loss(logits, token_ids, mask, normalize=True):
    logp = token_logprobs(logits, token_ids)
    weights = mask.to(logp.dtype)
    loss = -(logp * weights).sum()
    if normalize:
        loss = loss / selected_denominator(weights)
    else:
        loss = loss / logits.shape[0]
    return loss


def distillation_loss(predicted_hidden, target_hidden, draft_logits, target_logits,
                      mask, hidden_weight=1.0, probability_weight=0.1):
    weights = mask.float()
    denominator = selected_denominator(weights)
    hidden = F.smooth_l1_loss(predicted_hidden.float(), target_hidden.float(),
                             reduction='none').mean(dim=-1)
    target_p = F.softmax(target_logits.detach().float(), dim=-1)
    probability = -(target_p * F.log_softmax(draft_logits.float(), dim=-1)).sum(dim=-1)
    return ((hidden_weight * hidden + probability_weight * probability) * weights).sum() / denominator


def pool_visual_embeddings(image_embeddings, pool_type='mean'):
    if image_embeddings.ndim != 2 or image_embeddings.shape[0] == 0:
        raise ValueError('Image embeddings must have shape [N, hidden_size] with N > 0')
    values = image_embeddings.float()
    if pool_type == 'mean':
        return values.mean(dim=0, keepdim=True)
    if pool_type == 'max':
        return values.max(dim=0, keepdim=True).values
    if pool_type == 'mean_max':
        return (values.mean(dim=0, keepdim=True) + values.max(dim=0, keepdim=True).values) / 2
    raise ValueError(f'Unknown visual pooling type: {pool_type}')


def enhance_hidden_states(hidden_states, image_embeddings, question_mask, answer_mask,
                          eta_q=0.02, eta_a=0.1, pool_type='mean'):
    question_pooled = pool_visual_embeddings(image_embeddings, 'mean').squeeze(0).to(hidden_states.device)
    answer_pooled = (question_pooled if pool_type == 'mean' else
                     pool_visual_embeddings(image_embeddings, pool_type).squeeze(0).to(hidden_states.device))
    if question_pooled.shape[-1] != hidden_states.shape[-1]:
        raise ValueError('Image embeddings must be projected to the language hidden dimension')
    question_prior = question_mask.float().unsqueeze(-1) * eta_q * question_pooled
    answer_prior = answer_mask.float().unsqueeze(-1) * eta_a * answer_pooled
    return (hidden_states.float() + question_prior + answer_prior).to(hidden_states.dtype)
