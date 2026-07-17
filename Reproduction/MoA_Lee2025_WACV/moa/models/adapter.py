import torch
import torch.nn as nn

from .kronecker import MoAKroneckerExpert, SharedKroneckerRule
from .router import CosineRouter


class MoAAdapter(nn.Module):
    """Real-valued Mixture-of-Adapters core."""

    def __init__(
        self,
        in_features,
        out_features,
        num_experts=4,
        phm_dim=64,
        ranks=None,
        shared_rule=None,
        router_normalize_input=True,
        router_temperature=0.5,
        router_max_logit_scale=100.0,
        router_noise_std=0.0,
        aux_loss_type="load_importance",
        adapter_scale=1.0,
        init_std=0.01,
        expert_dropout=0.5,
    ):
        super().__init__()

        if ranks is None:
            ranks = [1, 2, 4, 8] if num_experts == 4 else [1] * num_experts
        else:
            ranks = list(ranks)
            num_experts = len(ranks)

        if num_experts < 1:
            raise ValueError(f"num_experts must be positive, got {num_experts}.")
        if aux_loss_type != "load_importance":
            raise ValueError("MoA now supports only aux_loss_type='load_importance'.")

        self.in_features = in_features
        self.out_features = out_features
        self.num_experts = num_experts
        self.phm_dim = phm_dim
        self.ranks = tuple(ranks)
        self.router_noise_std = router_noise_std
        self.aux_loss_type = aux_loss_type
        self.adapter_scale = adapter_scale

        if shared_rule is None:
            shared_rule = SharedKroneckerRule(phm_dim=phm_dim, init_std=init_std)
        self.shared_rule = shared_rule

        self.router = CosineRouter(
            dim=in_features,
            num_experts=num_experts,
            normalize_input=router_normalize_input,
            init_temperature=router_temperature,
            max_logit_scale=router_max_logit_scale,
        )
        self.experts = nn.ModuleList(
            [
                MoAKroneckerExpert(
                    in_features=in_features,
                    out_features=out_features,
                    phm_dim=phm_dim,
                    rank=rank,
                    shared_rule=shared_rule,
                    init_std=init_std,
                    dropout=expert_dropout,
                )
                for rank in ranks
            ]
        )
        self.b_adapter = nn.Parameter(torch.zeros(out_features))

        self.last_gates = None
        self.last_logits = None
        self.last_soft_gates = None
        self.last_delta = None
        self.aux_loss = None

    def forward(self, x, base_out):
        if x.size(-1) != self.in_features:
            raise ValueError(
                f"Expected x last dim {self.in_features}, got {x.size(-1)}."
            )
        if base_out.size(-1) != self.out_features:
            raise ValueError(
                f"Expected base_out last dim {self.out_features}, "
                f"got {base_out.size(-1)}."
            )

        clean_gates, logits = self.router(x)
        if self.training and self.router_noise_std > 0:
            routing_logits = logits + torch.randn_like(logits) * self.router_noise_std
        else:
            routing_logits = logits

        routing_gates = torch.softmax(routing_logits, dim=-1)
        top_scores, top_indices = torch.max(routing_gates, dim=-1, keepdim=True)
        gates = torch.zeros_like(routing_gates).scatter_(-1, top_indices, 1.0)
        gates = gates * routing_gates

        delta = x.new_zeros(*x.shape[:-1], self.out_features)
        for expert_idx, expert in enumerate(self.experts):
            delta = delta + gates[..., expert_idx : expert_idx + 1] * expert(x)

        self.last_gates = gates.detach()
        self.last_logits = logits.detach()
        self.last_soft_gates = clean_gates.detach()
        self.last_delta = (delta + self.b_adapter).detach()

        self.aux_loss = self._load_importance_loss(clean_gates, top_scores)

        return base_out + self.adapter_scale * (delta + self.b_adapter)

    def _load_importance_loss(self, scores_wo_noise, topk_logits, gate_noise=1.0):
        if self.num_experts <= 1:
            return scores_wo_noise.new_tensor(0.0)

        scores = scores_wo_noise.reshape(-1, self.num_experts).float()
        thresholds = topk_logits.reshape(-1, 1).float()
        importance = scores.sum(0)
        importance_loss = importance.var() / (importance.mean() ** 2 + 1e-10)

        normal = torch.distributions.Normal(
            scores.new_tensor(0.0),
            scores.new_tensor(gate_noise / self.num_experts),
        )
        load = normal.cdf(scores - thresholds).sum(0)
        load_loss = load.var() / (load.mean() ** 2 + 1e-10)

        return (importance_loss + load_loss) / 2.0
