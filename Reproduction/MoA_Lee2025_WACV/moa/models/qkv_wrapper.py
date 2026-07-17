import torch.nn as nn

from .adapter import MoAAdapter


class MoAQKVWrapper(nn.Module):
    """Wrap a frozen qkv Linear layer with real-valued MoA adaptation."""

    def __init__(
        self,
        base_qkv,
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
        freeze_base=True,
    ):
        super().__init__()

        if not isinstance(base_qkv, nn.Linear):
            raise TypeError(
                f"base_qkv must be nn.Linear, got {type(base_qkv).__name__}."
            )

        self.base_qkv = base_qkv
        self.in_features = base_qkv.in_features
        self.out_features = base_qkv.out_features
        self.freeze_base = freeze_base

        if freeze_base:
            self.base_qkv.requires_grad_(False)

        self.adapter = MoAAdapter(
            in_features=self.in_features,
            out_features=self.out_features,
            num_experts=num_experts,
            phm_dim=phm_dim,
            ranks=ranks,
            shared_rule=shared_rule,
            router_normalize_input=router_normalize_input,
            router_temperature=router_temperature,
            router_max_logit_scale=router_max_logit_scale,
            router_noise_std=router_noise_std,
            aux_loss_type=aux_loss_type,
            adapter_scale=adapter_scale,
            init_std=init_std,
            expert_dropout=expert_dropout,
        )

    @property
    def weight(self):
        return self.base_qkv.weight

    @property
    def bias(self):
        return self.base_qkv.bias

    @property
    def last_gates(self):
        return self.adapter.last_gates

    @property
    def last_logits(self):
        return self.adapter.last_logits

    @property
    def last_delta(self):
        return self.adapter.last_delta

    def forward(self, x):
        base_out = self.base_qkv(x)
        return self.adapter(x, base_out)

    def extra_repr(self):
        return (
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"freeze_base={self.freeze_base}, "
            f"phm_dim={self.adapter.phm_dim}, "
            f"ranks={self.adapter.ranks}"
        )
