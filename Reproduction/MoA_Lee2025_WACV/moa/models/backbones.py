import torch.nn as nn
import timm

from .inject import inject_moa_qkv


class Identity(nn.Module):
    """Identity layer used to remove classifier heads."""

    def forward(self, x):
        return x


def freeze_module(module):
    for param in module.parameters():
        param.requires_grad = False


def build_clip_vit_backbone(
    model_name="vit_base_patch16_clip_224.laion2b",
    pretrained=True,
    remove_head=True,
):
    network = timm.create_model(
        model_name,
        pretrained=pretrained,
        pretrained_strict=False,
    )
    n_outputs = network.num_features

    if remove_head:
        if hasattr(network, "head"):
            network.head = Identity()
        elif hasattr(network, "fc"):
            network.fc = Identity()
        else:
            raise AttributeError(
                f"Model {model_name} has neither 'head' nor 'fc' to remove."
            )

    return network, n_outputs


def build_moa_clip_vit(
    model_name="vit_base_patch16_clip_224.laion2b",
    pretrained=True,
    num_experts=4,
    phm_dim=64,
    ranks=None,
    router_normalize_input=True,
    router_temperature=0.5,
    router_max_logit_scale=100.0,
    router_noise_std=0.0,
    aux_loss_type="load_importance",
    adapter_scale=1.0,
    init_std=0.01,
    expert_dropout=0.5,
    target_suffix="qkv",
    freeze_backbone=True,
    freeze_base_qkv=True,
    share_rule=True,
    block_stride=2,
    verbose=True,
):
    """Build a frozen CLIP ViT backbone with MoA injected into qkv layers."""
    network, n_outputs = build_clip_vit_backbone(
        model_name=model_name,
        pretrained=pretrained,
        remove_head=True,
    )

    if freeze_backbone:
        freeze_module(network)

    trainable_params, injected_names = inject_moa_qkv(
        model=network,
        target_suffix=target_suffix,
        num_experts=num_experts,
        phm_dim=phm_dim,
        ranks=ranks,
        router_normalize_input=router_normalize_input,
        router_temperature=router_temperature,
        router_max_logit_scale=router_max_logit_scale,
        router_noise_std=router_noise_std,
        aux_loss_type=aux_loss_type,
        adapter_scale=adapter_scale,
        init_std=init_std,
        expert_dropout=expert_dropout,
        freeze_base=freeze_base_qkv,
        share_rule=share_rule,
        block_stride=block_stride,
        verbose=verbose,
    )

    if verbose:
        print(
            f"MoA injected {len(injected_names)} qkv layers. "
            f"Trainable parameter groups: {len(trainable_params)}"
        )

    return network, n_outputs, trainable_params, injected_names


def get_backbone(
    name="moa_clip_vit_b16",
    model_name=None,
    pretrained=True,
    **kwargs,
):
    if name in {"moa_clip_vit_b16", "moa_vitbase"}:
        model_name = model_name or "vit_base_patch16_clip_224.laion2b"
        return build_moa_clip_vit(
            model_name=model_name,
            pretrained=pretrained,
            **kwargs,
        )

    if name in {"clip_vit_b16", "vitbase_clip"}:
        model_name = model_name or "vit_base_patch16_clip_224.laion2b"
        network, n_outputs = build_clip_vit_backbone(
            model_name=model_name,
            pretrained=pretrained,
            remove_head=True,
        )
        return network, n_outputs, [], []

    raise ValueError(f"Unknown backbone name: {name}")
