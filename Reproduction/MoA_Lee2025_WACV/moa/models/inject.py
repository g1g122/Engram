import re

import torch.nn as nn

from .kronecker import SharedKroneckerRule
from .qkv_wrapper import MoAQKVWrapper


def _split_module_name(module_name):
    parts = module_name.split(".")
    if len(parts) == 1:
        return "", parts[0]
    return ".".join(parts[:-1]), parts[-1]


def _block_index(module_name):
    match = re.search(r"(?:^|\.)blocks\.(\d+)\.", module_name)
    if match is None:
        return None
    return int(match.group(1))


def find_qkv_linear_layers(model, target_suffix="qkv", block_stride=2):
    targets = []
    for module_name, module in model.named_modules():
        if not module_name.endswith(target_suffix) or not isinstance(module, nn.Linear):
            continue
        block_index = _block_index(module_name)
        if block_stride is not None and block_index is not None:
            if block_index % block_stride != 0:
                continue
        if block_stride is not None and block_index is None:
            continue
        targets.append((module_name, module))
    return targets


def _unique_trainable_params(modules):
    params = []
    seen = set()
    for module in modules:
        for param in module.parameters():
            if param.requires_grad and id(param) not in seen:
                params.append(param)
                seen.add(id(param))
    return params


def inject_moa_qkv(
    model,
    target_suffix="qkv",
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
    freeze_base=True,
    share_rule=True,
    block_stride=2,
    verbose=True,
):
    """Replace qkv Linear layers with MoA wrappers."""
    targets = find_qkv_linear_layers(
        model,
        target_suffix=target_suffix,
        block_stride=block_stride,
    )

    shared_rule = None
    if share_rule:
        shared_rule = SharedKroneckerRule(phm_dim=phm_dim, init_std=init_std)
        model.add_module("moa_shared_rule", shared_rule)

    wrappers = []
    injected_names = []

    for module_name, old_qkv in targets:
        parent_name, child_name = _split_module_name(module_name)
        parent = model.get_submodule(parent_name) if parent_name else model

        rule = shared_rule
        if rule is None:
            rule = SharedKroneckerRule(phm_dim=phm_dim, init_std=init_std)

        new_qkv = MoAQKVWrapper(
            base_qkv=old_qkv,
            num_experts=num_experts,
            phm_dim=phm_dim,
            ranks=ranks,
            shared_rule=rule,
            router_normalize_input=router_normalize_input,
            router_temperature=router_temperature,
            router_max_logit_scale=router_max_logit_scale,
            router_noise_std=router_noise_std,
            aux_loss_type=aux_loss_type,
            adapter_scale=adapter_scale,
            init_std=init_std,
            expert_dropout=expert_dropout,
            freeze_base=freeze_base,
        )

        new_qkv.to(device=old_qkv.weight.device, dtype=old_qkv.weight.dtype)
        setattr(parent, child_name, new_qkv)

        wrappers.append(new_qkv)
        injected_names.append(module_name)

        if verbose:
            print(f"Injected MoA into {module_name}")

    param_modules = list(wrappers)
    if shared_rule is not None:
        param_modules.append(shared_rule)
    trainable_params = _unique_trainable_params(param_modules)
    return trainable_params, injected_names


def iter_moa_wrappers(model):
    for module in model.modules():
        if isinstance(module, MoAQKVWrapper):
            yield module


def collect_moa_parameters(model):
    return _unique_trainable_params(iter_moa_wrappers(model))


def collect_moa_aux_loss(model):
    total = None
    for wrapper in iter_moa_wrappers(model):
        aux_loss = wrapper.adapter.aux_loss
        if aux_loss is None:
            continue
        total = aux_loss if total is None else total + aux_loss
    return total


def set_moa_router_noise_std(model, value):
    for wrapper in iter_moa_wrappers(model):
        wrapper.adapter.router_noise_std = value
