import hashlib
import random
from collections import defaultdict

import numpy as np

from torch.utils.data import (
    BatchSampler,
    ConcatDataset,
    DataLoader,
    RandomSampler,
    SequentialSampler,
    Subset,
)

from .datasets import get_dataset_class, resolve_domain_indices


def cfg_get(cfg, path, default=None):
    """Read a dotted config path from dict-like or attribute-like objects."""
    current = cfg
    for part in path.split("."):
        if isinstance(current, dict):
            if part not in current:
                return default
            current = current[part]
        else:
            if not hasattr(current, part):
                return default
            current = getattr(current, part)
    return current


def get_targets(dataset):
    """Return class targets for an ImageFolder-style dataset."""
    if hasattr(dataset, "targets"):
        return list(dataset.targets)
    if hasattr(dataset, "samples"):
        return [target for _, target in dataset.samples]
    raise AttributeError(
        f"Dataset {type(dataset).__name__} has neither 'targets' nor 'samples'."
    )


def seed_hash(*args):
    """Derive a stable integer seed from arbitrary arguments."""
    args_str = str(args)
    return int(hashlib.md5(args_str.encode("utf-8")).hexdigest(), 16) % (2 ** 31)


def split_indices_stratified(dataset, holdout_fraction, seed):
    """Split one domain into train/validation indices with class preservation."""
    if holdout_fraction <= 0 or holdout_fraction >= 1:
        raise ValueError(
            f"holdout_fraction must be in (0, 1), got {holdout_fraction}."
        )

    targets = get_targets(dataset)
    by_class = defaultdict(list)
    for index, target in enumerate(targets):
        by_class[target].append(index)

    rng = random.Random(seed)
    train_indices = []
    val_indices = []

    for indices in by_class.values():
        indices = list(indices)
        rng.shuffle(indices)

        if len(indices) == 1:
            train_indices.extend(indices)
            continue

        val_count = int(round(len(indices) * holdout_fraction))
        val_count = max(1, min(len(indices) - 1, val_count))

        val_indices.extend(indices[:val_count])
        train_indices.extend(indices[val_count:])

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    return train_indices, val_indices


def split_indices_random(dataset, holdout_fraction, seed):
    """Split one domain into train/validation indices with a random shuffle."""
    if holdout_fraction <= 0 or holdout_fraction >= 1:
        raise ValueError(
            f"holdout_fraction must be in (0, 1), got {holdout_fraction}."
        )

    indices = list(range(len(dataset)))
    np.random.RandomState(seed).shuffle(indices)

    val_count = int(len(indices) * holdout_fraction)
    val_count = max(1, min(len(indices) - 1, val_count))

    val_indices = indices[:val_count]
    train_indices = indices[val_count:]
    return train_indices, val_indices


def split_indices(dataset, holdout_fraction, seed, split_mode="random"):
    """Split one domain using the configured split mode."""
    if split_mode == "random":
        return split_indices_random(dataset, holdout_fraction, seed)
    if split_mode == "stratified":
        return split_indices_stratified(dataset, holdout_fraction, seed)
    raise ValueError(
        f"Unknown data.split_mode: {split_mode}. "
        "Expected 'random' or 'stratified'."
    )


def make_replacement_train_loader(
    dataset,
    batch_size,
    num_workers,
    pin_memory=True,
    persistent_workers=False,
    prefetch_factor=None,
):
    """Build a source-domain train loader with fixed-size replacement batches."""
    if len(dataset) == 0:
        raise ValueError("Cannot build a train loader for an empty dataset.")

    num_samples = max(len(dataset), batch_size)
    sampler = RandomSampler(
        dataset,
        replacement=True,
        num_samples=num_samples,
    )
    batch_sampler = BatchSampler(
        sampler,
        batch_size=batch_size,
        drop_last=True,
    )
    loader_kwargs = dataloader_worker_kwargs(
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )
    return DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        **loader_kwargs,
    )


def make_eval_loader(
    dataset,
    batch_size,
    num_workers,
    pin_memory=True,
    persistent_workers=False,
    prefetch_factor=None,
):
    """Build an eval loader that visits each example once in order."""
    loader_kwargs = dataloader_worker_kwargs(
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=SequentialSampler(dataset),
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        **loader_kwargs,
    )


def dataloader_worker_kwargs(num_workers, persistent_workers=False, prefetch_factor=None):
    """Return DataLoader worker kwargs that are valid for num_workers."""
    if num_workers <= 0:
        return {}

    kwargs = {"persistent_workers": persistent_workers}
    if prefetch_factor is not None:
        kwargs["prefetch_factor"] = prefetch_factor
    return kwargs


def infinite_loader(loader):
    """Yield batches forever by repeatedly iterating over a DataLoader."""
    while True:
        for batch in loader:
            yield batch


def make_infinite_loaders(loaders):
    """Wrap a dict of loaders as infinite iterators keyed by domain name."""
    return {
        domain_name: infinite_loader(loader)
        for domain_name, loader in loaders.items()
    }


def next_multidomain_batch(loader_iters):
    """Take one batch from every source-domain iterator."""
    return {
        domain_name: next(iterator)
        for domain_name, iterator in loader_iters.items()
    }


def build_dg_loaders(cfg):
    """Build DomainBed-style loaders for step-based DG training.

    The configured batch size is per source domain. If there are three source
    domains and batch_size=32, one training step consumes a merged batch of 96.
    """
    dataset_name = cfg_get(cfg, "data.dataset")
    data_root = cfg_get(cfg, "data.root", "datasets")
    test_domains = cfg_get(cfg, "data.test_domains", [])
    holdout_fraction = cfg_get(cfg, "data.holdout_fraction", 0.2)
    split_mode = cfg_get(cfg, "data.split_mode", "random")
    split_seed = cfg_get(cfg, "experiment.split_seed", 0)
    image_size = cfg_get(cfg, "data.image_size", 224)
    normalization = cfg_get(cfg, "data.normalization", "clip")
    augment = cfg_get(cfg, "data.augment", True)
    num_workers = cfg_get(cfg, "data.num_workers", 4)
    persistent_workers = cfg_get(cfg, "data.persistent_workers", False)
    prefetch_factor = cfg_get(cfg, "data.prefetch_factor", None)

    batch_size = cfg_get(cfg, "train.batch_size", 32)
    eval_batch_size = cfg_get(cfg, "train.eval_batch_size", None)
    eval_batch_size = eval_batch_size or cfg_get(cfg, "data.test_batchsize", None)
    eval_batch_size = eval_batch_size or 128
    pin_memory = cfg_get(cfg, "train.pin_memory", True)

    if dataset_name is None:
        raise ValueError("Missing config value: data.dataset")
    if not test_domains:
        raise ValueError("Missing config value: data.test_domains")

    dataset_cls = get_dataset_class(dataset_name)
    train_dataset_obj = dataset_cls(
        root=data_root,
        test_domains=test_domains,
        augment=augment,
        image_size=image_size,
        normalization=normalization,
    )
    eval_dataset_obj = dataset_cls(
        root=data_root,
        test_domains=list(range(len(train_dataset_obj.domains))),
        augment=False,
        image_size=image_size,
        normalization=normalization,
    )

    target_indices = resolve_domain_indices(train_dataset_obj.domains, test_domains)

    source_train_loaders = {}
    source_val_by_domain_loaders = {}
    target_by_domain_loaders = {}

    source_train_sets = []
    source_val_sets = []
    target_sets = []
    source_domains = []
    target_domains = []
    source_train_sizes = {}
    source_val_sizes = {}
    target_sizes = {}
    steps_per_epoch_by_domain = {}

    for domain_index, train_domain_dataset in enumerate(train_dataset_obj.datasets):
        domain_name = train_dataset_obj.domains[domain_index]

        if domain_index in target_indices:
            target_dataset = eval_dataset_obj.datasets[domain_index]
            target_sets.append(target_dataset)
            target_domains.append(domain_name)
            target_sizes[domain_name] = len(target_dataset)
            target_by_domain_loaders[domain_name] = make_eval_loader(
            target_dataset,
            batch_size=eval_batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
        )
            continue

        train_indices, val_indices = split_indices(
            train_domain_dataset,
            holdout_fraction=holdout_fraction,
            seed=seed_hash(split_seed, domain_index),
            split_mode=split_mode,
        )

        train_subset = Subset(train_domain_dataset, train_indices)
        val_subset = Subset(eval_dataset_obj.datasets[domain_index], val_indices)

        source_domains.append(domain_name)
        source_train_sets.append(train_subset)
        source_val_sets.append(val_subset)
        source_train_sizes[domain_name] = len(train_subset)
        source_val_sizes[domain_name] = len(val_subset)
        steps_per_epoch_by_domain[domain_name] = len(train_subset) / batch_size

        source_train_loaders[domain_name] = make_replacement_train_loader(
            train_subset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
        )
        source_val_by_domain_loaders[domain_name] = make_eval_loader(
            val_subset,
            batch_size=eval_batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            prefetch_factor=prefetch_factor,
        )

    if not source_train_sets:
        raise ValueError("No source domains remain after selecting data.test_domains.")
    if not source_val_sets:
        raise ValueError("No source validation split was created.")
    if not target_sets:
        raise ValueError("No target domains were selected.")

    source_val_dataset = ConcatDataset(source_val_sets)
    target_dataset = ConcatDataset(target_sets)

    source_val_loader = make_eval_loader(
        source_val_dataset,
        batch_size=eval_batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )
    target_loader = make_eval_loader(
        target_dataset,
        batch_size=eval_batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )

    steps_per_epoch = min(steps_per_epoch_by_domain.values())
    total_train_batch_size = batch_size * len(source_domains)

    return {
        "dataset": train_dataset_obj,
        "source_train_loaders": source_train_loaders,
        "source_val_loader": source_val_loader,
        "target_loader": target_loader,
        "source_val_by_domain_loaders": source_val_by_domain_loaders,
        "target_by_domain_loaders": target_by_domain_loaders,
        "source_domains": source_domains,
        "target_domains": target_domains,
        "steps_per_epoch": steps_per_epoch,
        "steps_per_epoch_by_domain": steps_per_epoch_by_domain,
        "batch_size_per_domain": batch_size,
        "total_train_batch_size": total_train_batch_size,
        "eval_batch_size": eval_batch_size,
        "persistent_workers": persistent_workers,
        "prefetch_factor": prefetch_factor,
        "split_mode": split_mode,
        "split_sizes": {
            "source_train": source_train_sizes,
            "source_val": source_val_sizes,
            "target": target_sizes,
        },
    }
