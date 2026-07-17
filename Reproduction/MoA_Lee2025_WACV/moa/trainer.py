import copy
from pathlib import Path

import torch

from moa import moa_logging
from moa.data.loaders import (
    build_dg_loaders,
    cfg_get,
    make_infinite_loaders,
    next_multidomain_batch,
)
from moa.engine import evaluate, train_one_step
from moa.models import build_image_classifier, set_moa_router_noise_std
from moa.utils import MetricLogger, set_seed, summarize_parameters


def cfg_set(cfg, path, value):
    """Set a dotted config path on a dict-like config."""
    current = cfg
    parts = path.split(".")
    for part in parts[:-1]:
        if part not in current or current[part] is None:
            current[part] = {}
        current = current[part]
    current[parts[-1]] = value


def prepare_run_config(cfg, target_domains=None, output_dir=None):
    """Return a deep-copied config with run-specific target/output settings."""
    run_cfg = copy.deepcopy(cfg)
    if target_domains is not None:
        cfg_set(run_cfg, "data.test_domains", list(target_domains))
    if output_dir is not None:
        cfg_set(run_cfg, "experiment.output_dir", str(output_dir))
    return run_cfg


def build_optimizer(name, params, lr, weight_decay):
    """Build the configured optimizer."""
    name = name.lower()
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, weight_decay=weight_decay, momentum=0.9)
    raise ValueError(f"Unsupported optimizer: {name}")


def save_checkpoint(path, model, optimizer, step, metrics, metadata):
    """Save model/optimizer state for one training run."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "metrics": metrics,
            "metadata": metadata,
        },
        path,
    )


def evaluate_domain_loaders(model, domain_loaders, device, aux_weight=0.0):
    """Evaluate named domain loaders and return metrics keyed by domain name."""
    return {
        domain_name: evaluate(
            model=model,
            data_loader=loader,
            device=device,
            aux_weight=aux_weight,
        )
        for domain_name, loader in domain_loaders.items()
    }


def aggregate_domain_metrics(domain_metrics):
    """Aggregate per-domain eval metrics using num_samples as weights."""
    total_samples = sum(
        metrics["num_samples"]
        for metrics in domain_metrics.values()
    )
    if total_samples <= 0:
        raise ValueError("Cannot aggregate metrics with zero samples.")

    metric_names = set()
    for metrics in domain_metrics.values():
        metric_names.update(metrics.keys())
    metric_names.discard("num_samples")

    aggregated = {"num_samples": total_samples}
    for metric_name in sorted(metric_names):
        aggregated[metric_name] = sum(
            metrics[metric_name] * metrics["num_samples"]
            for metrics in domain_metrics.values()
        ) / total_samples

    return aggregated


class DGTrainer:
    """Step-based Domain Generalization trainer for one target-domain setting."""

    def __init__(self, cfg, target_domains=None, output_dir=None):
        self.cfg = prepare_run_config(
            cfg,
            target_domains=target_domains,
            output_dir=output_dir,
        )
        self.output_dir = Path(cfg_get(self.cfg, "experiment.output_dir"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.steps = cfg_get(self.cfg, "train.steps", 5001)
        self.checkpoint_freq = cfg_get(self.cfg, "train.checkpoint_freq", 200)
        self.log_interval = cfg_get(self.cfg, "train.log_interval", 50)
        self.aux_weight = cfg_get(self.cfg, "model.aux_weight", 0.0)
        self.router_noise_std = cfg_get(self.cfg, "model.router_noise_std", 0.0)

        seed = cfg_get(self.cfg, "experiment.train_seed", 0)
        set_seed(seed)

        requested_device = cfg_get(self.cfg, "train.device", "cuda")
        self.device = torch.device(
            requested_device
            if torch.cuda.is_available() or requested_device == "cpu"
            else "cpu"
        )

        self.loaders = None
        self.model = None
        self.optimizer = None
        self.scaler = None
        self.metadata = {}

    def setup(self):
        """Build data, model, optimizer, scaler, and run metadata."""
        self.loaders = build_dg_loaders(self.cfg)
        dataset = self.loaders["dataset"]

        self.model, trainable_params, model_metadata = build_image_classifier(
            num_classes=dataset.num_classes,
            backbone_name=cfg_get(self.cfg, "model.backbone", "moa_clip_vit_b16"),
            model_name=cfg_get(self.cfg, "model.model_name", None),
            pretrained=cfg_get(self.cfg, "model.pretrained", True),
            num_experts=cfg_get(self.cfg, "model.num_experts", 4),
            phm_dim=cfg_get(self.cfg, "model.phm_dim", 64),
            ranks=cfg_get(
                self.cfg,
                "model.ranks",
                cfg_get(self.cfg, "model.expert_ranks", None),
            ),
            router_temperature=cfg_get(self.cfg, "model.router_temperature", 0.5),
            router_noise_std=cfg_get(self.cfg, "model.router_noise_std", 0.0),
            aux_loss_type=cfg_get(self.cfg, "model.aux_loss_type", "load_importance"),
            adapter_scale=cfg_get(self.cfg, "model.adapter_scale", 1.0),
            init_std=cfg_get(self.cfg, "model.init_std", 0.01),
            expert_dropout=cfg_get(self.cfg, "model.expert_dropout", 0.5),
            share_rule=cfg_get(self.cfg, "model.share_rule", True),
            block_stride=cfg_get(self.cfg, "model.block_stride", 2),
            verbose=cfg_get(self.cfg, "model.verbose", True),
        )
        self.model.to(self.device)

        self.optimizer = build_optimizer(
            name=cfg_get(self.cfg, "train.optimizer", "adam"),
            params=trainable_params,
            lr=cfg_get(self.cfg, "train.lr", 5e-5),
            weight_decay=cfg_get(self.cfg, "train.weight_decay", 0.0),
        )

        use_amp = cfg_get(self.cfg, "train.amp", False) and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda") if use_amp else None

        self.metadata = {
            "model": model_metadata,
            "parameter_summary": summarize_parameters(self.model),
            "source_domains": self.loaders["source_domains"],
            "target_domains": self.loaders["target_domains"],
            "steps_per_epoch": self.loaders["steps_per_epoch"],
            "steps_per_epoch_by_domain": self.loaders["steps_per_epoch_by_domain"],
            "batch_size_per_domain": self.loaders["batch_size_per_domain"],
            "total_train_batch_size": self.loaders["total_train_batch_size"],
            "eval_batch_size": self.loaders["eval_batch_size"],
            "persistent_workers": self.loaders["persistent_workers"],
            "prefetch_factor": self.loaders["prefetch_factor"],
            "split_mode": self.loaders["split_mode"],
            "split_sizes": self.loaders["split_sizes"],
            "device": str(self.device),
        }
        moa_logging.write_run_config(self.output_dir, self.cfg, metadata=self.metadata)

        print(f"[setup] output_dir: {self.output_dir}")
        print(f"[setup] source domains: {self.loaders['source_domains']}")
        print(f"[setup] target domains: {self.loaders['target_domains']}")
        print(f"[setup] split sizes: {self.loaders['split_sizes']}")
        print(f"[setup] parameter summary: {self.metadata['parameter_summary']}")
        print(f"[setup] injected layers: {model_metadata['injected_names']}")

    def evaluate_and_log(
        self,
        step,
        train_metrics,
        best_source_val_acc,
        aux_weight,
        router_noise_std,
    ):
        """Evaluate source validation/target loaders, log, and return metrics."""
        source_val_by_domain = evaluate_domain_loaders(
            model=self.model,
            domain_loaders=self.loaders["source_val_by_domain_loaders"],
            device=self.device,
            aux_weight=aux_weight,
        )
        target_by_domain = evaluate_domain_loaders(
            model=self.model,
            domain_loaders=self.loaders["target_by_domain_loaders"],
            device=self.device,
            aux_weight=aux_weight,
        )
        source_val_metrics = aggregate_domain_metrics(source_val_by_domain)
        target_metrics = aggregate_domain_metrics(target_by_domain)

        improved = source_val_metrics["acc1"] > best_source_val_acc
        estimated_epoch = step / self.loaders["steps_per_epoch"]
        metrics = {
            "train": train_metrics,
            "source_val": source_val_metrics,
            "source_val_by_domain": source_val_by_domain,
            "target": target_metrics,
            "target_by_domain": target_by_domain,
        }
        record = {
            "step": step,
            "estimated_epoch": estimated_epoch,
            "improved": improved,
            "aux_weight": aux_weight,
            "router_noise_std": router_noise_std,
            **metrics,
        }
        moa_logging.write_eval_record(self.output_dir, record)
        print(
            "[eval] "
            + moa_logging.format_eval_row(
                step=step,
                total_steps=self.steps,
                estimated_epoch=estimated_epoch,
                train_metrics=train_metrics,
                source_val_metrics=source_val_metrics,
                target_metrics=target_metrics,
                source_val_by_domain=source_val_by_domain,
                target_by_domain=target_by_domain,
                improved=improved,
                aux_weight=aux_weight,
                router_noise_std=router_noise_std,
            )
        )
        return improved, metrics, record

    def run(self):
        """Run one target-domain experiment and return its summary."""
        if self.model is None:
            self.setup()

        train_iters = make_infinite_loaders(self.loaders["source_train_loaders"])
        train_window = MetricLogger()
        best_step = None
        best_source_val_acc = -1.0
        target_acc_at_best_source_val = -1.0
        last_metrics = None
        last_eval_step = None

        max_grad_norm = cfg_get(self.cfg, "train.max_grad_norm", None)
        set_moa_router_noise_std(self.model, self.router_noise_std)

        for step in range(self.steps):
            domain_batches = next_multidomain_batch(train_iters)
            step_metrics = train_one_step(
                model=self.model,
                domain_batches=domain_batches,
                optimizer=self.optimizer,
                device=self.device,
                aux_weight=self.aux_weight,
                scaler=self.scaler,
                max_grad_norm=max_grad_norm,
            )

            batch_size = step_metrics.pop("batch_size")
            train_window.update_with_count(
                batch_size,
                loss=step_metrics["loss"],
                cls_loss=step_metrics["cls_loss"],
                aux_loss=step_metrics["aux_loss"],
                acc1=step_metrics["acc1"],
            )
            train_window.update(lr=step_metrics["lr"])

            if self.log_interval and (step + 1) % self.log_interval == 0:
                train_metrics = train_window.averages()
                train_record = {
                    "step": step + 1,
                    "estimated_epoch": (step + 1) / self.loaders["steps_per_epoch"],
                    "aux_weight": self.aux_weight,
                    "router_noise_std": self.router_noise_std,
                    **train_metrics,
                }
                train_metrics = {
                    **train_metrics,
                    "aux_weight": self.aux_weight,
                    "router_noise_std": self.router_noise_std,
                }
                moa_logging.write_train_record(self.output_dir, train_record)
                print(
                    "[train] "
                    + moa_logging.format_train_step_row(
                        step=step + 1,
                        total_steps=self.steps,
                        train_metrics=train_metrics,
                    )
                )

            if step % self.checkpoint_freq == 0:
                train_metrics = train_window.averages()
                train_window.reset()
                improved, metrics, _ = self.evaluate_and_log(
                    step=step,
                    train_metrics=train_metrics,
                    best_source_val_acc=best_source_val_acc,
                    aux_weight=self.aux_weight,
                    router_noise_std=self.router_noise_std,
                )

                if improved:
                    best_step = step
                    best_source_val_acc = metrics["source_val"]["acc1"]
                    target_acc_at_best_source_val = metrics["target"]["acc1"]
                    save_checkpoint(
                        self.output_dir / "best.pt",
                        model=self.model,
                        optimizer=self.optimizer,
                        step=step,
                        metrics=metrics,
                        metadata=self.metadata,
                    )

                last_metrics = metrics
                last_eval_step = step

        if last_metrics is not None:
            save_checkpoint(
                self.output_dir / "last.pt",
                model=self.model,
                optimizer=self.optimizer,
                step=last_eval_step,
                metrics=last_metrics,
                metadata=self.metadata,
            )

        summary = {
            "target_domains": self.loaders["target_domains"],
            "source_domains": self.loaders["source_domains"],
            "train_seed": cfg_get(self.cfg, "experiment.train_seed", 0),
            "split_seed": cfg_get(self.cfg, "experiment.split_seed", 0),
            "best_step": best_step,
            "best_source_val_acc": best_source_val_acc,
            "target_acc_at_best_source_val": target_acc_at_best_source_val,
            "output_dir": str(self.output_dir),
        }
        moa_logging.write_json(self.output_dir / "summary.json", summary)

        print(f"[done] best source_val acc1: {best_source_val_acc:.2f}")
        print(f"[done] target acc1 at best source_val: {target_acc_at_best_source_val:.2f}")
        print(f"[done] best step: {best_step}")
        return summary
