import argparse
import json
import sys
import tempfile
from pathlib import Path

from PIL import Image
import torch
import torch.nn as nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def make_toy_pacs(root, image_size=32):
    """Create a tiny PACS-like ImageFolder dataset for smoke testing."""
    domains = ["art_painting", "cartoon", "photo", "sketch"]
    classes = ["class0", "class1"]
    colors = {
        "art_painting": (220, 80, 80),
        "cartoon": (80, 220, 80),
        "photo": (80, 80, 220),
        "sketch": (180, 180, 180),
    }

    dataset_root = Path(root) / "PACS"
    for domain in domains:
        for class_index, class_name in enumerate(classes):
            class_dir = dataset_root / domain / class_name
            class_dir.mkdir(parents=True, exist_ok=True)

            for image_index in range(4):
                base = colors[domain]
                offset = class_index * 30 + image_index
                color = tuple(min(255, channel + offset) for channel in base)
                image = Image.new("RGB", (image_size, image_size), color)
                image.save(class_dir / f"{domain}_{class_name}_{image_index}.png")

    return dataset_root


class TinyClassifier(nn.Module):
    """Small image classifier used to avoid downloading CLIP in smoke tests."""

    def __init__(self, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(3, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def tiny_build_image_classifier(num_classes, **kwargs):
    model = TinyClassifier(num_classes=num_classes)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    metadata = {
        "backbone_name": "tiny_smoke_model",
        "n_outputs": 3,
        "num_classes": num_classes,
        "injected_names": [],
    }
    return model, trainable_params, metadata


def build_smoke_config(data_root, output_dir, steps, checkpoint_freq, log_interval):
    return {
        "experiment": {
            "name": "smoke_test",
            "output_dir": str(output_dir),
            "overwrite": True,
            "train_seeds": [0],
            "split_seeds": [0],
        },
        "data": {
            "root": str(data_root),
            "dataset": "PACS",
            "test_domains": ["sketch"],
            "holdout_fraction": 0.25,
            "image_size": 32,
            "normalization": "clip",
            "augment": False,
            "num_workers": 0,
        },
        "train": {
            "steps": steps,
            "checkpoint_freq": checkpoint_freq,
            "batch_size": 2,
            "eval_batch_size": 4,
            "optimizer": "adam",
            "lr": 1.0e-3,
            "weight_decay": 0.0,
            "device": "cpu",
            "amp": False,
            "max_grad_norm": None,
            "log_interval": log_interval,
            "pin_memory": False,
        },
        "model": {
            "backbone": "tiny_smoke_model",
            "pretrained": False,
            "num_experts": 1,
            "phm_dim": 1,
            "ranks": [1],
            "router_temperature": 0.5,
            "router_noise_std": 0.0,
            "aux_weight": 0.0,
            "adapter_scale": 1.0,
            "verbose": False,
        },
    }


def assert_exists(path):
    path = Path(path)
    if not path.exists():
        raise AssertionError(f"Expected file/directory does not exist: {path}")


def run_smoke_test(args):
    import moa.trainer as trainer_mod
    from moa.main import run_experiment

    trainer_mod.build_image_classifier = tiny_build_image_classifier

    with tempfile.TemporaryDirectory(prefix="moa_smoke_") as tmpdir:
        tmpdir = Path(tmpdir)
        data_root = tmpdir / "datasets"
        output_dir = tmpdir / "outputs" / "smoke_test"
        make_toy_pacs(data_root, image_size=32)

        cfg = build_smoke_config(
            data_root=data_root,
            output_dir=output_dir,
            steps=args.steps,
            checkpoint_freq=args.checkpoint_freq,
            log_interval=args.log_interval,
        )

        print(f"[smoke] temp_dir={tmpdir}")
        print("[smoke] running moa.main.run_experiment with tiny model")
        aggregate = run_experiment(cfg)

        run_dir = output_dir / "train_seed_0" / "sketch"
        expected_paths = [
            output_dir / "resolved_config.json",
            output_dir / "aggregate_results.json",
            output_dir / "aggregate_results.csv",
            output_dir / "aggregate_summary.txt",
            output_dir / "train_seed_0" / "results.json",
            output_dir / "train_seed_0" / "results.csv",
            output_dir / "train_seed_0" / "summary.txt",
            run_dir / "config.json",
            run_dir / "metrics.jsonl",
            run_dir / "metrics.csv",
            run_dir / "train_metrics.jsonl",
            run_dir / "train_metrics.csv",
            run_dir / "summary.json",
            run_dir / "best.pt",
            run_dir / "last.pt",
        ]
        for path in expected_paths:
            assert_exists(path)

        with (run_dir / "summary.json").open("r", encoding="utf-8") as f:
            run_summary = json.load(f)

        print("[smoke] run summary:")
        print(json.dumps(run_summary, indent=2, sort_keys=True))
        print("[smoke] aggregate metrics:")
        print(json.dumps(aggregate["metrics"], indent=2, sort_keys=True))

        if args.keep_output:
            keep_dir = PROJECT_ROOT / "smoke_test_output"
            if keep_dir.exists():
                import shutil

                shutil.rmtree(keep_dir)
            import shutil

            shutil.copytree(output_dir, keep_dir)
            print(f"[smoke] copied output to {keep_dir}")

    print("[smoke] OK")


def main():
    parser = argparse.ArgumentParser(description="Run a tiny end-to-end MoA smoke test.")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--checkpoint-freq", type=int, default=2)
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--keep-output", action="store_true")
    args = parser.parse_args()
    run_smoke_test(args)


if __name__ == "__main__":
    main()
