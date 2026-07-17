import argparse
import copy
import math
import shutil
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None

from moa import moa_logging
from moa.data.datasets import get_dataset_class, resolve_domain_indices
from moa.data.loaders import cfg_get
from moa.trainer import DGTrainer


def load_config(path):
    """Load a YAML experiment config."""
    if yaml is None:
        raise ImportError(
            "PyYAML is required to read config files. Install it with `pip install pyyaml`."
        )

    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if cfg is None:
        raise ValueError(f"Empty config file: {path}")
    return cfg


def cfg_set(cfg, path, value):
    """Set a dotted config path on a dict-like config."""
    current = cfg
    parts = path.split(".")
    for part in parts[:-1]:
        if part not in current or current[part] is None:
            current[part] = {}
        current = current[part]
    current[parts[-1]] = value


def parse_domain_items(items):
    """Parse CLI domain values as int indices when possible, else strings."""
    parsed = []
    for item in items:
        try:
            parsed.append(int(item))
        except ValueError:
            parsed.append(item)
    return parsed


def normalize_target_domains(domains, target_domains):
    """Return target domain names from indices or names."""
    indices = resolve_domain_indices(domains, target_domains)
    return [domains[index] for index in sorted(indices)]


def resolve_target_runs(cfg):
    """Resolve config test_domains into a list of leave-one-domain target runs."""
    dataset_name = cfg_get(cfg, "data.dataset")
    if dataset_name is None:
        raise ValueError("Missing config value: data.dataset")

    dataset_cls = get_dataset_class(dataset_name)
    domains = list(dataset_cls.DOMAINS)
    configured_targets = cfg_get(cfg, "data.test_domains", None)

    if configured_targets is None:
        return [[domain] for domain in domains]

    target_names = normalize_target_domains(domains, configured_targets)
    return [target_names]


def target_run_name(target_domains):
    """Create a stable directory name for one target-domain run."""
    return "__".join(str(domain).replace(" ", "_") for domain in target_domains)


def prepare_experiment_dir(output_dir, overwrite=False):
    """Create or reset the experiment output directory."""
    output_dir = Path(output_dir)
    if output_dir.exists() and overwrite:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def mean(values):
    """Return the arithmetic mean of a non-empty sequence."""
    return sum(values) / len(values)


def std(values):
    """Return population standard deviation over seed-level results."""
    if len(values) <= 1:
        return 0.0
    avg = mean(values)
    return math.sqrt(sum((value - avg) ** 2 for value in values) / len(values))


def resolve_seed_pairs(cfg):
    """Resolve paired train/data split seeds for repeated runs."""
    train_seeds = list(cfg_get(cfg, "experiment.train_seeds", [0]))
    split_seeds = list(cfg_get(cfg, "experiment.split_seeds", [0]))

    if not train_seeds:
        raise ValueError("experiment.train_seeds must not be empty.")
    if not split_seeds:
        raise ValueError("experiment.split_seeds must not be empty.")
    if len(train_seeds) != len(split_seeds):
        raise ValueError(
            "experiment.train_seeds and experiment.split_seeds must have "
            f"the same length, got {len(train_seeds)} and {len(split_seeds)}."
        )

    return list(zip(train_seeds, split_seeds))


def run_target_sweep(cfg, output_dir):
    """Run one seed over all configured target-domain settings."""
    target_runs = resolve_target_runs(cfg)
    domain_summaries = {}

    print(f"[main] target runs: {target_runs}")

    for target_domains in target_runs:
        run_name = target_run_name(target_domains)
        run_output_dir = Path(output_dir) / run_name
        run_cfg = copy.deepcopy(cfg)
        cfg_set(run_cfg, "experiment.output_dir", str(run_output_dir))
        cfg_set(run_cfg, "data.test_domains", target_domains)

        print(f"\n[main] target domains: {target_domains}")
        trainer = DGTrainer(run_cfg)
        summary = trainer.run()

        if len(target_domains) == 1:
            summary_key = target_domains[0]
        else:
            summary_key = run_name
        domain_summaries[summary_key] = summary

    experiment_summary = moa_logging.write_experiment_summary(
        output_dir,
        domain_summaries,
    )
    table = moa_logging.format_summary_table(domain_summaries)
    print("\n[summary]")
    print(table)
    return experiment_summary, domain_summaries


def aggregate_seed_summaries(seed_summaries, seed_pairs):
    """Aggregate per-seed leave-one-domain-out summaries."""
    metrics = ("best_source_val_acc", "target_acc_at_best_source_val")
    aggregate = {
        "seed_pairs": [
            {"train_seed": train_seed, "split_seed": split_seed}
            for train_seed, split_seed in seed_pairs
        ],
        "metrics": {},
        "seed_summaries": seed_summaries,
    }

    domains = []
    for seed_summary in seed_summaries.values():
        domains = list(seed_summary.keys())
        break

    for metric in metrics:
        by_domain = {}
        for domain in domains:
            values = [
                seed_summary[domain][metric]
                for seed_summary in seed_summaries.values()
            ]
            by_domain[domain] = {
                "mean": mean(values),
                "std": std(values),
                "values": values,
            }

        per_seed_avgs = [
            mean([
                seed_summary[domain][metric]
                for domain in domains
            ])
            for seed_summary in seed_summaries.values()
        ]
        aggregate["metrics"][metric] = {
            "by_domain": by_domain,
            "avg": {
                "mean": mean(per_seed_avgs),
                "std": std(per_seed_avgs),
                "values": per_seed_avgs,
            },
        }

    return aggregate


def format_mean_std(mean_value, std_value):
    """Format mean/std as a compact table cell."""
    return f"{mean_value:.2f}+/-{std_value:.2f}"


def format_seed_aggregate_table(aggregate):
    """Format cross-seed aggregate source-val/target results."""
    metric_names = [
        ("source_val_acc", "best_source_val_acc"),
        ("target_acc", "target_acc_at_best_source_val"),
    ]
    first_metric = aggregate["metrics"][metric_names[0][1]]
    domains = list(first_metric["by_domain"].keys())
    headers = ["Metric"] + domains + ["Avg."]
    rows = []

    for row_name, metric_key in metric_names:
        metric = aggregate["metrics"][metric_key]
        row = [row_name]
        for domain in domains:
            values = metric["by_domain"][domain]
            row.append(format_mean_std(values["mean"], values["std"]))
        avg = metric["avg"]
        row.append(format_mean_std(avg["mean"], avg["std"]))
        rows.append(row)

    table = [headers] + rows
    widths = [
        max(len(str(row[col_index])) for row in table)
        for col_index in range(len(headers))
    ]

    def fmt_row(row):
        return "| " + " | ".join(
            str(cell).ljust(widths[index])
            for index, cell in enumerate(row)
        ) + " |"

    border = "+-" + "-+-".join("-" * width for width in widths) + "-+"
    lines = [border, fmt_row(headers), border]
    lines.extend(fmt_row(row) for row in rows)
    lines.append(border)
    return "\n".join(lines)


def write_seed_aggregate(output_dir, aggregate):
    """Write cross-seed aggregate results to the experiment root."""
    output_dir = Path(output_dir)
    moa_logging.write_json(output_dir / "aggregate_results.json", aggregate)

    csv_record = {}
    for metric_key, metric in aggregate["metrics"].items():
        for domain, values in metric["by_domain"].items():
            csv_record[f"{metric_key}/{domain}/mean"] = values["mean"]
            csv_record[f"{metric_key}/{domain}/std"] = values["std"]
        csv_record[f"{metric_key}/avg/mean"] = metric["avg"]["mean"]
        csv_record[f"{metric_key}/avg/std"] = metric["avg"]["std"]
    moa_logging.append_csv(output_dir / "aggregate_results.csv", csv_record)

    table = format_seed_aggregate_table(aggregate)
    (output_dir / "aggregate_summary.txt").write_text(table + "\n", encoding="utf-8")
    return table


def run_experiment(cfg):
    """Run one config over all configured seed pairs and target domains."""
    experiment_output_dir = Path(
        cfg_get(cfg, "experiment.output_dir")
        or Path("experiments") / cfg_get(cfg, "experiment.name", "moa_experiment")
    )
    overwrite = bool(cfg_get(cfg, "experiment.overwrite", False))
    experiment_output_dir = prepare_experiment_dir(
        experiment_output_dir,
        overwrite=overwrite,
    )

    moa_logging.write_json(experiment_output_dir / "resolved_config.json", cfg)
    seed_pairs = resolve_seed_pairs(cfg)
    seed_summaries = {}

    print(f"[main] experiment output_dir: {experiment_output_dir}")
    print(f"[main] seed pairs: {seed_pairs}")

    for train_seed, split_seed in seed_pairs:
        seed_output_dir = experiment_output_dir / f"train_seed_{train_seed}"
        if seed_output_dir.exists() and any(seed_output_dir.iterdir()) and not overwrite:
            raise FileExistsError(
                f"Seed output directory already exists and is not empty: {seed_output_dir}. "
                "Use --overwrite or set experiment.overwrite: true to replace it."
            )

        seed_cfg = copy.deepcopy(cfg)
        cfg_set(seed_cfg, "experiment.output_dir", str(seed_output_dir))
        cfg_set(seed_cfg, "experiment.train_seed", train_seed)
        cfg_set(seed_cfg, "experiment.split_seed", split_seed)

        print(f"\n[main] train_seed: {train_seed}, split_seed: {split_seed}")
        _, domain_summaries = run_target_sweep(seed_cfg, seed_output_dir)
        seed_summaries[f"train_seed_{train_seed}"] = domain_summaries

    aggregate = aggregate_seed_summaries(seed_summaries, seed_pairs)
    table = write_seed_aggregate(experiment_output_dir, aggregate)
    print("\n[aggregate]")
    print(table)
    return aggregate


def main():
    parser = argparse.ArgumentParser(description="Run MoA DG experiments.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--test-domains", nargs="*", default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.output_dir is not None:
        cfg_set(cfg, "experiment.output_dir", args.output_dir)
    if args.test_domains is not None:
        if len(args.test_domains) == 0:
            cfg_set(cfg, "data.test_domains", None)
        else:
            cfg_set(cfg, "data.test_domains", parse_domain_items(args.test_domains))
    if args.overwrite:
        cfg_set(cfg, "experiment.overwrite", True)

    run_experiment(cfg)


if __name__ == "__main__":
    main()
