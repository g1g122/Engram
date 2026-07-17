import csv
import json
from pathlib import Path


def to_serializable(value):
    """Convert common numeric/tensor-like values into JSON-safe values."""
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        value = value.detach().cpu()
        if value.numel() == 1:
            return value.item()
        return value.tolist()
    if isinstance(value, dict):
        return {
            key: to_serializable(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [to_serializable(item) for item in value]
    return value


def write_json(path, data):
    """Write JSON with stable formatting."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(to_serializable(data), f, indent=2, sort_keys=True)
        f.write("\n")


def append_jsonl(path, record):
    """Append one JSON object as one line."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(to_serializable(record), sort_keys=True) + "\n")


def flatten_dict(data, prefix=""):
    """Flatten nested dicts into slash-separated keys for CSV logging."""
    flat = {}
    for key, value in data.items():
        name = f"{prefix}/{key}" if prefix else str(key)
        value = to_serializable(value)
        if isinstance(value, dict):
            flat.update(flatten_dict(value, name))
        else:
            flat[name] = value
    return flat


def append_csv(path, record):
    """Append one flattened record to a CSV file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    flat_record = flatten_dict(record)
    write_header = not path.exists()

    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(flat_record.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(flat_record)


def write_run_config(output_dir, cfg, metadata=None):
    """Save the resolved config and optional setup metadata for one run."""
    data = {
        "config": cfg,
        "metadata": metadata or {},
    }
    write_json(Path(output_dir) / "config.json", data)


def write_eval_record(output_dir, record):
    """Write one evaluation record to JSONL and CSV logs."""
    output_dir = Path(output_dir)
    append_jsonl(output_dir / "metrics.jsonl", record)
    append_csv(output_dir / "metrics.csv", record)


def write_train_record(output_dir, record):
    """Write one high-frequency train record to JSONL and CSV logs."""
    output_dir = Path(output_dir)
    append_jsonl(output_dir / "train_metrics.jsonl", record)
    append_csv(output_dir / "train_metrics.csv", record)


def format_domain_accs(domain_metrics):
    """Format domain acc1 values as compact name=value strings."""
    return " ".join(
        f"{domain_name}={metrics['acc1']:.2f}"
        for domain_name, metrics in domain_metrics.items()
    )


def format_eval_row(
    step,
    total_steps,
    estimated_epoch,
    train_metrics,
    source_val_metrics,
    target_metrics,
    source_val_by_domain,
    target_by_domain,
    improved,
    aux_weight=None,
    router_noise_std=None,
):
    """Build a one-line evaluation summary for the console."""
    best_mark = "*" if improved else " "
    source_domains = format_domain_accs(source_val_by_domain)
    target_domains = format_domain_accs(target_by_domain)
    schedule_text = ""
    if aux_weight is not None:
        schedule_text += f" aux_w={aux_weight:.6g}"
    if router_noise_std is not None:
        schedule_text += f" noise={router_noise_std:.6g}"

    return (
        f"{best_mark} step {step:05d}/{total_steps:05d} "
        f"epoch={estimated_epoch:.2f} | "
        f"train acc={train_metrics['acc1']:.2f} "
        f"loss={train_metrics['loss']:.4f} | "
        f"source_val acc={source_val_metrics['acc1']:.2f} "
        f"loss={source_val_metrics['loss']:.4f} | "
        f"target acc={target_metrics['acc1']:.2f} "
        f"loss={target_metrics['loss']:.4f} | "
        f"source_val [{source_domains}] | "
        f"target [{target_domains}]"
        f"{schedule_text}"
    )


def format_train_step_row(step, total_steps, train_metrics):
    """Build a compact train-step row for the console."""
    schedule_text = ""
    if "aux_weight" in train_metrics:
        schedule_text += f" aux_w={train_metrics['aux_weight']:.6g}"
    if "router_noise_std" in train_metrics:
        schedule_text += f" noise={train_metrics['router_noise_std']:.6g}"

    return (
        f"step {step:05d}/{total_steps:05d} "
        f"loss={train_metrics['loss']:.4f} "
        f"cls={train_metrics['cls_loss']:.4f} "
        f"aux={train_metrics['aux_loss']:.4f} "
        f"acc1={train_metrics['acc1']:.2f} "
        f"lr={train_metrics['lr']:.6g}"
        f"{schedule_text}"
    )


def final_table_rows(domain_summaries):
    """Create rows for the leave-one-domain-out summary table."""
    domains = list(domain_summaries.keys())

    source_vals = [
        domain_summaries[domain]["best_source_val_acc"]
        for domain in domains
    ]
    target_vals = [
        domain_summaries[domain]["target_acc_at_best_source_val"]
        for domain in domains
    ]

    rows = [
        ("source_val_acc", source_vals),
        ("target_acc", target_vals),
    ]
    return domains, rows


def format_summary_table(domain_summaries):
    """Format final leave-one-domain-out source-val/target summary table."""
    domains, rows = final_table_rows(domain_summaries)
    headers = ["Metric"] + domains + ["Avg."]
    body = []

    for metric_name, values in rows:
        avg = sum(values) / len(values) if values else 0.0
        body.append([metric_name] + [f"{value:.2f}" for value in values] + [f"{avg:.2f}"])

    table = [headers] + body
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
    lines.extend(fmt_row(row) for row in body)
    lines.append(border)
    return "\n".join(lines)


def write_experiment_summary(output_dir, domain_summaries):
    """Save final leave-one-domain-out summary files."""
    output_dir = Path(output_dir)
    domains, rows = final_table_rows(domain_summaries)

    summary = {
        "domains": domain_summaries,
        "metrics": {},
    }
    for metric_name, values in rows:
        summary["metrics"][metric_name] = {
            "by_domain": {
                domain: value
                for domain, value in zip(domains, values)
            },
            "avg": sum(values) / len(values) if values else 0.0,
        }

    write_json(output_dir / "results.json", summary)

    csv_record = {}
    for metric_name, values in rows:
        for domain, value in zip(domains, values):
            csv_record[f"{metric_name}/{domain}"] = value
        csv_record[f"{metric_name}/avg"] = (
            sum(values) / len(values) if values else 0.0
        )
    append_csv(output_dir / "results.csv", csv_record)

    table = format_summary_table(domain_summaries)
    summary_txt = output_dir / "summary.txt"
    summary_txt.parent.mkdir(parents=True, exist_ok=True)
    summary_txt.write_text(table + "\n", encoding="utf-8")
    return summary
