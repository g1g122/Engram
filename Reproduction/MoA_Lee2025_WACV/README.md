# MoA reproduction

Independent reproduction of the WACV 2025 paper [*Domain Generalization using
Large Pretrained Models with Mixture-of-Adapters*](https://doi.org/10.1109/WACV61041.2025.00801).

This repository is not affiliated with or maintained by the original authors.

A reading note for the original paper is available at
`Literature/Technical Methods/Mixture-of-Adapters for domain generalization.md`.

## Example training run

The example trains leave-one-domain-out MoA models on PACS.

Before running, replace every `path/to/your/...` placeholder:

- Set `ROOT_DIR` in `examples/example_run.sh` to this repository's local path.
- Set `data.root` in `examples/example_config.yaml` to the directory containing
  your `PACS/` dataset directory.
- Set `experiment.output_dir` in `examples/example_config.yaml` to where you
  want training outputs to be saved.

```bash
bash path/to/your/MoA_Lee2025_WACV/examples/example_run.sh
```

## PACS checkpoints

The four checkpoints from the example seed are available in the
[Hugging Face model repository](https://huggingface.co/g1g122/MoA_Lee2025_WACV).
Download them manually from **Files and versions**. Their repository layout is:

```text
pacs/
├── art_painting.pt
├── cartoon.pt
├── photo.pt
└── sketch.pt
```

This reproduction provides training code only; it does not include a configured
checkpoint-loading or evaluation workflow. To use the published checkpoints,
implement the corresponding loading and evaluation code in your own workflow.

## Results

| Metric | art_painting | cartoon | photo | sketch | Avg. |
| --- | ---: | ---: | ---: | ---: | ---: |
| target_acc | 98.19 | 98.93 | 99.88 | 92.95 | 97.49 |
