"""
TDANN (Margalit et al. 2024) Reproduction Entry Point.

This script serves as a CLI to trigger different stages:
    1. `python main.py --stage 1`: Run Pretraining.
    2. `python main.py --stage 2`: Run Shuffling (SwapOpt).
    3. `python main.py --stage 3`: Run Final Training.
    4. `python main.py --analyze`: Run Analysis metrics.
    5. `python main.py --all`: Run full pipeline (1 -> 2 -> 3 -> analyze).

See `configs/default.yaml` for parameters.
"""

import argparse
from pathlib import Path
import yaml
import torch
import matplotlib.pyplot as plt

# Import stage functions
from stages.stage1_pretrain import run_stage1
from stages.stage2_shuffle import run_stage2
from stages.stage3_train import run_stage3

# Import analysis functions
from analysis.v1_metrics import (
    extract_v1_responses_with_positions, compute_opm_with_positions,
    detect_pinwheels_physical, compute_pinwheel_density,
    compute_smoothness_profile_physical, plot_opm_physical, plot_smoothness_profile,
    plot_opm_interpolated, filter_by_tuning_strength
)
from analysis.vtc_metrics import (
    compute_category_responses_with_positions, compute_all_selectivity_maps,
    detect_category_patches_physical, compute_rsm,
    plot_all_selectivity_maps_physical, plot_rsm,
    plot_selectivity_map_smoothed, TDANN_SELECTIVITY_THRESHOLD
)
# Note: wiring.py contains wiring length analysis (Figure 7) - not yet implemented


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def run_analysis(
    model_path: str,
    config: dict,
    output_dir: str,
    device: torch.device
):
    """
    Run complete V1, VTC, and Wiring analysis on trained model.

    Args:
        model_path: Path to trained TDANN model checkpoint.
        config: Configuration dictionary.
        output_dir: Directory to save analysis results.
    """
    from models.tdann import create_tdann
    from models.cortical_sheet import CorticalSheet, get_feature_map_sizes
    from data.gratings import create_grating_dataset
    from data.floc import create_floc_dataloader
    from torch.utils.data import DataLoader

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"Loading model from {model_path}")
    layer_configs = config['model']['layers']
    model = create_tdann(layer_configs=layer_configs, use_simclr=True)

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    # Get cortical sheet for physical positions
    cortical_sheet = model.cortical_sheet

    # =========================================================================
    # V1 Analysis
    # =========================================================================
    print("\n=== V1 Analysis ===")

    # Create grating dataset
    grating_dataset = create_grating_dataset()
    grating_loader = DataLoader(grating_dataset, batch_size=64, shuffle=False, num_workers=0)

    # Extract V1 responses with physical positions
    v1_responses, v1_positions, orientations = extract_v1_responses_with_positions(
        model.backbone, cortical_sheet, grating_loader, v1_layer_id='L4', device=device
    )

    # Compute OPM with detailed tuning
    preferred_ori, cv, tuning_mag = compute_opm_with_positions(v1_responses, orientations)

    # Filter to top 25% tuning strength (paper method)
    filtered_ori, filtered_cv, filtered_mag, filtered_pos = filter_by_tuning_strength(
        preferred_ori, cv, tuning_mag, v1_positions, top_percentile=25.0
    )
    print(f"Total units: {len(preferred_ori)}, Top 25%: {len(filtered_ori)}")

    # Detect pinwheels on filtered units
    pinwheels = detect_pinwheels_physical(filtered_pos, filtered_ori)
    v1_area = config['model']['layers']['L4']['surface_area_mm2']
    density = compute_pinwheel_density(pinwheels, v1_area)

    print(f"Pinwheels detected: {len(pinwheels)}")
    print(f"Pinwheel density: {density:.4f} per mm²")
    print(f"Mean CV (all): {cv.mean():.3f}, Mean CV (top 25%): {filtered_cv.mean():.3f}")

    # Save OPM plots
    # 1. Interpolated grid (paper style)
    fig = plot_opm_interpolated(filtered_pos, filtered_ori, pinwheels, 
                                title='V1 OPM (Top 25% Tuning)')
    fig.savefig(output_dir / 'v1_opm_interpolated.png', dpi=150)
    plt.close(fig)
    print(f"Saved: {output_dir / 'v1_opm_interpolated.png'}")

    # 2. Scatter plot (all units)
    fig = plot_opm_physical(v1_positions, preferred_ori, None, 
                            title='V1 OPM (All Units)', point_size=0.5)
    fig.savefig(output_dir / 'v1_opm_scatter.png', dpi=150)
    plt.close(fig)
    print(f"Saved: {output_dir / 'v1_opm_scatter.png'}")

    # Smoothness profile
    distances, similarities = compute_smoothness_profile_physical(
        filtered_pos, filtered_ori, normalize_by_chance=True
    )
    fig = plot_smoothness_profile(distances, similarities, 
                                  title='V1 Smoothness Profile (Normalized)')
    fig.savefig(output_dir / 'v1_smoothness.png', dpi=150)
    plt.close(fig)
    print(f"Saved: {output_dir / 'v1_smoothness.png'}")

    # =========================================================================
    # VTC Analysis
    # =========================================================================
    print("\n=== VTC Analysis ===")

    # Load fLoc data
    floc_path = Path(__file__).parent.parent / 'datasets' / 'fLoc'
    if floc_path.exists():
        floc_loader = create_floc_dataloader(str(floc_path), batch_size=32, num_workers=0)

        # Extract VTC responses with physical positions
        category_responses, vtc_positions = compute_category_responses_with_positions(
            model.backbone, cortical_sheet, floc_loader, vtc_layer_id='L9', device=device
        )

        # Compute selectivity maps
        selectivity_maps = compute_all_selectivity_maps(category_responses)

        # Detect and report patches
        all_patches = {}
        for cat, sel in selectivity_maps.items():
            patches = detect_category_patches_physical(vtc_positions, sel)
            all_patches[cat] = patches
            total_area = sum(p['size_mm2'] for p in patches)
            print(f"{cat.capitalize()}: {len(patches)} patches, total area: {total_area:.1f} mm²")

        # Save selectivity maps (scatter)
        fig = plot_all_selectivity_maps_physical(vtc_positions, selectivity_maps, point_size=0.5)
        fig.savefig(output_dir / 'vtc_selectivity_scatter.png', dpi=150)
        plt.close(fig)
        print(f"Saved: {output_dir / 'vtc_selectivity_scatter.png'}")

        # Save smoothed selectivity maps (paper style)
        for cat, sel in selectivity_maps.items():
            fig = plot_selectivity_map_smoothed(
                vtc_positions, sel, cat, 
                patches=all_patches.get(cat, [])
            )
            fig.savefig(output_dir / f'vtc_{cat}_smoothed.png', dpi=150)
            plt.close(fig)
        print(f"Saved: {output_dir / 'vtc_*_smoothed.png'} (5 files)")

        # Compute and save RSM
        rsm, categories = compute_rsm(category_responses)
        fig = plot_rsm(rsm, categories, title='VTC Representational Similarity')
        fig.savefig(output_dir / 'vtc_rsm.png', dpi=150)
        plt.close(fig)
        print(f"Saved: {output_dir / 'vtc_rsm.png'}")

        # Compute overlap (face-body example)
        from analysis.vtc_metrics import compute_category_overlap
        if 'faces' in selectivity_maps and 'bodies' in selectivity_maps:
            overlap = compute_category_overlap(
                vtc_positions, selectivity_maps['faces'], selectivity_maps['bodies']
            )
            print(f"Face-Body Overlap Score: {overlap:.3f} (lower = more overlap)")

    else:
        print(f"fLoc dataset not found at {floc_path}, skipping VTC analysis")

    # Note: Wiring length analysis (Figure 7 in paper) is not implemented.
    # The paper's method uses k-means clustering to find fiber endpoints
    # and linear sum assignment for optimal matching. See STAR Methods.

    print("\n=== Analysis Complete ===")


def run_full_pipeline(args, config, device):
    """Run the complete TDANN training pipeline."""
    from datasets import load_dataset

    output_base = Path(args.output_dir)

    # Load dataset
    print(f"Loading dataset from {args.dataset_path}...")
    hf_dataset = load_dataset('timm/mini-imagenet', cache_dir=args.dataset_path)

    # Stage 1: Pretrain
    print("\n" + "=" * 50)
    print("STAGE 1: Pretraining")
    print("=" * 50)

    stage1_output = output_base / 'stage1'
    checkpoint_path = run_stage1(
        hf_dataset=hf_dataset['train'],
        config=config,
        output_dir=str(stage1_output),
        use_simclr=True,
        use_pretrained=args.use_pretrained,
        device=device
    )

    # Stage 2: Shuffling
    print("\n" + "=" * 50)
    print("STAGE 2: Shuffling (SwapOpt)")
    print("=" * 50)

    stage2_output = output_base / 'stage2'
    permutation_path = run_stage2(
        checkpoint_path=checkpoint_path,
        config=config,
        output_dir=str(stage2_output),
        device=device
    )

    # Stage 3: Final Training
    print("\n" + "=" * 50)
    print("STAGE 3: Final Training")
    print("=" * 50)

    stage3_output = output_base / 'stage3'
    model_path = run_stage3(
        hf_dataset=hf_dataset['train'],
        permutation_path=permutation_path,
        config=config,
        output_dir=str(stage3_output),
        use_simclr=True,
        device=device
    )

    # Analysis
    print("\n" + "=" * 50)
    print("ANALYSIS")
    print("=" * 50)

    analysis_output = output_base / 'analysis'
    run_analysis(
        model_path=model_path,
        config=config,
        output_dir=str(analysis_output),
        device=device
    )

    print("\n" + "=" * 50)
    print("PIPELINE COMPLETE")
    print("=" * 50)
    print(f"Results saved to: {output_base}")


def main():
    parser = argparse.ArgumentParser(
        description='TDANN Reproduction Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --all --dataset-path ../datasets
  python main.py --stage 1 --dataset-path ../datasets --use-pretrained
  python main.py --stage 2 --checkpoint outputs/stage1/stage1_checkpoint.pt
  python main.py --stage 3 --dataset-path ../datasets --permutation outputs/stage2/permutations.pt
  python main.py --analyze --model outputs/stage3/tdann_final.pt
        """
    )

    # Stage selection
    parser.add_argument('--stage', type=int, choices=[1, 2, 3],
                        help='Run specific stage (1, 2, or 3)')
    parser.add_argument('--analyze', action='store_true',
                        help='Run analysis on trained model')
    parser.add_argument('--all', action='store_true',
                        help='Run full pipeline (stages 1-3 + analysis)')

    # Paths
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='Path to config file')
    parser.add_argument('--dataset-path', type=str,
                        help='Path to HuggingFace dataset cache')
    parser.add_argument('--output-dir', type=str, default='outputs',
                        help='Output directory')

    # Stage-specific arguments
    parser.add_argument('--checkpoint', type=str,
                        help='Stage 1 checkpoint (for Stage 2)')
    parser.add_argument('--permutation', type=str,
                        help='Stage 2 permutation file (for Stage 3)')
    parser.add_argument('--model', type=str,
                        help='Trained model path (for analysis)')
    parser.add_argument('--use-pretrained', action='store_true',
                        help='Use torchvision pretrained weights (Stage 1)')

    # Device
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Run selected stage/pipeline
    if args.all:
        if not args.dataset_path:
            parser.error("--all requires --dataset-path")
        run_full_pipeline(args, config, device)

    elif args.stage == 1:
        if not args.dataset_path:
            parser.error("--stage 1 requires --dataset-path")
        from datasets import load_dataset
        hf_dataset = load_dataset('timm/mini-imagenet', cache_dir=args.dataset_path)
        run_stage1(
            hf_dataset=hf_dataset['train'],
            config=config,
            output_dir=f"{args.output_dir}/stage1",
            use_simclr=True,
            use_pretrained=args.use_pretrained,
            device=device
        )

    elif args.stage == 2:
        if not args.checkpoint:
            parser.error("--stage 2 requires --checkpoint")
        run_stage2(
            checkpoint_path=args.checkpoint,
            config=config,
            output_dir=f"{args.output_dir}/stage2",
            device=device
        )

    elif args.stage == 3:
        if not args.dataset_path or not args.permutation:
            parser.error("--stage 3 requires --dataset-path and --permutation")
        from datasets import load_dataset
        hf_dataset = load_dataset('timm/mini-imagenet', cache_dir=args.dataset_path)
        run_stage3(
            hf_dataset=hf_dataset['train'],
            permutation_path=args.permutation,
            config=config,
            output_dir=f"{args.output_dir}/stage3",
            use_simclr=True,
            device=device
        )

    elif args.analyze:
        if not args.model:
            parser.error("--analyze requires --model")
        run_analysis(
            model_path=args.model,
            config=config,
            output_dir=f"{args.output_dir}/analysis",
            device=device
        )

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
