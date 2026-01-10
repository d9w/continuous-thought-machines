"""
Comparison script for Standard CTM vs Hebbian CTM on multiple RL environments.

This script runs both approaches with the same hyperparameters and compares their performance
on CartPole, Acrobot, LunarLander, and MiniGrid FourRooms.
"""
import subprocess
import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))


def get_env_config(env_id):
    """Get environment-specific configuration."""
    configs = {
        "CartPole-v1": {
            "total_timesteps": 100_000,
            "d_model": 64,
            "n_synch_out": 8,
            "iterations": 1,
            "memory_length": 3,
            "mask_velocity": True,
        },
        "Acrobot-v1": {
            "total_timesteps": 200_000,
            "d_model": 64,
            "n_synch_out": 8,
            "iterations": 1,
            "memory_length": 3,
            "mask_velocity": True,
        },
        "LunarLander-v3": {
            "total_timesteps": 500_000,
            "d_model": 128,
            "n_synch_out": 16,
            "iterations": 1,
            "memory_length": 5,
            "mask_velocity": True,
        },
        "MiniGrid-FourRooms-v0": {
            "total_timesteps": 1_000_000,
            "d_model": 128,
            "n_synch_out": 16,
            "iterations": 2,
            "memory_length": 5,
            "mask_velocity": False,
        },
    }
    return configs.get(env_id, configs["CartPole-v1"])


def parse_args():
    parser = argparse.ArgumentParser(description="Compare Standard CTM vs Hebbian CTM")
    parser.add_argument('--env_id', type=str, default="CartPole-v1",
                       choices=["CartPole-v1", "Acrobot-v1", "LunarLander-v3", "MiniGrid-FourRooms-v0"],
                       help='Environment to test on')
    parser.add_argument('--num_runs', type=int, default=3, help='Number of runs for each approach')
    parser.add_argument('--seed_start', type=int, default=42, help='Starting seed')
    # Optional overrides for environment config
    parser.add_argument('--total_timesteps', type=int, default=None, help='Override total timesteps per run')
    parser.add_argument('--d_model', type=int, default=None, help='Override model dimension')
    parser.add_argument('--n_synch_out', type=int, default=None, help='Override number of sync neurons')
    parser.add_argument('--iterations', type=int, default=None, help='Override number of internal ticks')
    parser.add_argument('--memory_length', type=int, default=None, help='Override memory length')
    return parser.parse_args()


def run_experiment(enable_hebbian, seed, env_config, args):
    """Run a single experiment."""
    hebbian_str = "hebbian" if enable_hebbian else "standard"
    print(f"\n{'='*60}")
    print(f"Running {hebbian_str.upper()} CTM with seed {seed}")
    print(f"{'='*60}\n")

    cmd = [
        "python", "tasks/rl/train_hebbian.py",
        "--model_type", "ctm",
        "--enable_hebbian" if enable_hebbian else "--no-enable_hebbian",
        "--env_id", args.env_id,
        "--total_timesteps", str(env_config['total_timesteps']),
        "--seed", str(seed),
        "--d_model", str(env_config['d_model']),
        "--n_synch_out", str(env_config['n_synch_out']),
        "--iterations", str(env_config['iterations']),
        "--memory_length", str(env_config['memory_length']),
        "--run_name", f"comparison_{args.env_id}_{hebbian_str}_seed{seed}",
        "--log_dir", f"logs/comparison/{args.env_id}/{hebbian_str}",
        "--save_every", "20",
    ]

    # Add velocity masking flag
    if env_config['mask_velocity']:
        cmd.append("--mask_velocity")
    else:
        cmd.append("--no-mask_velocity")

    # Add Hebbian-specific parameters
    if enable_hebbian:
        cmd.extend([
            "--hebbian_lr", "0.02",
            "--curiosity_weight", "0.5",
            "--base_noise_variance", "0.02",
        ])

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Error running {hebbian_str} experiment:")
        print(result.stderr)
        return None

    return f"logs/comparison/{args.env_id}/{hebbian_str}"


def parse_tensorboard_logs(log_dir):
    """Parse TensorBoard logs to extract episode returns."""
    from tensorboard.backend.event_processing import event_accumulator

    returns = []
    steps = []

    # Find all event files in the directory
    for root, dirs, files in os.walk(log_dir):
        for file in files:
            if file.startswith('events.out.tfevents'):
                event_file = os.path.join(root, file)
                ea = event_accumulator.EventAccumulator(event_file)
                ea.Reload()

                if 'charts/episodic_return' in ea.Tags()['scalars']:
                    for event in ea.Scalars('charts/episodic_return'):
                        steps.append(event.step)
                        returns.append(event.value)

    return np.array(steps), np.array(returns)


def plot_comparison(standard_results, hebbian_results, env_id, save_path="comparison_results.png"):
    """Plot comparison between standard and Hebbian CTM."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Clean environment name for title
    env_name = env_id.replace('-v', ' v').replace('-', ' ')

    # Plot 1: Learning curves
    ax = axes[0]

    # Plot standard CTM results
    for i, (steps, returns) in enumerate(standard_results):
        if len(steps) > 0:
            ax.plot(steps, returns, 'b-', alpha=0.3, linewidth=1)
            # Moving average
            window = min(10, len(returns))
            if window > 1:
                moving_avg = np.convolve(returns, np.ones(window)/window, mode='valid')
                ax.plot(steps[window-1:], moving_avg, 'b-', alpha=0.8, linewidth=2,
                       label='Standard CTM' if i == 0 else '')

    # Plot Hebbian CTM results
    for i, (steps, returns) in enumerate(hebbian_results):
        if len(steps) > 0:
            ax.plot(steps, returns, 'r-', alpha=0.3, linewidth=1)
            # Moving average
            window = min(10, len(returns))
            if window > 1:
                moving_avg = np.convolve(returns, np.ones(window)/window, mode='valid')
                ax.plot(steps[window-1:], moving_avg, 'r-', alpha=0.8, linewidth=2,
                       label='Hebbian CTM' if i == 0 else '')

    ax.set_xlabel('Environment Steps')
    ax.set_ylabel('Episode Return')
    ax.set_title(f'Learning Curves: {env_name}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Final performance comparison
    ax = axes[1]

    # Compute final performance (last 20% of episodes)
    standard_final = []
    hebbian_final = []

    for steps, returns in standard_results:
        if len(returns) > 0:
            cutoff = int(len(returns) * 0.8)
            standard_final.append(np.mean(returns[cutoff:]))

    for steps, returns in hebbian_results:
        if len(returns) > 0:
            cutoff = int(len(returns) * 0.8)
            hebbian_final.append(np.mean(returns[cutoff:]))

    positions = [1, 2]
    box_data = [standard_final, hebbian_final]
    bp = ax.boxplot(box_data, positions=positions, widths=0.6,
                    patch_artist=True, showmeans=True)

    colors = ['lightblue', 'lightcoral']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    ax.set_xticks(positions)
    ax.set_xticklabels(['Standard CTM', 'Hebbian CTM'])
    ax.set_ylabel('Average Episode Return (Final 20%)')
    ax.set_title('Final Performance Comparison')
    ax.grid(True, alpha=0.3, axis='y')

    # Add statistical summary
    if len(standard_final) > 0 and len(hebbian_final) > 0:
        textstr = f'Standard: {np.mean(standard_final):.1f} ± {np.std(standard_final):.1f}\n'
        textstr += f'Hebbian: {np.mean(hebbian_final):.1f} ± {np.std(hebbian_final):.1f}'
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to {save_path}")
    plt.show()

    return standard_final, hebbian_final


def main():
    args = parse_args()

    # Get environment-specific configuration
    env_config = get_env_config(args.env_id)

    # Apply any command-line overrides
    if args.total_timesteps is not None:
        env_config['total_timesteps'] = args.total_timesteps
    if args.d_model is not None:
        env_config['d_model'] = args.d_model
    if args.n_synch_out is not None:
        env_config['n_synch_out'] = args.n_synch_out
    if args.iterations is not None:
        env_config['iterations'] = args.iterations
    if args.memory_length is not None:
        env_config['memory_length'] = args.memory_length

    print("="*60)
    print("COMPARING STANDARD CTM VS HEBBIAN CTM")
    print("="*60)
    print(f"Environment: {args.env_id}")
    print(f"Total timesteps per run: {env_config['total_timesteps']}")
    print(f"Number of runs: {args.num_runs}")
    print(f"Model config: d_model={env_config['d_model']}, n_synch_out={env_config['n_synch_out']}")
    print(f"Iterations: {env_config['iterations']}, Memory length: {env_config['memory_length']}")
    print(f"Mask velocity: {env_config['mask_velocity']}")
    print("="*60)

    # Run experiments
    standard_log_dirs = []
    hebbian_log_dirs = []

    for run in range(args.num_runs):
        seed = args.seed_start + run

        # Run standard CTM
        log_dir = run_experiment(enable_hebbian=False, seed=seed, env_config=env_config, args=args)
        if log_dir:
            standard_log_dirs.append(log_dir)

        # Run Hebbian CTM
        log_dir = run_experiment(enable_hebbian=True, seed=seed, env_config=env_config, args=args)
        if log_dir:
            hebbian_log_dirs.append(log_dir)

    # Parse results
    print("\n" + "="*60)
    print("PARSING RESULTS...")
    print("="*60)

    try:
        standard_results = []
        for log_dir in standard_log_dirs:
            steps, returns = parse_tensorboard_logs(log_dir)
            standard_results.append((steps, returns))
            print(f"Standard CTM: {len(returns)} episodes logged")

        hebbian_results = []
        for log_dir in hebbian_log_dirs:
            steps, returns = parse_tensorboard_logs(log_dir)
            hebbian_results.append((steps, returns))
            print(f"Hebbian CTM: {len(returns)} episodes logged")

        # Plot comparison
        print("\n" + "="*60)
        print("GENERATING COMPARISON PLOTS...")
        print("="*60)

        save_path = f"comparison_results_{args.env_id}.png"
        standard_final, hebbian_final = plot_comparison(standard_results, hebbian_results, args.env_id, save_path)

        # Print summary statistics
        print("\n" + "="*60)
        print("SUMMARY STATISTICS")
        print("="*60)
        print(f"\nStandard CTM:")
        print(f"  Mean final performance: {np.mean(standard_final):.2f}")
        print(f"  Std final performance: {np.std(standard_final):.2f}")
        print(f"  Best run: {np.max(standard_final):.2f}")
        print(f"  Worst run: {np.min(standard_final):.2f}")

        print(f"\nHebbian CTM:")
        print(f"  Mean final performance: {np.mean(hebbian_final):.2f}")
        print(f"  Std final performance: {np.std(hebbian_final):.2f}")
        print(f"  Best run: {np.max(hebbian_final):.2f}")
        print(f"  Worst run: {np.min(hebbian_final):.2f}")

        # Statistical comparison
        improvement = ((np.mean(hebbian_final) - np.mean(standard_final)) / np.mean(standard_final)) * 100
        print(f"\nRelative improvement: {improvement:+.2f}%")

    except Exception as e:
        print(f"Error parsing results: {e}")
        print("You can manually check the TensorBoard logs in logs/comparison/")


if __name__ == "__main__":
    main()
