"""
PPO implementation with Hebbian learning support for CTM-RL.
Based on tasks/rl/train.py but extended to support reward-modulated Hebbian learning.
"""
import os
import time
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
from gymnasium.wrappers import NormalizeReward
import argparse
from tqdm import tqdm

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from models.ctm_rl import ContinuousThoughtMachineRL
from models.hebbian_components import RewardModulatedHebbianLearner
from models.lstm_rl import LSTMBaseline
from utils.housekeeping import set_seed
from tasks.rl.envs import MaskVelocityWrapper
from tasks.rl.utils import combine_tracking_data
from tasks.rl.plotting import make_rl_gif
from tasks.image_classification.plotting import plot_neural_dynamics


def parse_args():
    parser = argparse.ArgumentParser(description="Train CTM with RL and optional Hebbian learning.")

    # Model Architecture
    parser.add_argument('--model_type', type=str, default="ctm", choices=['ctm', 'lstm'], help='Model type.')
    parser.add_argument('--enable_hebbian', action=argparse.BooleanOptionalAction, default=False, help='Enable Hebbian learning for synchronization parameters.')
    parser.add_argument('--d_model', type=int, default=128, help='Dimension of the model.')
    parser.add_argument('--d_input', type=int, default=64, help='Dimension of the input projection.')
    parser.add_argument('--synapse_depth', type=int, default=1, help='Depth of U-NET model for synapse. 1=linear.')
    parser.add_argument('--n_synch_out', type=int, default=16, help='Number of neurons for output sync.')
    parser.add_argument('--neuron_select_type', type=str, default='first-last', choices=['first-last'], help='Protocol for selecting neuron subset.')
    parser.add_argument('--iterations', type=int, default=1, help='Number of internal ticks.')
    parser.add_argument('--memory_length', type=int, default=5, help='Length of pre-activation history for NLMs.')
    parser.add_argument('--deep_memory', action=argparse.BooleanOptionalAction, default=True, help='Use deep NLMs.')
    parser.add_argument('--memory_hidden_dims', type=int, default=2, help='Hidden dimensions for deep NLMs.')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout rate.')
    parser.add_argument('--do_normalisation', action=argparse.BooleanOptionalAction, default=False, help='Apply normalization in NLMs.')
    parser.add_argument('--continuous_state_trace', action=argparse.BooleanOptionalAction, default=True, help='Flag to carry over state trace between environment steps.')

    # Hebbian Learning Parameters
    parser.add_argument('--curiosity_lr', type=float, default=0.001, help='Learning rate for curiosity predictor.')
    parser.add_argument('--hebbian_lr', type=float, default=0.01, help='Learning rate for Hebbian updates.')
    parser.add_argument('--curiosity_weight', type=float, default=1.0, help='Weight for curiosity reward vs environment reward.')
    parser.add_argument('--temporal_window', type=int, default=20, help='Window size for temporal reward spreading.')
    parser.add_argument('--temporal_decay', type=float, default=0.9, help='Decay rate for temporal reward spreading.')
    parser.add_argument('--base_noise_variance', type=float, default=0.01, help='Base noise variance for exploration.')
    parser.add_argument('--expansion_rate', type=float, default=1.05, help='Rate of noise expansion when stuck.')
    parser.add_argument('--reset_threshold', type=float, default=0.1, help='Reward threshold for noise variance reset.')

    # Environment Configuration
    parser.add_argument('--env_id', type=str, default="CartPole-v1", help='Environment ID.')
    parser.add_argument('--mask_velocity', action=argparse.BooleanOptionalAction, default=True, help='Mask the velocity components of the observation.')
    parser.add_argument('--max_environment_steps', type=int, default=500, help='The maximum number of environment steps.')

    # Training Configuration
    parser.add_argument('--num_steps', type=int, default=128, help='The number of environment steps to run in each environment per policy rollout.')
    parser.add_argument('--total_timesteps', type=int, default=200_000, help='The combined total of all environment steps (across all batches).')
    parser.add_argument('--num_envs', type=int, default=4, help='The number of parallel game environments.')
    parser.add_argument('--anneal_lr', action=argparse.BooleanOptionalAction, default=True, help='Use learning rate annealing.')
    parser.add_argument('--discount_gamma', type=float, default=0.99, help='The discount factor gamma.')
    parser.add_argument('--gae_lambda', type=float, default=0.95, help='The lambda for the Generalized Advantage Estimation (GAE).')
    parser.add_argument('--num_minibatches', type=int, default=4, help='The number of mini-batches.')
    parser.add_argument('--update_epochs', type=int, default=4, help='The number of epochs to update the policy.')
    parser.add_argument('--norm_adv', action=argparse.BooleanOptionalAction, default=True, help='Toggle advantages normalization.')
    parser.add_argument('--clip_coef', type=float, default=0.2, help='The surrogate clipping coefficient.')
    parser.add_argument('--clip_vloss', action=argparse.BooleanOptionalAction, default=True, help='Use clipped loss for the value function (as per the PPO paper).')
    parser.add_argument('--ent_coef', type=float, default=0.01, help='Entropy coefficient.')
    parser.add_argument('--vf_coef', type=float, default=0.5, help='Value function coefficient.')
    parser.add_argument('--max_grad_norm', type=float, default=0.5, help='The maximum norm for gradient clipping.')
    parser.add_argument('--target_kl', type=float, default=None, help='Target KL divergence threshold.')
    parser.add_argument('--lr', type=float, default=2.5e-4, help='Learning rate.')

    # Housekeeping
    parser.add_argument('--log_dir', type=str, default='logs/rl/hebbian_comparison', help='Directory for logging.')
    parser.add_argument('--run_name', type=str, default='hebbian_experiment', help='Name of the run for logging and tracking.')
    parser.add_argument('--save_every', type=int, default=50, help='Save checkpoint frequency.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--device', type=int, nargs='+', default=[-1], help='GPU(s) or -1 for CPU.')

    args = parser.parse_args()
    return args


def make_env_classic_control(env_id, max_environment_steps, mask_velocity=True, render_mode=None):
    def thunk():
        env = gym.make(env_id, render_mode=render_mode)
        if mask_velocity:
            env = MaskVelocityWrapper(env)
        env = NormalizeReward(env, gamma=0.99, epsilon=1e-8)
        env = gym.wrappers.TimeLimit(env, max_episode_steps=max_environment_steps)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env
    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class HebbianAgent(nn.Module):
    """Agent wrapper with optional Hebbian learning support."""

    def __init__(self, size_action_space, args, device):
        super().__init__()

        self.continuous_state_trace = args.continuous_state_trace
        self.device = device
        self.model_type = args.model_type
        self.enable_hebbian = args.enable_hebbian

        # Determine backbone type
        backbone_type = 'classic-control-backbone'

        if args.model_type == "ctm":
            self.recurrent_model = ContinuousThoughtMachineRL(
                iterations=args.iterations,
                d_model=args.d_model,
                d_input=args.d_input,
                n_synch_out=args.n_synch_out,
                synapse_depth=args.synapse_depth,
                memory_length=args.memory_length,
                deep_nlms=args.deep_memory,
                memory_hidden_dims=args.memory_hidden_dims,
                do_layernorm_nlm=args.do_normalisation,
                backbone_type=backbone_type,
                prediction_reshaper=[-1],
                dropout=args.dropout,
                neuron_select_type=args.neuron_select_type,
            )
            actor_input_dim = critic_input_dim = self.recurrent_model.synch_representation_size_out

            # Initialize Hebbian learner if enabled
            if self.enable_hebbian:
                self.hebbian_learner = RewardModulatedHebbianLearner(
                    sync_dim=self.recurrent_model.synch_representation_size_out,
                    curiosity_lr=args.curiosity_lr,
                    hebbian_lr=args.hebbian_lr,
                    curiosity_weight=args.curiosity_weight,
                    temporal_window=args.temporal_window,
                    temporal_decay=args.temporal_decay,
                    base_variance=args.base_noise_variance,
                    expansion_rate=args.expansion_rate,
                    reset_threshold=args.reset_threshold
                )
                self.recent_rewards = []
            else:
                self.hebbian_learner = None
        else:
            self.recurrent_model = LSTMBaseline(
                iterations=args.iterations,
                d_model=args.d_model,
                d_input=args.d_input,
                backbone_type=backbone_type,
            )
            actor_input_dim = critic_input_dim = args.d_model
            self.hebbian_learner = None

        self.actor = nn.Sequential(
            layer_init(nn.Linear(actor_input_dim, 64), std=1),
            nn.ReLU(),
            layer_init(nn.Linear(64, 64), std=1),
            nn.ReLU(),
            layer_init(nn.Linear(64, size_action_space), std=1)
        )

        self.critic = nn.Sequential(
            layer_init(nn.Linear(critic_input_dim, 64), std=1),
            nn.ReLU(),
            layer_init(nn.Linear(64, 64), std=1),
            nn.ReLU(),
            layer_init(nn.Linear(64, 1), std=1)
        )

    def get_initial_hidden_states(self, batch_size):
        """Get initial hidden states for the recurrent model."""
        device = self.device

        if self.model_type == "ctm":
            state_trace = torch.zeros((batch_size, self.recurrent_model.d_model, self.recurrent_model.memory_length), device=device)
            activated_state_trace = self.recurrent_model.start_activated_trace.unsqueeze(0).repeat(batch_size, 1, 1).to(device)
            return (state_trace, activated_state_trace)
        else:
            return torch.zeros((batch_size, self.recurrent_model.d_model), device=device)

    def get_value(self, x, hidden_states, done):
        """Get value estimate."""
        hidden = self.get_states(x, hidden_states, done)
        return self.critic(hidden)

    def get_states(self, x, hidden_states, done):
        """Get hidden representation from recurrent model."""
        if self.continuous_state_trace:
            hidden, new_hidden_states = self.recurrent_model(x, hidden_states, track=False)
        else:
            # Reset state trace on done
            reset_mask = done.unsqueeze(-1).unsqueeze(-1)
            if self.model_type == "ctm":
                state_trace, activated_state_trace = hidden_states
                state_trace = state_trace * (1 - reset_mask)
                activated_state_trace = activated_state_trace * (1 - reset_mask)
                hidden_states = (state_trace, activated_state_trace)
            else:
                hidden_states = hidden_states * (1 - reset_mask.squeeze(-1))

            hidden, new_hidden_states = self.recurrent_model(x, hidden_states, track=False)

        return hidden

    def get_action_and_value(self, x, hidden_states, done, action=None, env_reward=None):
        """Get action and value with optional Hebbian updates."""
        # Apply Hebbian updates if enabled
        if self.enable_hebbian and self.hebbian_learner is not None and env_reward is not None:
            # Get current synchronization for curiosity computation
            with torch.no_grad():
                if self.continuous_state_trace:
                    sync_out, _ = self.recurrent_model(x, hidden_states, track=False)
                else:
                    # Reset on done
                    reset_mask = done.unsqueeze(-1).unsqueeze(-1)
                    state_trace, activated_state_trace = hidden_states
                    state_trace = state_trace * (1 - reset_mask)
                    activated_state_trace = activated_state_trace * (1 - reset_mask)
                    reset_hidden = (state_trace, activated_state_trace)
                    sync_out, _ = self.recurrent_model(x, reset_hidden, track=False)

                # Update decay parameters with Hebbian learning
                avg_env_reward = env_reward.mean().item() if isinstance(env_reward, torch.Tensor) else float(env_reward)
                updated_params, metrics = self.hebbian_learner.update_sync_parameters(
                    self.recurrent_model.decay_params_out,
                    avg_env_reward,
                    sync_out[0:1]  # Use first sample for prediction
                )

                # Apply updated parameters (in-place)
                with torch.no_grad():
                    self.recurrent_model.decay_params_out.data = updated_params.data

        # Get hidden representation
        hidden = self.get_states(x, hidden_states, done)

        # Get action logits and value
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)

        if action is None:
            action = probs.sample()

        value = self.critic(hidden)

        return action, probs.log_prob(action), probs.entropy(), value

    def update_reward_history(self, reward):
        """Track recent rewards for Hebbian learning."""
        if self.enable_hebbian:
            self.recent_rewards.append(reward)
            if len(self.recent_rewards) > 100:
                self.recent_rewards = self.recent_rewards[-50:]

    def get_hebbian_metrics(self):
        """Get Hebbian learning metrics for logging."""
        if not self.enable_hebbian or self.hebbian_learner is None:
            return {}

        return {
            'noise_variance': self.hebbian_learner.noise_scheduler.get_variance(),
            'recent_avg_reward': np.mean(self.recent_rewards[-10:]) if len(self.recent_rewards) > 0 else 0.0
        }

    def parameters_for_ppo(self):
        """Get parameters that should be updated by PPO (excludes Hebbian-managed params)."""
        if self.enable_hebbian:
            # Exclude decay parameters from PPO updates
            params = []
            for name, param in self.named_parameters():
                if 'decay_params' not in name:
                    params.append(param)
            return params
        else:
            return self.parameters()


def train(args):
    """Main training function."""
    run_name = f"{args.run_name}__hebbian_{args.enable_hebbian}__seed_{args.seed}__{int(time.time())}"
    log_dir = os.path.join(args.log_dir, run_name)
    os.makedirs(log_dir, exist_ok=True)

    writer = SummaryWriter(log_dir)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n" + "\n".join([f"|{key}|{value}|" for key, value in vars(args).items()]),
    )

    # Seeding
    set_seed(args.seed)

    # Device setup
    if args.device[0] == -1:
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{args.device[0]}" if torch.cuda.is_available() else "cpu")

    # Create environments
    envs = gym.vector.SyncVectorEnv(
        [make_env_classic_control(args.env_id, args.max_environment_steps, args.mask_velocity)
         for i in range(args.num_envs)]
    )

    # Initialize agent
    agent = HebbianAgent(envs.single_action_space.n, args, device).to(device)

    # Optimizer (only for PPO-updatable parameters)
    optimizer = optim.Adam(agent.parameters_for_ppo(), lr=args.lr, eps=1e-5)

    # Storage
    batch_size = int(args.num_envs * args.num_steps)
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # Initialize
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    next_hidden_states = agent.get_initial_hidden_states(args.num_envs)
    num_updates = args.total_timesteps // batch_size

    # Training loop
    for update in tqdm(range(1, num_updates + 1), desc="Training"):
        initial_hidden_states = tuple(h.clone() for h in next_hidden_states)

        # Annealing learning rate
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.lr
            optimizer.param_groups[0]["lr"] = lrnow

        # Collect rollout
        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(
                    next_obs, next_hidden_states, next_done
                )
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # Execute action
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            # Update hidden states
            if agent.continuous_state_trace:
                with torch.no_grad():
                    _, next_hidden_states = agent.recurrent_model(obs[step], next_hidden_states, track=False)

            # Update Hebbian learner with environment rewards
            if agent.enable_hebbian:
                agent.update_reward_history(reward.mean())
                # Apply Hebbian update with current reward
                with torch.no_grad():
                    action_heb, _, _, _ = agent.get_action_and_value(
                        next_obs, next_hidden_states, next_done, env_reward=rewards[step]
                    )

            # Log episode statistics (gymnasium vector env format)
            if isinstance(infos, dict) and "episode" in infos:
                episode_info = infos["episode"]
                # episode_info contains arrays with boolean masks for which envs completed
                if "_r" in episode_info:  # newer gymnasium format
                    # _r, _l, _t are boolean masks indicating which envs completed
                    completed_mask = episode_info["_r"]
                    for env_idx in range(len(completed_mask)):
                        if completed_mask[env_idx]:
                            writer.add_scalar("charts/episodic_return", episode_info["r"][env_idx], global_step)
                            writer.add_scalar("charts/episodic_length", episode_info["l"][env_idx], global_step)

            # Fallback for older gymnasium API
            elif "final_info" in infos:
                for info in infos["final_info"]:
                    episode_info = info[0] if isinstance(info, (list, tuple)) and len(info) > 0 else info
                    if episode_info and "episode" in episode_info:
                        writer.add_scalar("charts/episodic_return", episode_info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", episode_info["episode"]["l"], global_step)

        # Bootstrap value
        with torch.no_grad():
            next_value = agent.get_value(next_obs, next_hidden_states, next_done).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.discount_gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.discount_gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # Flatten batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Prepare hidden states for training
        if agent.model_type == "ctm":
            b_hidden_states = tuple(h.repeat_interleave(args.num_steps, dim=0) for h in initial_hidden_states)
        else:
            b_hidden_states = initial_hidden_states.repeat_interleave(args.num_steps, dim=0)

        # Optimize policy and value network
        b_inds = np.arange(batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, batch_size // args.num_minibatches):
                end = start + batch_size // args.num_minibatches
                mb_inds = b_inds[start:end]

                if agent.model_type == "ctm":
                    mb_hidden_states = tuple(h[mb_inds] for h in b_hidden_states)
                else:
                    mb_hidden_states = b_hidden_states[mb_inds]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds],
                    mb_hidden_states,
                    torch.zeros(len(mb_inds)).to(device),
                    b_actions.long()[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters_for_ppo(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # Logging
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

        # Log Hebbian metrics
        if agent.enable_hebbian:
            heb_metrics = agent.get_hebbian_metrics()
            writer.add_scalar("hebbian/noise_variance", heb_metrics.get('noise_variance', 0), global_step)
            writer.add_scalar("hebbian/recent_avg_reward", heb_metrics.get('recent_avg_reward', 0), global_step)

        # Save checkpoint
        if update % args.save_every == 0:
            checkpoint_path = os.path.join(log_dir, f"checkpoint_{update}.pt")
            torch.save({
                'agent_state_dict': agent.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'global_step': global_step,
                'args': args
            }, checkpoint_path)

    envs.close()
    writer.close()

    return log_dir


if __name__ == "__main__":
    args = parse_args()
    train(args)
