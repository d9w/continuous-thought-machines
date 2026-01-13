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
import wandb
from gymnasium.wrappers import NormalizeReward
import minigrid
from minigrid.wrappers import ImgObsWrapper
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
    parser.add_argument('--num_steps', type=int, default=100, help='The number of environment steps to run in each environment per policy rollout.')
    parser.add_argument('--total_timesteps', type=int, default=1_000_000, help='The combined total of all environment steps (across all batches).')
    parser.add_argument('--num_envs', type=int, default=8, help='The number of parallel game environments.')
    parser.add_argument('--anneal_lr', action=argparse.BooleanOptionalAction, default=True, help='Use learning rate annealing.')
    parser.add_argument('--discount_gamma', type=float, default=0.99, help='The discount factor gamma.')
    parser.add_argument('--gae_lambda', type=float, default=0.95, help='The lambda for the Generalized Advantage Estimation (GAE).')
    parser.add_argument('--num_minibatches', type=int, default=4, help='The number of mini-batches.')
    parser.add_argument('--update_epochs', type=int, default=1, help='The number of epochs to update the policy.')
    parser.add_argument('--norm_adv', action=argparse.BooleanOptionalAction, default=True, help='Toggle advantages normalization.')
    parser.add_argument('--clip_coef', type=float, default=0.1, help='The surrogate clipping coefficient.')
    parser.add_argument('--clip_vloss', action=argparse.BooleanOptionalAction, default=False, help='Use clipped loss for the value function (as per the PPO paper).')
    parser.add_argument('--ent_coef', type=float, default=0.1, help='Entropy coefficient.')
    parser.add_argument('--vf_coef', type=float, default=0.25, help='Value function coefficient.')
    parser.add_argument('--max_grad_norm', type=float, default=0.5, help='The maximum norm for gradient clipping.')
    parser.add_argument('--target_kl', type=float, default=None, help='Target KL divergence threshold.')
    parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate.')

    # Housekeeping
    parser.add_argument('--log_dir', type=str, default='logs/rl/hebbian_comparison', help='Directory for logging.')
    parser.add_argument('--wandb_project', type=str, default='ctm-rl', help='Weights & Biases project name.')
    parser.add_argument('--wandb_entity', type=str, default=None, help='Weights & Biases entity (username or team).')
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


def make_env_minigrid(env_id, max_environment_steps):
    def thunk():
        env = gym.make(env_id, max_steps=max_environment_steps, render_mode="rgb_array")
        env = ImgObsWrapper(env)
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

        # Determine backbone type based on environment
        if "MiniGrid" in args.env_id:
            backbone_type = 'navigation-backbone'
        else:
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

    def get_initial_state(self, batch_size):
        """Get initial hidden states for the recurrent model."""
        device = self.device

        if self.model_type == "ctm":
            initial_state_trace = torch.repeat_interleave(self.recurrent_model.start_trace.unsqueeze(0), batch_size, 0)
            initial_activated_state_trace = torch.repeat_interleave(self.recurrent_model.start_activated_trace.unsqueeze(0), batch_size, 0)
            return (initial_state_trace, initial_activated_state_trace)
        else:
            initial_hidden_state = torch.repeat_interleave(self.recurrent_model.start_hidden_state.unsqueeze(0), batch_size, 0)
            initial_cell_state = torch.repeat_interleave(self.recurrent_model.start_cell_state.unsqueeze(0), batch_size, 0)
            return (initial_hidden_state, initial_cell_state)

    def _get_hidden_states(self, state, done, num_envs):
        """Get hidden states with proper reset on done."""
        if self.model_type == "ctm":
            return self._get_ctm_hidden_states(state, done, num_envs)
        elif self.model_type == "lstm":
            return self._get_lstm_hidden_states(state, done, num_envs)
        else:
            raise ValueError("Model type not supported.")

    def _get_lstm_hidden_states(self, lstm_state, done, num_envs):
        initial_hidden_state, initial_cell_state = self.get_initial_state(num_envs)
        # Assuming continuous hidden states
        masked_previous_hidden_state = (1.0 - done).view(-1, 1) * lstm_state[0]
        masked_previous_cell_state_state = (1.0 - done).view(-1, 1) * lstm_state[1]
        masked_initial_hidden_state = done.view(-1, 1) * initial_hidden_state
        masked_initial_cell_state = done.view(-1, 1) * initial_cell_state
        return (masked_previous_hidden_state + masked_initial_hidden_state), (masked_previous_cell_state_state + masked_initial_cell_state)

    def _get_ctm_hidden_states(self, ctm_state, done, num_envs):
        initial_state_trace, initial_activated_state_trace = self.get_initial_state(num_envs)
        if self.continuous_state_trace:
            masked_previous_state_trace = (1.0 - done).view(-1, 1, 1) * ctm_state[0]
            masked_previous_activated_state_trace = (1.0 - done).view(-1, 1, 1) * ctm_state[1]
            masked_initial_state_trace = done.view(-1, 1, 1) * initial_state_trace
            masked_initial_activated_state_trace = done.view(-1, 1, 1) * initial_activated_state_trace
            return (masked_previous_state_trace + masked_initial_state_trace), (masked_previous_activated_state_trace + masked_initial_activated_state_trace)
        else:
            return (initial_state_trace, initial_activated_state_trace)

    def get_states(self, x, ctm_state, done, track=False):
        """Get hidden representation from recurrent model."""
        num_envs = ctm_state[0].shape[0]

        if len(x.shape) == 4:
            _, C, H, W = x.shape
            xs = x.reshape((-1, num_envs, C, H, W))
        elif len(x.shape) == 2:
            _, C = x.shape
            xs = x.reshape((-1, num_envs, C))
        else:
            raise ValueError("Input shape not supported.")

        done = done.reshape((-1, num_envs))
        new_hidden = []
        for x_step, d in zip(xs, done):
            if not track:
                synchronisation, ctm_state = self.recurrent_model(x_step, self._get_hidden_states(ctm_state, d, num_envs))
                tracking_data = None
                new_hidden += [synchronisation]
            else:
                synchronisation, ctm_state, pre_activations, post_activations = self.recurrent_model(x_step, self._get_hidden_states(ctm_state, d, num_envs), track=True)
                tracking_data = {
                    'pre_activations': pre_activations,
                    'post_activations': post_activations,
                    'synchronisation': synchronisation.detach().cpu().numpy(),
                }
                new_hidden += [synchronisation]

        return torch.cat(new_hidden), ctm_state, tracking_data

    def get_value(self, x, ctm_state, done):
        """Get value estimate."""
        hidden, _, _ = self.get_states(x, ctm_state, done)
        return self.critic(hidden)

    def get_action_and_value(self, x, ctm_state, done, action=None, track=False):
        """Get action and value."""
        hidden, ctm_state, tracking_data = self.get_states(x, ctm_state, done, track=track)
        action_logits = self.actor(hidden)
        action_probs = Categorical(logits=action_logits)

        if action is None:
            action = action_probs.sample()

        value = self.critic(hidden)

        return action, action_probs.log_prob(action), action_probs.entropy(), value, ctm_state, tracking_data, action_logits, action_probs.probs

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


def train(args):
    """Main training function."""
    run_name = f"{args.run_name}__hebbian_{args.enable_hebbian}__seed_{args.seed}__{int(time.time())}"
    log_dir = os.path.join(args.log_dir, run_name)
    os.makedirs(log_dir, exist_ok=True)

    # Initialize Weights & Biases
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=run_name,
        config=vars(args),
        dir=log_dir,
    )
    print(f"Logging to Weights & Biases: {wandb.run.url}")

    # Seeding
    set_seed(args.seed)

    # Device setup
    if args.device[0] == -1:
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{args.device[0]}" if torch.cuda.is_available() else "cpu")

    # Create environments
    if "MiniGrid" in args.env_id:
        envs = gym.vector.SyncVectorEnv(
            [make_env_minigrid(args.env_id, args.max_environment_steps)
             for i in range(args.num_envs)]
        )
    else:
        envs = gym.vector.SyncVectorEnv(
            [make_env_classic_control(args.env_id, args.max_environment_steps, args.mask_velocity)
             for i in range(args.num_envs)]
        )

    # Initialize agent
    agent = HebbianAgent(envs.single_action_space.n, args, device).to(device)

    # Optimizer - exclude decay parameters if Hebbian learning is enabled
    if args.enable_hebbian and agent.enable_hebbian:
        # Separate PPO parameters from Hebbian-managed decay parameters
        ppo_params = []
        for name, param in agent.named_parameters():
            if 'decay_params' in name:
                # Detach decay parameters from computational graph - no gradients should flow
                param.requires_grad = False
                print(f"Excluding from PPO optimizer (Hebbian-managed): {name} [requires_grad=False]")
                continue
            else:
                ppo_params.append(param)
        optimizer = optim.Adam(ppo_params, lr=args.lr, eps=1e-5)
        print(f"PPO optimizer: {len(ppo_params)} parameter groups (decay_params excluded)")
    else:
        # Standard PPO - update all parameters via gradients
        optimizer = optim.Adam(agent.parameters(), lr=args.lr, eps=1e-5)
        print(f"PPO optimizer: all parameters included (no Hebbian learning)")

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
    next_state = agent.get_initial_state(args.num_envs)
    num_updates = args.total_timesteps // batch_size

    # Training loop
    for update in tqdm(range(1, num_updates + 1), desc="Training"):
        initial_state = (next_state[0].clone(), next_state[1].clone())

        # Annealing learning rate
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.lr
            optimizer.param_groups[0]["lr"] = lrnow

        # Collect rollout
        for step in range(0, args.num_steps):
            next_obs = torch.Tensor(next_obs).to(device)
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value, next_state, _, _, _ = agent.get_action_and_value(
                    next_obs, next_state, next_done
                )
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # Execute action
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            # Apply Hebbian updates to synchronization decay parameters
            if agent.enable_hebbian and agent.hebbian_learner is not None:
                # Update reward history
                agent.update_reward_history(reward.mean())

                # Apply Hebbian updates to decay parameters based on reward
                # We need to access the current synchronization state from the last forward pass
                # The recurrent model should have stored this internally
                with torch.no_grad():
                    # Get the decay parameters from the CTM model
                    if hasattr(agent.recurrent_model, 'decay_params_out'):
                        decay_params = agent.recurrent_model.decay_params_out

                        # Create a dummy synchronization vector for the Hebbian learner
                        # In a proper implementation, we'd track the actual sync from the forward pass
                        # For now, use a placeholder based on the current hidden state
                        current_sync = next_state[0].mean(dim=0)  # Average across batch

                        # Apply Hebbian update
                        updated_params, heb_metrics = agent.hebbian_learner.update_sync_parameters(
                            decay_params,
                            reward.mean().item(),
                            current_sync.unsqueeze(0)
                        )

                        # Update the decay parameters in-place (no gradients)
                        agent.recurrent_model.decay_params_out.data = updated_params.data

            # Log episode statistics
            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info[0]:
                        wandb.log({
                            "charts/episodic_return": info[0]["episode"]["r"],
                            "charts/episodic_length": info[0]["episode"]["l"],
                        }, step=global_step)

            elif "episode" in infos:
                if infos["episode"]:
                    episode_rewards = infos["episode"]["r"]
                    episode_lengths = infos["episode"]["l"]
                    completed_episodes = infos["episode"]["_r"]
                    for env_idx in range(len(completed_episodes)):
                        if completed_episodes[env_idx]:
                            wandb.log({
                                "charts/episodic_return": episode_rewards[env_idx],
                                "charts/episodic_length": episode_lengths[env_idx],
                            }, step=global_step)

        # Bootstrap value
        with torch.no_grad():
            next_value = agent.get_value(next_obs, next_state, next_done).reshape(1, -1)
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
        b_dones = dones.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimize policy and value network
        assert args.num_envs % args.num_minibatches == 0
        envsperbatch = args.num_envs // args.num_minibatches
        envinds = np.arange(args.num_envs)
        flatinds = np.arange(batch_size).reshape(args.num_steps, args.num_envs)
        clipfracs = []
        for epoch in range(args.update_epochs):
            for start in range(0, args.num_envs, envsperbatch):
                end = start + envsperbatch
                mbenvinds = envinds[start:end]
                mb_inds = flatinds[:, mbenvinds].ravel()

                if agent.model_type == "ctm":
                    selected_hidden_state = (initial_state[0][mbenvinds,:,:], initial_state[1][mbenvinds,:,:])
                elif agent.model_type == "lstm":
                    selected_hidden_state = (initial_state[0][mbenvinds,:], initial_state[1][mbenvinds,:])

                _, newlogprob, entropy, newvalue, _, _, _, _ = agent.get_action_and_value(
                    b_obs[mb_inds],
                    selected_hidden_state,
                    b_dones[mb_inds],
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

                # Verify that decay parameters don't receive gradients when Hebbian learning is enabled
                if args.enable_hebbian and agent.enable_hebbian and update == 1 and epoch == 0 and start == 0:
                    # One-time verification on first update
                    for name, param in agent.named_parameters():
                        if 'decay_params' in name and param.grad is not None:
                            grad_norm = param.grad.norm().item()
                            if grad_norm > 1e-8:
                                print(f"WARNING: {name} received gradients (norm={grad_norm:.6f}) but should be Hebbian-only!")
                            else:
                                print(f"✓ Verified: {name} has zero/negligible gradients (norm={grad_norm:.2e})")

                # Clip gradients only for parameters being optimized by PPO
                if args.enable_hebbian and agent.enable_hebbian:
                    nn.utils.clip_grad_norm_(ppo_params, args.max_grad_norm)
                else:
                    nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)

                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # Logging
        log_dict = {
            "charts/learning_rate": optimizer.param_groups[0]["lr"],
            "losses/value_loss": v_loss.item(),
            "losses/policy_loss": pg_loss.item(),
            "losses/entropy": entropy_loss.item(),
            "losses/old_approx_kl": old_approx_kl.item(),
            "losses/approx_kl": approx_kl.item(),
            "losses/clipfrac": np.mean(clipfracs),
            "losses/explained_variance": explained_var,
            "charts/SPS": int(global_step / (time.time() - start_time)),
        }

        # Log Hebbian metrics
        if agent.enable_hebbian and agent.hebbian_learner is not None:
            heb_metrics = agent.get_hebbian_metrics()
            log_dict["hebbian/noise_variance"] = heb_metrics.get('noise_variance', 0)
            log_dict["hebbian/recent_avg_reward"] = heb_metrics.get('recent_avg_reward', 0)

            # Log decay parameter statistics
            if hasattr(agent.recurrent_model, 'decay_params_out'):
                decay_params = agent.recurrent_model.decay_params_out
                log_dict["hebbian/decay_params_mean"] = decay_params.mean().item()
                log_dict["hebbian/decay_params_std"] = decay_params.std().item()
                log_dict["hebbian/decay_params_min"] = decay_params.min().item()
                log_dict["hebbian/decay_params_max"] = decay_params.max().item()

        wandb.log(log_dict, step=global_step)

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
    wandb.finish()

    return log_dir


if __name__ == "__main__":
    args = parse_args()
    train(args)
