"""
Modified CTM-RL implementation with reward-modulated Hebbian learning for synchronization parameters.

This module extends the standard CTM-RL to replace gradient-based learning of r_ij parameters
with biologically-inspired local learning rules.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional, Dict, Any
from torch.distributions.categorical import Categorical

# Import the original CTM-RL components
from models.ctm_rl import ContinuousThoughtMachineRL
from models.ctm import ContinuousThoughtMachine

# Import our Hebbian learning components
from models.hebbian_components import RewardModulatedHebbianLearner


class CTMHebbianRL(ContinuousThoughtMachineRL):
    """
    CTM-RL with reward-modulated Hebbian learning for synchronization parameters.
    
    This class replaces gradient-based learning of r_ij decay parameters with
    biologically-inspired local learning rules that use reward modulation.
    """
    
    def __init__(self, 
                 envs,
                 # Standard CTM parameters
                 d_model: int = 128,
                 d_input: int = 64,
                 synapse_depth: int = 1,
                 n_synch_out: int = 16,
                 n_synch_action: int = 16,
                 neuron_select_type: str = 'random',
                 iterations: int = 1,
                 memory_length: int = 5,
                 deep_memory: bool = True,
                 memory_hidden_dims: int = 2,
                 dropout: float = 0.0,
                 do_normalisation: bool = False,
                 continuous_state_trace: bool = True,
                 # Hebbian learning parameters
                 enable_hebbian: bool = True,
                 curiosity_lr: float = 0.001,
                 hebbian_lr: float = 0.01,
                 curiosity_weight: float = 1.0,
                 temporal_window: int = 20,
                 temporal_decay: float = 0.9,
                 base_noise_variance: float = 0.01,
                 expansion_rate: float = 1.05,
                 reset_threshold: float = 0.1):
        
        # Initialize the base CTM-RL model
        super().__init__(
            envs=envs,
            d_model=d_model,
            d_input=d_input,
            synapse_depth=synapse_depth,
            n_synch_out=n_synch_out,
            n_synch_action=n_synch_action,
            neuron_select_type=neuron_select_type,
            iterations=iterations,
            memory_length=memory_length,
            deep_memory=deep_memory,
            memory_hidden_dims=memory_hidden_dims,
            dropout=dropout,
            do_normalisation=do_normalisation,
            continuous_state_trace=continuous_state_trace
        )
        
        self.enable_hebbian = enable_hebbian
        
        if self.enable_hebbian:
            # Initialize Hebbian learning systems for output and action synchronization
            self.hebbian_out = RewardModulatedHebbianLearner(
                sync_dim=self.synch_representation_size_out,
                curiosity_lr=curiosity_lr,
                hebbian_lr=hebbian_lr,
                curiosity_weight=curiosity_weight,
                temporal_window=temporal_window,
                temporal_decay=temporal_decay,
                base_variance=base_noise_variance,
                expansion_rate=expansion_rate,
                reset_threshold=reset_threshold
            )
            
            if hasattr(self, 'synch_representation_size_action') and self.synch_representation_size_action > 0:
                self.hebbian_action = RewardModulatedHebbianLearner(
                    sync_dim=self.synch_representation_size_action,
                    curiosity_lr=curiosity_lr,
                    hebbian_lr=hebbian_lr,
                    curiosity_weight=curiosity_weight,
                    temporal_window=temporal_window,
                    temporal_decay=temporal_decay,
                    base_variance=base_noise_variance,
                    expansion_rate=expansion_rate,
                    reset_threshold=reset_threshold
                )
            else:
                self.hebbian_action = None
        
        # Track Hebbian learning metrics
        self.hebbian_metrics = {
            'out': {},
            'action': {}
        }
        self.last_sync_out = None
        self.last_sync_action = None
        
    def forward(self, 
                x: torch.Tensor, 
                ctm_state: Tuple[torch.Tensor, torch.Tensor], 
                track: bool = False,
                reward_signal: Optional[float] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Optional[Dict]]:
        """
        Forward pass with optional Hebbian learning updates.
        
        Args:
            x: Input tensor
            ctm_state: Current CTM state (activated_state, state_trace)  
            track: Whether to track internal dynamics
            reward_signal: Optional reward signal for Hebbian updates
            
        Returns:
            synchronisation_out: Output synchronization representation
            new_ctm_state: Updated CTM state
            tracking_data: Optional tracking information
        """
        activated_state, state_trace = ctm_state
        batch_size = x.shape[0]
        
        # Standard CTM forward pass setup
        if track:
            pre_activations_tracking = []
            post_activations_tracking = []
            attention_tracking = []
            synch_out_tracking = []
            synch_action_tracking = []
            
        # Initialize decay states
        decay_alpha_out = torch.zeros(batch_size, self.synch_representation_size_out, device=x.device)
        decay_beta_out = torch.zeros(batch_size, self.synch_representation_size_out, device=x.device)
        
        if self.synch_representation_size_action:
            decay_alpha_action = torch.zeros(batch_size, self.synch_representation_size_action, device=x.device)
            decay_beta_action = torch.zeros(batch_size, self.synch_representation_size_action, device=x.device)
        
        # Get current decay parameters (potentially with Hebbian updates)
        if self.enable_hebbian and reward_signal is not None:
            # Apply Hebbian updates to decay parameters
            r_out, metrics_out = self._update_hebbian_params('out', reward_signal)
            self.hebbian_metrics['out'] = metrics_out
            
            if self.synch_representation_size_action and self.hebbian_action is not None:
                r_action, metrics_action = self._update_hebbian_params('action', reward_signal)
                self.hebbian_metrics['action'] = metrics_action
            else:
                r_action = self.decay_params_action if hasattr(self, 'decay_params_action') else None
        else:
            # Use standard parameters
            r_out = self.decay_params_out
            r_action = self.decay_params_action if hasattr(self, 'decay_params_action') else None
        
        # Process inputs and run internal iterations
        for stepi in range(self.iterations):
            # Input processing
            current_input = self.input_projector(x)
            
            # Concatenate with current state
            new_activated_state = torch.cat([activated_state, current_input], dim=-1)
            
            # Synapse model processing
            synapse_output = self.synapse_model(new_activated_state)
            
            # Update state trace
            state_trace = torch.cat([state_trace, synapse_output.unsqueeze(-1)], dim=-1)
            if state_trace.shape[-1] > self.memory_length:
                state_trace = state_trace[..., -self.memory_length:]
            
            # Neuron-level models
            current_state = state_trace[..., -1]
            nlm_output = self.neuron_level_models(state_trace.reshape(-1, self.memory_length))
            nlm_output = nlm_output.reshape(batch_size, self.d_model, -1)
            
            # Update activated state
            activated_state = current_state + nlm_output.squeeze(-1)
            
            # Compute synchronization for output
            synchronisation_out, decay_alpha_out, decay_beta_out = self.compute_synchronisation(
                activated_state, decay_alpha_out, decay_beta_out, r_out, synch_type='out'
            )
            
            # Compute synchronization for action (if applicable)
            if self.synch_representation_size_action:
                synchronisation_action, decay_alpha_action, decay_beta_action = self.compute_synchronisation(
                    activated_state, decay_alpha_action, decay_beta_action, r_action, synch_type='action'
                )
            else:
                synchronisation_action = None
            
            # Store last synchronization vectors for Hebbian learning
            if self.enable_hebbian:
                self.last_sync_out = synchronisation_out.detach().clone()
                if synchronisation_action is not None:
                    self.last_sync_action = synchronisation_action.detach().clone()
            
            # Tracking
            if track:
                pre_activations_tracking.append(state_trace[:, :, -1].detach().cpu().numpy())
                post_activations_tracking.append(activated_state.detach().cpu().numpy())
                synch_out_tracking.append(synchronisation_out.detach().cpu().numpy())
                if synchronisation_action is not None:
                    synch_action_tracking.append(synchronisation_action.detach().cpu().numpy())
        
        new_ctm_state = (activated_state, state_trace)
        
        if track:
            tracking_data = {
                'pre_activations': np.array(pre_activations_tracking),
                'post_activations': np.array(post_activations_tracking),
                'synch_out': np.array(synch_out_tracking),
                'synch_action': np.array(synch_action_tracking) if synch_action_tracking else None,
                'hebbian_metrics': self.hebbian_metrics.copy()
            }
            return synchronisation_out, new_ctm_state, tracking_data
        
        return synchronisation_out, new_ctm_state, None
    
    def _update_hebbian_params(self, synch_type: str, reward_signal: float) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Update synchronization parameters using Hebbian learning.
        
        Args:
            synch_type: 'out' or 'action'
            reward_signal: Reward signal for modulation
            
        Returns:
            updated_params: Updated decay parameters
            metrics: Learning metrics
        """
        if synch_type == 'out':
            decay_params = self.decay_params_out
            hebbian_learner = self.hebbian_out
            last_sync = self.last_sync_out
        elif synch_type == 'action' and self.hebbian_action is not None:
            decay_params = self.decay_params_action
            hebbian_learner = self.hebbian_action
            last_sync = self.last_sync_action
        else:
            # Return original parameters if Hebbian learning not available
            return getattr(self, f'decay_params_{synch_type}'), {}
        
        if last_sync is None:
            # No previous synchronization available
            return decay_params, {}
        
        # Apply Hebbian update
        updated_params, metrics = hebbian_learner.update_sync_parameters(
            decay_params, reward_signal, last_sync
        )
        
        return updated_params, metrics
    
    def get_action_and_value(self, 
                           x: torch.Tensor, 
                           ctm_state: Tuple[torch.Tensor, torch.Tensor], 
                           done: torch.Tensor, 
                           action: Optional[torch.Tensor] = None, 
                           track: bool = False,
                           reward_signal: Optional[float] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Optional[Dict], torch.Tensor, torch.Tensor]:
        """
        Get action and value with optional Hebbian learning updates.
        """
        # Get hidden states (synchronization representations)
        hidden, new_ctm_state, tracking_data = self.get_states(x, ctm_state, done, track=track, reward_signal=reward_signal)
        
        # Get action logits and value
        action_logits = self.actor(hidden)
        action_probs = Categorical(logits=action_logits)
        
        if action is None:
            action = action_probs.sample()
        
        value = self.critic(hidden)
        
        return (
            action, 
            action_probs.log_prob(action), 
            action_probs.entropy(), 
            value, 
            new_ctm_state, 
            tracking_data, 
            action_logits, 
            action_probs.probs
        )
    
    def get_states(self, 
                   xs: torch.Tensor, 
                   ctm_state: Tuple[torch.Tensor, torch.Tensor], 
                   done: torch.Tensor, 
                   track: bool = False,
                   reward_signal: Optional[float] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Optional[Dict]]:
        """
        Get states (synchronization representations) with optional Hebbian updates.
        """
        num_envs = xs.shape[1] if xs.dim() > 2 else 1
        
        # Handle episode resets
        if xs.dim() > 2:  # Multiple timesteps
            xs = xs.reshape(-1, *xs.shape[2:])
            done = done.reshape(-1, num_envs)
            
        new_hidden = []
        tracking_data_list = []
        
        for i, (x, d) in enumerate(zip(xs, done)):
            # Reset state for terminated environments
            reset_ctm_state = self._get_hidden_states(ctm_state, d, num_envs)
            
            # Forward pass with optional Hebbian updates
            if not track:
                synchronisation, ctm_state = self.forward(x, reset_ctm_state, track=False, reward_signal=reward_signal)
                tracking_data = None
                new_hidden.append(synchronisation)
            else:
                synchronisation, ctm_state, tracking_data = self.forward(x, reset_ctm_state, track=True, reward_signal=reward_signal)
                tracking_data_list.append(tracking_data)
                new_hidden.append(synchronisation)
        
        combined_hidden = torch.cat(new_hidden) if len(new_hidden) > 1 else new_hidden[0]
        combined_tracking = tracking_data_list[0] if tracking_data_list else None
        
        return combined_hidden, ctm_state, combined_tracking
    
    def get_hebbian_metrics(self) -> Dict[str, Any]:
        """Get current Hebbian learning metrics for monitoring."""
        return {
            'out_metrics': self.hebbian_metrics.get('out', {}),
            'action_metrics': self.hebbian_metrics.get('action', {}),
            'hebbian_enabled': self.enable_hebbian
        }
    
    def cleanup_hebbian_history(self) -> None:
        """Clean up Hebbian learning history to prevent memory leaks."""
        if self.enable_hebbian:
            self.hebbian_out.cleanup()
            if self.hebbian_action is not None:
                self.hebbian_action.cleanup()
    
    def set_hebbian_enabled(self, enabled: bool) -> None:
        """Enable or disable Hebbian learning."""
        self.enable_hebbian = enabled
    
    def get_curiosity_predictors(self) -> Dict[str, nn.Module]:
        """Get curiosity predictor networks for separate optimization if needed."""
        predictors = {}
        if self.enable_hebbian:
            predictors['out'] = self.hebbian_out.curiosity_system
            if self.hebbian_action is not None:
                predictors['action'] = self.hebbian_action.curiosity_system
        return predictors


class HebbianPPOAgent(nn.Module):
    """
    Complete PPO agent with CTM-Hebbian learning integration.
    
    This agent wraps the CTM-Hebbian model and provides the standard PPO interface
    while enabling reward-modulated Hebbian learning for synchronization parameters.
    """
    
    def __init__(self, envs, **ctm_kwargs):
        super().__init__()
        
        # Initialize CTM with Hebbian learning
        self.recurrent_model = CTMHebbianRL(envs, **ctm_kwargs)
        
        # Actor and critic heads remain the same
        action_space_size = envs.single_action_space.n
        self.actor = nn.Linear(self.recurrent_model.synch_representation_size_out, action_space_size)
        self.critic = nn.Linear(self.recurrent_model.synch_representation_size_out, 1)
        
        # Track recent rewards for Hebbian learning
        self.recent_rewards = []
        self.episode_rewards = []
        self.current_episode_reward = 0.0
        
    def get_initial_state(self, batch_size: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get initial CTM state."""
        return self.recurrent_model.get_initial_state(batch_size)
    
    def get_value(self, x: torch.Tensor, ctm_state: Tuple[torch.Tensor, torch.Tensor], done: torch.Tensor) -> torch.Tensor:
        """Get state value."""
        hidden, _, _ = self.get_states(x, ctm_state, done)
        return self.critic(hidden)
    
    def get_states(self, 
                   xs: torch.Tensor, 
                   ctm_state: Tuple[torch.Tensor, torch.Tensor], 
                   done: torch.Tensor, 
                   track: bool = False,
                   reward_signal: Optional[float] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Optional[Dict]]:
        """Get states with optional reward signal for Hebbian learning."""
        return self.recurrent_model.get_states(xs, ctm_state, done, track=track, reward_signal=reward_signal)
    
    def get_action_and_value(self, 
                           x: torch.Tensor, 
                           ctm_state: Tuple[torch.Tensor, torch.Tensor], 
                           done: torch.Tensor, 
                           action: Optional[torch.Tensor] = None, 
                           track: bool = False,
                           reward_signal: Optional[float] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor], Optional[Dict], torch.Tensor, torch.Tensor]:
        """Get action and value with optional reward signal for Hebbian learning."""
        
        # Get hidden representation
        hidden, new_ctm_state, tracking_data = self.get_states(x, ctm_state, done, track=track, reward_signal=reward_signal)
        
        # Get action logits and sample action
        action_logits = self.actor(hidden)
        action_probs = Categorical(logits=action_logits)
        
        if action is None:
            action = action_probs.sample()
        
        # Get state value
        value = self.critic(hidden)
        
        return (
            action,
            action_probs.log_prob(action),
            action_probs.entropy(),
            value,
            new_ctm_state,
            tracking_data,
            action_logits,
            action_probs.probs
        )
    
    def update_reward_history(self, reward: float, done: bool = False) -> None:
        """Update reward history for Hebbian learning."""
        self.recent_rewards.append(reward)
        self.current_episode_reward += reward
        
        if done:
            self.episode_rewards.append(self.current_episode_reward)
            self.current_episode_reward = 0.0
        
        # Keep only recent rewards
        if len(self.recent_rewards) > 100:
            self.recent_rewards = self.recent_rewards[-50:]
    
    def get_recent_average_reward(self, window: int = 10) -> float:
        """Get recent average reward for Hebbian learning."""
        if len(self.recent_rewards) == 0:
            return 0.0
        recent = self.recent_rewards[-window:]
        return sum(recent) / len(recent)
    
    def get_hebbian_metrics(self) -> Dict[str, Any]:
        """Get Hebbian learning metrics."""
        base_metrics = self.recurrent_model.get_hebbian_metrics()
        base_metrics.update({
            'recent_avg_reward': self.get_recent_average_reward(),
            'episode_count': len(self.episode_rewards),
            'current_episode_reward': self.current_episode_reward
        })
        return base_metrics
    
    def cleanup_history(self) -> None:
        """Clean up learning history."""
        self.recurrent_model.cleanup_hebbian_history()
        
        # Keep only recent episode rewards
        if len(self.episode_rewards) > 100:
            self.episode_rewards = self.episode_rewards[-50:]
    
    def parameters_for_ppo(self):
        """Get parameters that should be updated by PPO (excludes Hebbian-managed params)."""
        ppo_params = []
        
        # Include all parameters except decay parameters if Hebbian learning is enabled
        for name, param in self.named_parameters():
            if self.recurrent_model.enable_hebbian and 'decay_params' in name:
                # Skip decay parameters - they're managed by Hebbian learning
                continue
            else:
                ppo_params.append(param)
        
        return ppo_params
    
    def get_curiosity_parameters(self):
        """Get curiosity predictor parameters for separate optimization."""
        if not self.recurrent_model.enable_hebbian:
            return []
        
        curiosity_params = []
        predictors = self.recurrent_model.get_curiosity_predictors()
        
        for predictor in predictors.values():
            curiosity_params.extend(predictor.parameters())
        
        return curiosity_params
