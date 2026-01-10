"""
Reward-Modulated Hebbian Learning Components for CTM

This module implements the core components for replacing gradient-based learning
of synchronization parameters with biologically-inspired local learning rules.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
from typing import Tuple, Optional, Dict, Any


class CuriosityReward(nn.Module):
    """
    Curiosity-driven reward system that generates dense learning signals
    by measuring prediction errors in synchronization patterns.
    """
    
    def __init__(self, sync_dim: int, prediction_hidden: int = 64, history_length: int = 10):
        super().__init__()
        self.sync_dim = sync_dim
        self.history_length = history_length
        
        # Simple MLP predictor for synchronization patterns
        self.predictor = nn.Sequential(
            nn.Linear(sync_dim, prediction_hidden),
            nn.ReLU(),
            nn.Linear(prediction_hidden, prediction_hidden),
            nn.ReLU(),
            nn.Linear(prediction_hidden, sync_dim)
        )
        
        # Buffer to store synchronization history
        self.sync_history = deque(maxlen=history_length)
        self.prediction_errors = deque(maxlen=100)  # Track recent prediction errors
        
    def compute_reward(self, current_sync: torch.Tensor) -> float:
        """
        Compute curiosity reward based on prediction error of synchronization patterns.
        
        Args:
            current_sync: Current synchronization vector [batch_size, sync_dim]
            
        Returns:
            curiosity_reward: Scalar reward value
        """
        if len(self.sync_history) < 2:
            self.sync_history.append(current_sync.detach().clone())
            return 0.0
            
        # Use previous synchronization to predict current one
        previous_sync = self.sync_history[-1]
        
        with torch.no_grad():
            predicted_sync = self.predictor(previous_sync)
            prediction_error = F.mse_loss(predicted_sync, current_sync)
            
        # Store history and error
        self.sync_history.append(current_sync.detach().clone())
        self.prediction_errors.append(prediction_error.item())
        
        # Normalize prediction error to [0, 1] range based on recent history
        if len(self.prediction_errors) > 10:
            max_error = max(self.prediction_errors)
            min_error = min(self.prediction_errors)
            if max_error > min_error:
                normalized_error = (prediction_error.item() - min_error) / (max_error - min_error)
            else:
                normalized_error = 0.5
        else:
            normalized_error = prediction_error.item()
            
        return float(normalized_error)
    
    def update_predictor(self, optimizer: torch.optim.Optimizer) -> Optional[float]:
        """
        Update the curiosity predictor using recent synchronization history.

        Args:
            optimizer: Optimizer for the predictor network

        Returns:
            loss: Training loss if update was performed, None otherwise
        """
        if len(self.sync_history) < 3:
            return None

        # Create training pairs from recent history
        inputs = []
        targets = []

        for i in range(len(self.sync_history) - 1):
            inputs.append(self.sync_history[i])
            targets.append(self.sync_history[i + 1])

        if len(inputs) == 0:
            return None

        # Stack tensors (they're already detached from storage)
        inputs = torch.stack(inputs)
        targets = torch.stack(targets)

        # Ensure inputs don't require grad (they're data, not parameters)
        inputs = inputs.detach()
        targets = targets.detach()

        # Update predictor
        optimizer.zero_grad()
        predictions = self.predictor(inputs)
        loss = F.mse_loss(predictions, targets)

        # Check if loss has grad_fn before backward
        if loss.requires_grad:
            loss.backward()
            optimizer.step()
            return loss.item()
        else:
            # Predictor has no parameters or inputs have no grad
            return 0.0


class AdaptiveNoiseScheduler:
    """
    Adaptive noise scheduling system that increases exploration when stuck
    and resets to focused exploitation when learning occurs.
    """
    
    def __init__(self, 
                 base_variance: float = 0.01,
                 expansion_rate: float = 1.05,
                 reset_threshold: float = 0.1,
                 max_variance: float = 0.5,
                 min_variance: float = 0.001):
        self.base_variance = base_variance
        self.current_variance = base_variance
        self.expansion_rate = expansion_rate
        self.reset_threshold = reset_threshold
        self.max_variance = max_variance
        self.min_variance = min_variance
        
        # Track recent rewards for adaptation decisions
        self.recent_rewards = deque(maxlen=20)
        self.steps_since_good_reward = 0
        
    def update(self, total_reward: float) -> None:
        """
        Update noise variance based on recent reward signal.
        
        Args:
            total_reward: Combined environment + curiosity reward
        """
        self.recent_rewards.append(total_reward)
        
        if total_reward > self.reset_threshold:
            # Good reward received - reset to focused exploitation
            self.current_variance = self.base_variance
            self.steps_since_good_reward = 0
        else:
            # No good reward - increase exploration
            self.steps_since_good_reward += 1
            if self.steps_since_good_reward > 5:  # Wait a few steps before expanding
                self.current_variance = min(
                    self.current_variance * self.expansion_rate,
                    self.max_variance
                )
    
    def sample_noise(self, shape: torch.Size, device: torch.device) -> torch.Tensor:
        """
        Sample Gaussian noise with current variance.
        
        Args:
            shape: Shape of noise tensor to generate
            device: Device to place tensor on
            
        Returns:
            noise: Gaussian noise tensor
        """
        return torch.normal(0, self.current_variance, shape, device=device)
    
    def get_variance(self) -> float:
        """Get current noise variance."""
        return self.current_variance


class TemporalRewardSpreader:
    """
    Temporal reward spreading system for credit assignment without complex gradients.
    Spreads rewards backwards in time with exponential decay.
    """
    
    def __init__(self, window_size: int = 20, decay_rate: float = 0.9):
        self.window_size = window_size
        self.decay_rate = decay_rate
        self.reward_history = deque(maxlen=window_size)
        
    def add_reward(self, reward: float) -> None:
        """Add a new reward to the history."""
        self.reward_history.append(reward)
        
    def get_spread_rewards(self) -> list:
        """
        Compute temporally spread rewards for the current history.
        
        Returns:
            spread_rewards: List of spread reward values
        """
        if len(self.reward_history) == 0:
            return []
            
        spread_rewards = []
        rewards = list(self.reward_history)
        
        for i in range(len(rewards)):
            spread_reward = 0.0
            # Spread future rewards backwards to current timestep
            for j in range(i, len(rewards)):
                time_diff = j - i
                spread_reward += rewards[j] * (self.decay_rate ** time_diff)
            spread_rewards.append(spread_reward)
            
        return spread_rewards
    
    def get_current_spread_reward(self) -> float:
        """Get the temporally spread reward for the most recent timestep."""
        spread_rewards = self.get_spread_rewards()
        return spread_rewards[-1] if spread_rewards else 0.0


class HebbianSyncUpdater:
    """
    Reward-modulated Hebbian updates for synchronization decay parameters.
    Implements local learning rules that strengthen/weaken connections based on reward feedback.
    """
    
    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.9):
        self.learning_rate = learning_rate
        self.momentum = momentum
        
        # Track noise patterns and their associated rewards
        self.noise_history: Dict[int, torch.Tensor] = {}
        self.velocity_history: Dict[int, torch.Tensor] = {}
        
    def update_sync_params(self, 
                          decay_params: torch.Tensor,
                          reward_signal: float,
                          noise_scheduler: AdaptiveNoiseScheduler) -> torch.Tensor:
        """
        Apply reward-modulated Hebbian update to synchronization decay parameters.
        
        Args:
            decay_params: Current decay parameters r_ij
            reward_signal: Reward signal for modulation
            noise_scheduler: Noise scheduler for exploration
            
        Returns:
            updated_params: Updated decay parameters
        """
        param_id = id(decay_params)
        
        # Sample noise for exploration
        noise = noise_scheduler.sample_noise(decay_params.shape, decay_params.device)
        
        # Store noise pattern for potential reinforcement
        self.noise_history[param_id] = noise.clone()
        
        # Apply reward-modulated update
        if reward_signal > 0:
            # Positive reward: reinforce recent noise patterns
            if param_id in self.noise_history:
                # Hebbian-style update: strengthen connections that led to reward
                update = self.learning_rate * reward_signal * self.noise_history[param_id]
                
                # Add momentum
                if param_id in self.velocity_history:
                    self.velocity_history[param_id] = (
                        self.momentum * self.velocity_history[param_id] + update
                    )
                else:
                    self.velocity_history[param_id] = update
                
                # Apply update with momentum
                decay_params.data += self.velocity_history[param_id]
                
        elif reward_signal < -0.1:  # Significant negative reward
            # Negative reward: anti-Hebbian update (weaken recent patterns)
            if param_id in self.noise_history:
                update = -0.5 * self.learning_rate * abs(reward_signal) * self.noise_history[param_id]
                decay_params.data += update
        
        # Add exploration noise
        noisy_params = decay_params + noise
        
        # Ensure parameters stay non-negative (as required by CTM)
        with torch.no_grad():
            decay_params.data = torch.clamp(decay_params.data, min=0.0)
            noisy_params = torch.clamp(noisy_params, min=0.0)
        
        return noisy_params
    
    def cleanup_old_history(self, max_history_size: int = 1000) -> None:
        """Clean up old noise and velocity history to prevent memory leaks."""
        if len(self.noise_history) > max_history_size:
            # Remove oldest entries
            old_keys = list(self.noise_history.keys())[:-max_history_size//2]
            for key in old_keys:
                self.noise_history.pop(key, None)
                self.velocity_history.pop(key, None)


class RewardModulatedHebbianLearner:
    """
    Complete reward-modulated Hebbian learning system that coordinates all components.
    """
    
    def __init__(self,
                 sync_dim: int,
                 curiosity_lr: float = 0.001,
                 hebbian_lr: float = 0.01,
                 curiosity_weight: float = 1.0,
                 temporal_window: int = 20,
                 temporal_decay: float = 0.9,
                 **noise_scheduler_kwargs):

        # Initialize components
        self.curiosity_system = CuriosityReward(sync_dim)
        self.noise_scheduler = AdaptiveNoiseScheduler(**noise_scheduler_kwargs)
        self.temporal_spreader = TemporalRewardSpreader(temporal_window, temporal_decay)
        self.hebbian_updater = HebbianSyncUpdater(hebbian_lr)

        # Optimizer for curiosity predictor
        self.curiosity_optimizer = torch.optim.Adam(
            self.curiosity_system.predictor.parameters(),  # Fixed: predictor is a Sequential, not the module itself
            lr=curiosity_lr
        )

        self.curiosity_weight = curiosity_weight
        
    def compute_total_reward(self, 
                           env_reward: float, 
                           current_sync: torch.Tensor) -> Tuple[float, float]:
        """
        Compute combined environment and curiosity reward.
        
        Args:
            env_reward: Reward from environment
            current_sync: Current synchronization vector
            
        Returns:
            total_reward: Combined reward signal
            curiosity_reward: Just the curiosity component
        """
        curiosity_reward = self.curiosity_system.compute_reward(current_sync)
        
        # Combine rewards (prioritize environment reward when available)
        if env_reward > 0:
            total_reward = env_reward + self.curiosity_weight * curiosity_reward
        else:
            total_reward = self.curiosity_weight * curiosity_reward
            
        return total_reward, curiosity_reward
    
    def update_sync_parameters(self, 
                             decay_params: torch.Tensor,
                             env_reward: float,
                             current_sync: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Perform complete reward-modulated Hebbian update cycle.
        
        Args:
            decay_params: Current synchronization decay parameters
            env_reward: Environment reward signal
            current_sync: Current synchronization vector
            
        Returns:
            updated_params: Updated decay parameters (with noise)
            metrics: Dictionary of learning metrics
        """
        # Compute total reward
        total_reward, curiosity_reward = self.compute_total_reward(env_reward, current_sync)
        
        # Add to temporal reward spreading
        self.temporal_spreader.add_reward(total_reward)
        spread_reward = self.temporal_spreader.get_current_spread_reward()
        
        # Update noise scheduling
        self.noise_scheduler.update(total_reward)
        
        # Apply Hebbian updates with temporal reward
        updated_params = self.hebbian_updater.update_sync_params(
            decay_params, spread_reward, self.noise_scheduler
        )
        
        # Update curiosity predictor periodically
        predictor_loss = self.curiosity_system.update_predictor(self.curiosity_optimizer)
        
        # Return metrics for monitoring
        metrics = {
            'total_reward': total_reward,
            'curiosity_reward': curiosity_reward,
            'spread_reward': spread_reward,
            'noise_variance': self.noise_scheduler.get_variance(),
            'predictor_loss': predictor_loss if predictor_loss is not None else 0.0
        }
        
        return updated_params, metrics
    
    def cleanup(self) -> None:
        """Clean up internal state to prevent memory leaks."""
        self.hebbian_updater.cleanup_old_history()
