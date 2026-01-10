import gymnasium as gym
import numpy as np

class MaskVelocityWrapper(gym.Wrapper):
    """
    Simple wrapper that automatically resets the environment on done.
    Modeled after EpisodicLifeEnv but simplified since we don't need
    to handle lives or partial resets.
    """
    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._apply_velocity_mask(obs), info 

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._apply_velocity_mask(obs), reward, terminated, truncated, info

    def _apply_velocity_mask(self, observation):
        gym_id = self.env.spec.id
        if gym_id == "CartPole-v1":
            return self._apply_velocity_mask_cartpole(observation)
        elif gym_id == "Acrobot-v1":
            return self._apply_velocity_mask_acrobot(observation)
        elif gym_id == "LunarLander-v3":
            return self._apply_velocity_mask_lunarlander(observation)
        else:
            raise NotImplementedError(f"Velocity masking not implemented for {gym_id}")

    def _apply_velocity_mask_cartpole(self, observation):
        # Mask velocities (indices 1, 3)
        return observation * np.array([1, 0, 1, 0], dtype="float32")

    def _apply_velocity_mask_acrobot(self, observation):
        # Mask angular velocities (indices 4, 5)
        return observation * np.array([1, 1, 1, 1, 0, 0], dtype="float32")

    def _apply_velocity_mask_lunarlander(self, observation):
        # LunarLander-v3 observation: [x, y, vx, vy, angle, angular_vel, left_leg_contact, right_leg_contact]
        # Mask velocities (indices 2, 3, 5)
        return observation * np.array([1, 1, 0, 0, 1, 0, 1, 1], dtype="float32")
