import os
import sys
import time
import argparse
import torch
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize, VecMonitor, DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import NormalActionNoise

from env import TrackEnv

class RenderCallback(BaseCallback):
    def __init__(self, env):
        super().__init__()
        self.env = env
    def _on_step(self):
        self.env.envs[0].render()
        return True

def train_visual(timesteps=50000, track="circular"):
    env = DummyVecEnv([lambda: TrackEnv(render_mode="human", track_type=track)])
    env = VecMonitor(env)
    env = VecNormalize(env, norm_obs=True, norm_reward=True)
    model = SAC("MlpPolicy", env, verbose=1,
                learning_rate=5e-4,
                buffer_size=300000,
                learning_starts=2000,
                batch_size=256,
                tau=0.005,
                gamma=0.99,
                train_freq=1,
                gradient_steps=1,
                action_noise=NormalActionNoise(mean=np.zeros(2), sigma=0.1*np.ones(2)),
                policy_kwargs=dict(net_arch=[256,256], activation_fn=torch.nn.ReLU),
                ent_coef='auto')
    model.learn(total_timesteps=timesteps, callback=RenderCallback(env))
    model.save("models/track_racer")
    env.save("models/vec_normalize.pkl")
    env.close()

def demo():
    env = TrackEnv(render_mode="human")
    model = SAC.load("models/track_racer")
    try:
        from stable_baselines3.common.vec_env import VecNormalize
        env = VecNormalize.load("models/vec_normalize.pkl", env)
        env.training = False
    except:
        pass
    ep = 0
    while True:
        obs, _ = env.reset()
        done = False
        ep_reward = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, _ = env.step(action)
            ep_reward += reward
            env.render()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    env.close()
                    return
            if done or truncated:
                break
        print(f"Demo Episode {ep+1}: Reward = {ep_reward:.2f}")
        ep += 1

def manual(track="circular"):
    import pygame
    env = TrackEnv(render_mode="human", track_type=track)
    obs, _ = env.reset()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        keys = pygame.key.get_pressed()
        throttle = 0.0
        steering = 0.0
        if keys[pygame.K_UP]:
            throttle = 1.0
        if keys[pygame.K_DOWN]:
            throttle = -1.0
        if keys[pygame.K_LEFT]:
            steering = -1.0
        if keys[pygame.K_RIGHT]:
            steering = 1.0
        action = np.array([throttle, steering], dtype=np.float32)
        obs, reward, done, truncated, _ = env.step(action)
        env.render()
        if done or truncated:
            obs, _ = env.reset()
    env.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "demo", "manual"], default="train")
    parser.add_argument("--steps", type=int, default=50000)
    parser.add_argument("--track", choices=["circular", "oval", "figure8"], default="circular")
    args = parser.parse_args()
    if args.mode == "train":
        train_visual(timesteps=args.steps, track=args.track)
    elif args.mode == "manual":
        manual(track=args.track)
    else:
        demo()

if __name__ == "__main__":
    main()
