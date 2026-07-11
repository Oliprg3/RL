# RL
RL based Game play

**A visual reinforcement learning agent that learns to drive a car around a track in minutes.**  
Watch the car go from crashing randomly to completing fast, smooth laps all in real time.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Stable-Baselines3](https://img.shields.io/badge/Stable--Baselines3-2.3.0-orange)](https://github.com/DLR-RM/stable-baselines3)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)](https://pytorch.org/)
[![Demo](assets/demo.gif)](assets/demo.gif)

Features

- **Live training GUI** –-- see the car learn in real time
- **Multiple track shapes** –-- circular, oval, figure‑8
- **Realistic car physics** –-- acceleration, drag, steering, slip angle
- **Ray‑based perception** –-- 15 LIDAR‑like sensors (no pixels)
- **State‑of‑the‑art algorithm** –-- Soft Actor‑Critic (SAC)
- **Fast training** –-- completes in 2‑3 minutes on a CPU
- **Headless mode** –-- train without GUI for speed
- **Manual control** –-- drive the car yourself with arrow keys

How It Works
The Environment
Track: White road on a black background (circular, oval, or figure‑8).

Car: Bicycle model with speed, acceleration, drag, and steering.

Sensors: 15 rays shooting from -80° to +80° – return distances to track edges.

Actions: Continuous throttle [-1,1] and steering [-1,1].

The Algorithm (SAC)
Off‑policy – learns from a replay buffer (sample‑efficient).

Continuous actions – perfect for driving.

Entropy regularisation – balances exploration vs exploitation automatically.

Two critics – reduces overestimation (Clipped Double‑Q).

Reward Function
Signal	Value
Staying on track	+10 per step
Leaving track	–200 per step
Forward progress	+0.5 per unit
Speed	+0.1 per unit
Centre distance	–0.02 per unit
Heading alignment	–0.02 per degree error
Lap completion	+400

After 50,000 training steps (~2‑3 minutes):

Success rate: >90% lap completion

Average speed: 20–25 units

Lap time: ~200 steps

Zero off‑track incidents (consistent)
  Quick Start

### 1, Install dependencies
```bash
pip install -r requirements.txt
Train with GUI:
    python src/train.py --mode train --steps 50000
Run the trained agent:
    python src/demo.py
Drive manually:
    python src/train.py --mode manual



