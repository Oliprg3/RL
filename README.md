# RL
RL based Game play

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Stable-Baselines3](https://img.shields.io/badge/Stable--Baselines3-2.3.0-orange)](https://github.com/DLR-RM/stable-baselines3)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)](https://pytorch.org/)
[![Demo](assets/demo.gif)](assets/demo.gif)


- **Live training GUI** –-- see the car learn in real time
- **Realistic car physics** –-- acceleration, drag, steering, slip angle
- **State‑of‑the‑art algorithm** –-- Soft Actor‑Critic (SAC)
- **Fast training** –-- completes in 2‑3 minutes on a CPU
- **Headless mode** –-- train without GUI for speed
- **Manual control** –-- drive the car yourself with arrow ke
Off‑policy – learns from a replay buffer (sample‑efficient).

### 1, Install dependencies
```bash
pip install -r requirements.txt
Train with GUI:
    python src/train.py --mode train --steps 50000
Run the trained agent:
    python src/demo.py
Drive manually:
    python src/train.py --mode manual



