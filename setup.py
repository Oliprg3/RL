from setuptools import setup, find_packages

setup(
    name="rl-racer",
    version="0.1.0",
    description="Self-driving car with reinforcement learning",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.24.0",
        "pygame>=2.5.0",
        "gymnasium>=0.29.0",
        "torch>=2.0.0",
        "stable-baselines3>=2.3.0",
    ],
    entry_points={
        "console_scripts": [
            "rl-racer-train=src.train:main",
            "rl-racer-demo=src.demo:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
)
