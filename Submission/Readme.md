# Shallow Q-Network for Tic-Tac-Toe

## Overview

This project implements a Shallow Q-Network (SQN) for the game Tic-Tac-Toe. The SQN is a simplified reinforcement learning agent designed to learn optimal strategies by interacting with the game environment. This implementation focuses on exploring reinforcement learning concepts such as epsilon-greedy exploration, experience replay, and Q-value updates using the Bellman equation.

The project investigates multiple iterations of the SQN, refining exploration strategies, reward systems, and neural network architectures to achieve better performance against opponents of varying intelligence levels.

---

## Features

- **Reinforcement Learning**: Uses SQN to approximate Q-values for optimal decision-making.
- **Epsilon-Greedy Exploration**: Balances exploration and exploitation dynamically.
- **Experience Replay**: Stores past game states for efficient training.
- **Dynamic Reward Scaling**: Adapts rewards and penalties based on opponent difficulty.
- **Opponent Smartness**: Progressive training against opponents with varying strategies.
- **Performance Metrics**: Tracks wins, losses, and draws across training and evaluation.

---

## File Structure

The project files are organized under the `submission/` folder:

- `submission/2021A7PS2627G.py`: Main Python script implementing the SQN and training pipeline.
- `submission/2021A7SP2627G.pdf`: Documentation describing the models, methodology, and results.
- `submission/TicTacToe.py`: Game logic for Tic-Tac-Toe.
- `submission/2021A7PS2627G_MODEL.h5`: Pretrained model for the SQN agent.

---

## Requirements

- Python 3.7+
- TensorFlow 2.0+
- NumPy

Install dependencies using:
```bash
pip install tensorflow numpy
```

---

## Usage

### Training
To train the SQN agent:
```bash
python submission/2021A7PS2627G.py train
```

### Evaluation
To evaluate the agent's performance against an opponent:
```bash
python submission/2021A7PS2627G.py <smartMovePlayer1Probability>
```
- `smartMovePlayer1Probability`: A float between 0 and 1 indicating the opponent's intelligence level.

---

## Models and Results

### Key Models

- **Model 1**: Baseline implementation with static epsilon decay.
- **Model 6**: Best-performing model with dynamic epsilon decay and variable draw penalties.

### Model 6 Results

| Opponent Smartness | Wins | Losses | Draws |
|--------------------|------|--------|-------|
| 0.0               | 719  | 124    | 157   |
| 0.5               | 385  | 227    | 388   |
| 1.0               | 78   | 314    | 608   |

---

## Methodology

1. **Neural Network Architecture**:
    - Input: 9 nodes (board state).
    - Hidden Layers: Two layers, each with 64 neurons and ReLU activation.
    - Output: 9 nodes (Q-values for each action).
2. **Reward System**:
    - Win: +1
    - Loss: -1
    - Draw: Dynamic penalty based on opponent intelligence.
3. **Training Process**:
    - Experience replay for stability.
    - Gradual epsilon decay.
    - Progressive opponent difficulty.

---

## Observations

- Simpler architectures (e.g., Model 6) outperform more complex models in this environment.
- Dynamic exploration strategies significantly improve adaptability.
- Balanced reward schemes enhance overall performance.

---

## Future Work

- Implement Monte Carlo Tree Search (MCTS) for decision-making.
- Explore self-play for training without predefined opponent strategies.
- Introduce curriculum learning for progressive skill improvement.
- Optimize computational efficiency and reward shaping.

---

## Acknowledgments

This project was developed as part of coursework for **2021A7PS2627G - Kislay Ranjan Nee Tandon**.
