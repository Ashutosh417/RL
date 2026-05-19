# Flappy Bird RL Agent using PPO

This repository contains a deep reinforcement learning agent trained to play Flappy Bird. The agent is built using **PyTorch** and implements the **Proximal Policy Optimization (PPO)** algorithm with an Actor-Critic architecture and Generalized Advantage Estimation (GAE). The environment is provided by the `flappy-bird-gymnasium` library.

---

## Features
* **Algorithm**: Proximal Policy Optimization (PPO) with clipped objective.
* **Architecture**: Separate Multilayer Perceptron (MLP) networks for the Actor and the Critic.
* **Advantage Estimation**: Generalized Advantage Estimation (GAE) for stable updates.
* **Environment**: `FlappyBird-v0` utilizing a 12-dimensional continuous state vector and a discrete action space (flap or do nothing).

---

## Dependencies
Ensure you have Python 3.x installed. You can install the required packages using pip:

```bash
pip install torch numpy matplotlib gymnasium flappy-bird-gymnasium
```

---

## Project Structure
* **`Flappy_Bird.ipynb`**: The main Jupyter Notebook containing the environment setup, the neural network definitions (`Actor` and `Critic`), the `RLAgent` class with the PPO training loop, and the evaluation code.
* **`Flappy_save.pth`**: (Generated after training) A PyTorch state dictionary containing the trained weights for the Actor, Critic, and Optimizer.

---

## Architecture details
* **Actor Network**: Maps the 12-dimensional state to a probability distribution over the 2 possible actions using `Tanh` activations and a final `Softmax` layer.
* **Critic Network**: Maps the 12-dimensional state to a scalar value estimating the expected return, using `Tanh` activations.
* **Hyperparameters**: 
  * Learning Rate: 3e-4 (Actor), 1e-3 (Critic)
  * Gamma (Discount factor): 0.99
  * GAE Lambda: 0.95
  * PPO Epsilon (Clip ratio): 0.2
  * Rollout steps: 2048

---

## Usage

### 1. Training the Agent
Run the cells in the notebook sequentially. The `Agent.train()` method will start the training process for 3000 episodes. It will print the average reward of the last 10 episodes and the last completed reward every 10 updates.

### 2. Plotting Progress
After training, running `Agent.plot()` generates a graph showing the 100-episode moving average of the rewards, helping visualize the agent's learning progression.

### 3. Saving and Loading
The notebook includes cells to save the trained model weights:
```python
torch.save({
    'actor': Agent.Actor.state_dict(),
    'critic': Agent.Critic.state_dict(),
    'optimizer': Agent.optim.state_dict(),
}, 'Flappy_save.pth')
```
You can load a pre-trained model by executing the checkpoint loading cell.

### 4. Watch the Agent Play
The final cell initializes the environment in `human` render mode. It loads the trained weights and runs the agent for 3 episodes, printing the total reward achieved in each run so you can visually verify the agent's performance.
