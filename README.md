# Forward-Forward Optimization of Deep Q-Network

This repository implements a comparative study of three reinforcement learning methods for training Deep Q-Network (DQN) agents:
- **Standard Backpropagation**
- **Hebbian Learning**: unsupervised representation learning + Q-head training.
- **Forward-Forward (FF) Learning**: goodness-based unsupervised layers + Q-head learning.

The code trains and evaluates agents on OpenAI Gym environments, such as `CartPole-v1`, and performs statistical analysis and plotting.

## Requirements

Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run experiments:
```bash
python main.py --epochs 70 --trials 4 --lr 0.001 --lr_rep 0.0001 --batch_size 128 --eps_start 0.1 --min_buffer 1000 --wd_rep 0 --threshold 2
```

Options include:
- `--method {backprop, hebb, ff, all}` – choose training algorithm  
- `--epochs N` – number of training epochs (default: 50)  
- `--trials N` – independent runs (default: 1)  
- `--eval_episodes N` – evaluation episodes (default: 50)

## Outputs

- `training_results.csv` – training curve data  
- `training_results.pdf` – plot of smoothed average returns across epochs  
