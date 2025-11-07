import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from utils import seed_everything, make_env, handle_reset, handle_step, analyze_results, plot_results, parse_arguments


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def __len__(self):
        return len(self.buffer)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.stack(states),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.stack(next_states),
            np.array(dones, dtype=np.float32),
        )


# ------------------------------------------------------------
# Action overlay (for FF)
# ------------------------------------------------------------


def overlay_action_on_state(state, action, n_actions, overlay_type="standard"):
    """
    state: 1D [state_dim] or 2D [B, state_dim] tensor
    action: int or 1D tensor of ints of length B
    """
    if state.dim() == 1:
        s = state.clone()
        a_one = torch.zeros(n_actions, device=s.device)
        if overlay_type == "standard":
            a_one[action] = 1.0
        else:
            a_one[action] = s.abs().max()
        return torch.cat([s, a_one])

    elif state.dim() == 2:
        B, state_dim = state.shape
        a_one = torch.zeros(B, n_actions, device=state.device)
        if isinstance(action, int):
            if overlay_type == "standard":
                a_one[:, action] = 1.0
            else:
                absmax = state.abs().max(dim=1)[0]
                a_one[:, action] = absmax
        else:
            if overlay_type == "standard":
                a_one[torch.arange(B), action] = 1.0
            else:
                absmax = state.abs().max(dim=1)[0]
                a_one[torch.arange(B), action] = absmax
        return torch.cat([state, a_one], dim=1)
    else:
        raise ValueError("overlay_action_on_state: state must be 1D or 2D")


# ============================================================
# Standard DQN (Backprop) network
# ============================================================


class DQNNet(nn.Module):
    def __init__(self, state_dim, n_actions, hidden_dims=[128, 64], device=None):
        super().__init__()
        self.device = device or torch.device("cpu")
        layers = []
        dims = [state_dim] + hidden_dims
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
        self.feature = nn.Sequential(*layers)
        self.q_head = nn.Linear(dims[-1], n_actions)
        self.to(self.device)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = x.to(self.device)
        h = self.feature(x)
        return self.q_head(h)

    def select_action(self, state, epsilon=0.1):
        # state: 1D torch tensor on any device
        if random.random() < epsilon:
            return random.randint(0, self.q_head.out_features - 1)
        with torch.no_grad():
            q = self.forward(state.unsqueeze(0))[0]
            return q.argmax().item()


# ============================================================
# Hebbian representation + DQN head
# ============================================================


class HebbLayer(nn.Linear):
    def __init__(
        self, in_features, out_features, lr=0.03, weight_decay=0.0, device=None
    ):
        super().__init__(in_features, out_features, bias=True, device=device)
        self.lr = lr
        self.weight_decay = weight_decay
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)
        self.weight.requires_grad = False
        if self.bias is not None:
            self.bias.requires_grad = False

    def forward(self, x):
        return torch.relu(
            torch.matmul(x, self.weight.t())
            + (self.bias.unsqueeze(0) if self.bias is not None else 0)
        )

    def train_step(self, x):
        # x: [B, in_features]
        x_norm = x / (x.norm(p=2, dim=1, keepdim=True) + 1e-8)
        y = self.forward(x_norm)
        B = x_norm.size(0)
        dw = torch.einsum("bi,bj->ij", y, x_norm) / B
        self.weight.data += self.lr * dw
        if self.weight_decay > 0:
            self.weight.data *= 1.0 - self.weight_decay
        if self.bias is not None:
            db = self.lr * y.mean(dim=0)
            self.bias.data += db
        return y


class HebbQNet(nn.Module):
    def __init__(
        self,
        state_dim,
        n_actions,
        hidden_dims=[128, 64],
        lr_rep=0.03,
        wd_rep=0.0,
        device=None,
        lr_head=1e-3,
        wd_head=0.0,
    ):
        super().__init__()
        self.device = device or torch.device("cpu")
        self.n_actions = n_actions
        dims = [state_dim] + hidden_dims
        self.repr_layers = nn.ModuleList()
        for i in range(len(dims) - 1):
            self.repr_layers.append(
                HebbLayer(
                    dims[i],
                    dims[i + 1],
                    lr=lr_rep,
                    weight_decay=wd_rep,
                    device=self.device,
                )
            )

        self.q_head = nn.Linear(dims[-1], n_actions).to(self.device)
        nn.init.xavier_uniform_(self.q_head.weight)
        nn.init.zeros_(self.q_head.bias)
        self.q_opt = AdamW(self.q_head.parameters(), lr=lr_head, weight_decay=wd_head)
        self.to(self.device)

    def forward(self, x):
        x = x.to(self.device)
        h = x
        for layer in self.repr_layers:
            h = layer(h)
        return self.q_head(h)

    def train_representation(self, states):
        # states: [B, state_dim] tensor on any device
        x = states.to(self.device)
        with torch.no_grad():
            h = x
            for layer in self.repr_layers:
                h = layer.train_step(h)
        return h

    def select_action(self, state, epsilon=0.1):
        if random.random() < epsilon:
            return random.randint(0, self.n_actions - 1)
        with torch.no_grad():
            q = self.forward(state.unsqueeze(0))[0]
            return q.argmax().item()


# ============================================================
# Forward-Forward representation + DQN head
# ============================================================


class FFLayer(nn.Linear):
    def __init__(
        self,
        in_features,
        out_features,
        threshold=2.0,
        lr=0.03,
        weight_decay=0.0,
        device=None,
    ):
        super().__init__(in_features, out_features, bias=True, device=device)
        self.relu = nn.ReLU()
        self.opt = AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
        self.threshold = threshold
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)
        self.pos_goodness_history = []
        self.neg_goodness_history = []

    def forward(self, x):
        x_norm = x / (x.norm(p=2, dim=1, keepdim=True) + 1e-8)
        return self.relu(torch.matmul(x_norm, self.weight.t()) + self.bias.unsqueeze(0))

    def goodness(self, x):
        h = self.forward(x)
        return h.pow(2).mean(dim=1)

    def train_step(self, x_pos, x_neg):
        self.train()
        g_pos = self.goodness(x_pos)
        g_neg = self.goodness(x_neg)
        self.pos_goodness_history.append(g_pos.mean().item())
        self.neg_goodness_history.append(g_neg.mean().item())
        pos_loss = torch.relu(self.threshold - g_pos).pow(2).mean()
        neg_loss = torch.relu(g_neg - self.threshold).pow(2).mean()
        loss = pos_loss + neg_loss
        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
        self.opt.step()
        return loss.item()


class FFQNet(nn.Module):
    def __init__(
        self,
        state_dim,
        n_actions,
        hidden_dims=[128, 64],
        threshold=2.0,
        lr_ff=0.03,
        wd_ff=0.0,
        lr_head=1e-3,
        wd_head=0.0,
        overlay_type="standard",
        device=None,
    ):
        super().__init__()
        self.device = device or torch.device("cpu")
        self.n_actions = n_actions
        self.overlay_type = overlay_type

        dims = [state_dim + n_actions] + hidden_dims
        self.layers = nn.ModuleList()
        for i in range(len(dims) - 1):
            self.layers.append(
                FFLayer(
                    dims[i],
                    dims[i + 1],
                    threshold=threshold,
                    lr=lr_ff,
                    weight_decay=wd_ff,
                    device=self.device,
                )
            )

        # Q-head: scalar Q-value for (s,a) pair
        self.q_head = nn.Linear(hidden_dims[-1], 1).to(self.device)
        nn.init.xavier_uniform_(self.q_head.weight)
        nn.init.zeros_(self.q_head.bias)
        self.q_opt = AdamW(self.q_head.parameters(), lr=lr_head, weight_decay=wd_head)
        self.to(self.device)

    # ---- FF representation helpers ----
    def _forward_layers(self, x):
        h = x
        for layer in self.layers:
            layer.eval()
            h = layer.forward(h)
        return h

    def q_values(self, states):
        """
        states: [B, state_dim] -> [B, n_actions]
        """
        states = states.to(self.device)
        B = states.size(0)
        all_q = []
        for a in range(self.n_actions):
            x_in = overlay_action_on_state(states, a, self.n_actions, self.overlay_type)
            h = self._forward_layers(x_in)
            q_a = self.q_head(h).view(B)
            all_q.append(q_a.unsqueeze(1))
        return torch.cat(all_q, dim=1)

    def get_q_sa(self, states, actions):
        """
        For FF labeling / TD-error computation. No gradient needed on FF layers or head.
        states: [B, state_dim], actions: [B]
        """
        states = states.to(self.device)
        actions = actions.to(self.device)
        with torch.no_grad():
            x_in = overlay_action_on_state(
                states, actions, self.n_actions, self.overlay_type
            )
            h = self._forward_layers(x_in)
            q = self.q_head(h).view(-1)
        return q

    def get_q_sa_for_head(self, states, actions):
        """
        For Q-head backprop; FF layers are frozen (no grad).
        """
        states = states.to(self.device)
        actions = actions.to(self.device)
        with torch.no_grad():
            x_in = overlay_action_on_state(
                states, actions, self.n_actions, self.overlay_type
            )
            h = self._forward_layers(x_in).detach()
        q = self.q_head(h).view(-1)
        return q

    def train_ff_layers(self, pos_inputs, neg_inputs):
        """
        pos_inputs, neg_inputs: [N_pos, d_in], [N_neg, d_in]
        """
        h_pos = pos_inputs
        h_neg = neg_inputs
        total_loss = 0.0
        for layer in self.layers:
            loss = layer.train_step(h_pos, h_neg)
            total_loss += loss
            with torch.no_grad():
                layer.eval()
                h_pos = layer.forward(h_pos).detach()
                h_neg = layer.forward(h_neg).detach()
        return total_loss

    def select_action(self, state, epsilon=0.1):
        if random.random() < epsilon:
            return random.randint(0, self.n_actions - 1)
        with torch.no_grad():
            q = self.q_values(state.unsqueeze(0))[0]
            return q.argmax().item()


# ============================================================
# DQN updates
# ============================================================


def dqn_update_backprop(
    policy_net, target_net, optimizer, replay_buffer, batch_size, gamma, device
):
    states_np, actions_np, rewards_np, next_states_np, dones_np = replay_buffer.sample(
        batch_size
    )
    states = torch.tensor(states_np, dtype=torch.float32, device=device)
    next_states = torch.tensor(next_states_np, dtype=torch.float32, device=device)
    actions = torch.tensor(actions_np, dtype=torch.long, device=device)
    rewards = torch.tensor(rewards_np, dtype=torch.float32, device=device)
    dones = torch.tensor(dones_np, dtype=torch.float32, device=device)

    q_values = policy_net(states)  # [B, n_actions]
    q_sa = q_values.gather(1, actions.unsqueeze(1)).view(-1)

    with torch.no_grad():
        next_q_values = target_net(next_states)  # [B, n_actions]
        max_next_q = next_q_values.max(dim=1)[0]
        targets = rewards + gamma * max_next_q * (1.0 - dones)

    loss = nn.MSELoss()(q_sa, targets)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.0)
    optimizer.step()
    return loss.item()


def dqn_update_hebb(policy_net, target_net, replay_buffer, batch_size, gamma, device):
    states_np, actions_np, rewards_np, next_states_np, dones_np = replay_buffer.sample(
        batch_size
    )
    states = torch.tensor(states_np, dtype=torch.float32, device=device)
    next_states = torch.tensor(next_states_np, dtype=torch.float32, device=device)
    actions = torch.tensor(actions_np, dtype=torch.long, device=device)
    rewards = torch.tensor(rewards_np, dtype=torch.float32, device=device)
    dones = torch.tensor(dones_np, dtype=torch.float32, device=device)

    # Hebbian representation update (unsupervised)
    policy_net.train_representation(states)

    # DQN on Q-head
    q_values = policy_net(states)  # [B, n_actions]
    q_sa = q_values.gather(1, actions.unsqueeze(1)).view(-1)

    with torch.no_grad():
        next_q_values = target_net(next_states)
        max_next_q = next_q_values.max(dim=1)[0]
        targets = rewards + gamma * max_next_q * (1.0 - dones)

    loss = nn.MSELoss()(q_sa, targets)
    policy_net.q_opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.q_head.parameters(), max_norm=1.0)
    policy_net.q_opt.step()
    return loss.item()


def dqn_update_ff(
    policy_net,
    target_net,
    replay_buffer,
    batch_size,
    gamma,
    device,
    n_actions,
    overlay_type,
):
    states_np, actions_np, rewards_np, next_states_np, dones_np = replay_buffer.sample(
        batch_size
    )
    states = torch.tensor(states_np, dtype=torch.float32, device=device)
    next_states = torch.tensor(next_states_np, dtype=torch.float32, device=device)
    actions = torch.tensor(actions_np, dtype=torch.long, device=device)
    rewards = torch.tensor(rewards_np, dtype=torch.float32, device=device)
    dones = torch.tensor(dones_np, dtype=torch.float32, device=device)

    # 1) Compute TD targets and TD errors using current policy + target
    with torch.no_grad():
        q_sa = policy_net.get_q_sa(states, actions)  # [B]
        next_q_values = target_net.q_values(next_states)  # [B, n_actions]
        max_next_q = next_q_values.max(dim=1)[0]
        targets = rewards + gamma * max_next_q * (1.0 - dones)
        td_errors = targets - q_sa  # [B]

    # 2) FF representation training: label by sign(td_error)
    overlay_batch = overlay_action_on_state(states, actions, n_actions, overlay_type)
    pos_mask = td_errors >= 0.0
    neg_mask = td_errors < 0.0
    if pos_mask.any() and neg_mask.any():
        pos_inputs = overlay_batch[pos_mask]
        neg_inputs = overlay_batch[neg_mask]
        policy_net.train_ff_layers(pos_inputs, neg_inputs)

    # 3) Q-head DQN training on frozen FF layers
    q_sa_head = policy_net.get_q_sa_for_head(states, actions)
    loss = nn.MSELoss()(q_sa_head, targets)

    policy_net.q_opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.q_head.parameters(), max_norm=1.0)
    policy_net.q_opt.step()
    return loss.item()


# ============================================================
# Evaluation
# ============================================================


def evaluate_policy(env, policy, device, n_episodes=50, max_steps=200):
    rewards_all = []
    for _ in range(n_episodes):
        state = handle_reset(env)
        ep_reward = 0.0
        for _ in range(max_steps):
            state_tensor = torch.tensor(state, dtype=torch.float32, device=device)
            action = policy.select_action(state_tensor, epsilon=0.0)
            next_state, reward, done, _ = handle_step(env, action)
            ep_reward += reward
            state = next_state
            if done:
                break
        rewards_all.append(ep_reward)
    return rewards_all


# ============================================================
# Training loops for each method
# ============================================================


def train_dqn_backprop(env, state_dim, n_actions, args):
    all_epoch_returns = []
    all_eval_rewards = []

    for trial in range(args.trials):
        seed_everything(args.seed + trial)
        device = args.device
        policy = DQNNet(state_dim, n_actions, hidden_dims=args.ff_dims, device=device)
        target = DQNNet(state_dim, n_actions, hidden_dims=args.ff_dims, device=device)
        target.load_state_dict(policy.state_dict())
        optimizer = AdamW(policy.parameters(), lr=args.lr, weight_decay=args.wd_rep)
        replay = ReplayBuffer(args.buffer_size)

        total_steps = 0
        for epoch in range(args.epochs):
            epoch_rewards = []
            for _ in range(args.episodes_per_epoch):
                state = handle_reset(env)
                ep_reward = 0.0
                for _ in range(args.max_steps):
                    state_tensor = torch.tensor(
                        state, dtype=torch.float32, device=device
                    )
                    action = policy.select_action(state_tensor, epsilon=args.epsilon)
                    next_state, reward, done, _ = handle_step(env, action)
                    replay.push(state, action, reward, next_state, done)
                    state = next_state
                    ep_reward += reward
                    total_steps += 1

                    if (
                        len(replay) >= args.min_buffer
                        and total_steps % args.train_freq == 0
                    ):
                        dqn_update_backprop(
                            policy,
                            target,
                            optimizer,
                            replay,
                            args.batch_size,
                            args.gamma,
                            device,
                        )

                    if total_steps % args.target_update == 0:
                        target.load_state_dict(policy.state_dict())

                    if done:
                        break
                epoch_rewards.append(ep_reward)

            avg_ret = np.mean(epoch_rewards) if epoch_rewards else 0.0
            all_epoch_returns.append(avg_ret)
            print(
                f"[Backprop] Trial {trial+1}/{args.trials}, "
                f"Epoch {epoch+1}/{args.epochs}, AvgReturn={avg_ret:.2f}, "
                f"Buffer={len(replay)}"
            )

        eval_rewards = evaluate_policy(
            env, policy, device, n_episodes=args.eval_episodes, max_steps=args.max_steps
        )
        all_eval_rewards.extend(eval_rewards)
        print("bp_mean:", float(np.mean(eval_rewards)))
        print("bp_std:", float(np.std(eval_rewards)))

    return policy, all_epoch_returns, all_eval_rewards


def train_dqn_hebb(env, state_dim, n_actions, args):
    all_epoch_returns = []
    all_eval_rewards = []

    for trial in range(args.trials):
        seed_everything(args.seed + trial)
        device = args.device
        policy = HebbQNet(
            state_dim,
            n_actions,
            hidden_dims=args.hebb_dims,
            lr_rep=args.lr_rep,
            wd_rep=args.wd_rep,
            lr_head=args.lr,
            wd_head=args.wd_head,
            device=device,
        )
        target = HebbQNet(
            state_dim,
            n_actions,
            hidden_dims=args.hebb_dims,
            lr_rep=args.lr_rep,
            wd_rep=args.wd_rep,
            lr_head=args.lr,
            wd_head=args.wd_head,
            device=device,
        )
        target.load_state_dict(policy.state_dict())
        replay = ReplayBuffer(args.buffer_size)

        total_steps = 0
        for epoch in range(args.epochs):
            epoch_rewards = []
            for _ in range(args.episodes_per_epoch):
                state = handle_reset(env)
                ep_reward = 0.0
                for _ in range(args.max_steps):
                    state_tensor = torch.tensor(
                        state, dtype=torch.float32, device=device
                    )
                    action = policy.select_action(state_tensor, epsilon=args.epsilon)
                    next_state, reward, done, _ = handle_step(env, action)
                    replay.push(state, action, reward, next_state, done)
                    state = next_state
                    ep_reward += reward
                    total_steps += 1

                    if (
                        len(replay) >= args.min_buffer
                        and total_steps % args.train_freq == 0
                    ):
                        dqn_update_hebb(
                            policy, target, replay, args.batch_size, args.gamma, device
                        )

                    if total_steps % args.target_update == 0:
                        target.load_state_dict(policy.state_dict())

                    if done:
                        break
                epoch_rewards.append(ep_reward)

            avg_ret = np.mean(epoch_rewards) if epoch_rewards else 0.0
            all_epoch_returns.append(avg_ret)
            print(
                f"[Hebb]     Trial {trial+1}/{args.trials}, "
                f"Epoch {epoch+1}/{args.epochs}, AvgReturn: {avg_ret:.2f}, "
                f"Buffer: {len(replay)}"
            )

        eval_rewards = evaluate_policy(
            env, policy, device, n_episodes=args.eval_episodes, max_steps=args.max_steps
        )
        all_eval_rewards.extend(eval_rewards)
        print("trial:", trial + 1)
        print("hebb_mean:", float(np.mean(eval_rewards)))
        print("hebb_std:", float(np.std(eval_rewards)))

    return policy, all_epoch_returns, all_eval_rewards


def train_dqn_ff(env, state_dim, n_actions, args):
    all_epoch_returns = []
    all_eval_rewards = []

    for trial in range(args.trials):
        seed_everything(args.seed + trial)
        device = args.device
        policy = FFQNet(
            state_dim,
            n_actions,
            hidden_dims=args.ff_dims,
            threshold=args.threshold,
            lr_ff=args.lr_rep,
            wd_ff=args.wd_rep,
            lr_head=args.lr,
            wd_head=args.wd_head,
            overlay_type=args.overlay_type,
            device=device,
        )
        target = FFQNet(
            state_dim,
            n_actions,
            hidden_dims=args.ff_dims,
            threshold=args.threshold,
            lr_ff=args.lr_rep,
            wd_ff=args.wd_rep,
            lr_head=args.lr,
            wd_head=args.wd_head,
            overlay_type=args.overlay_type,
            device=device,
        )
        target.load_state_dict(policy.state_dict())
        replay = ReplayBuffer(args.buffer_size)

        total_steps = 0
        for epoch in range(args.epochs):
            epoch_rewards = []
            for _ in range(args.episodes_per_epoch):
                state = handle_reset(env)
                ep_reward = 0.0
                for _ in range(args.max_steps):
                    state_tensor = torch.tensor(
                        state, dtype=torch.float32, device=device
                    )
                    action = policy.select_action(state_tensor, epsilon=args.epsilon)
                    next_state, reward, done, _ = handle_step(env, action)
                    replay.push(state, action, reward, next_state, done)
                    state = next_state
                    ep_reward += reward
                    total_steps += 1

                    if (
                        len(replay) >= args.min_buffer
                        and total_steps % args.train_freq == 0
                    ):
                        dqn_update_ff(
                            policy,
                            target,
                            replay,
                            args.batch_size,
                            args.gamma,
                            device,
                            n_actions,
                            args.overlay_type,
                        )

                    if total_steps % args.target_update == 0:
                        target.load_state_dict(policy.state_dict())

                    if done:
                        break
                epoch_rewards.append(ep_reward)

            avg_ret = np.mean(epoch_rewards) if epoch_rewards else 0.0
            all_epoch_returns.append(avg_ret)
            print(f"[FF]       Trial {trial+1}/{args.trials}")
            print("epoch:", epoch + 1)
            print(f"avg_return: {avg_ret:.2f}")
            print(f"buffer_size: {len(replay)}")

        eval_rewards = evaluate_policy(
            env, policy, device, n_episodes=args.eval_episodes, max_steps=args.max_steps
        )
        all_eval_rewards.extend(eval_rewards)
        print("ff_mean:", float(np.mean(eval_rewards)))
        print("ff_std:", float(np.std(eval_rewards)))

    return policy, all_epoch_returns, all_eval_rewards


def main():
    args = parse_arguments()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    env = make_env(args.env)
    state_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    training_results = {}
    eval_results = {}

    methods_to_run = ["backprop", "ff"] if args.method == "all" else [args.method]

    for method in methods_to_run:
        print("\n" + "=" * 60)
        print(f"Running method: {method.upper()}")
        print("=" * 60)

        if method == "backprop":
            policy, train_curve, eval_rewards = train_dqn_backprop(
                env, state_dim, n_actions, args
            )
        elif method == "hebb":
            policy, train_curve, eval_rewards = train_dqn_hebb(
                env, state_dim, n_actions, args
            )
        else:
            policy, train_curve, eval_rewards = train_dqn_ff(
                env, state_dim, n_actions, args
            )

        training_results[method] = train_curve
        eval_results[method] = eval_rewards

    analyze_results(training_results, eval_results, args.eval_episodes)
    plot_results(args, training_results)


if __name__ == "__main__":
    main()
