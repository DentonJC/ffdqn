import pandas as pd
import random
from scipy import stats

import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_env(env_name):
    env = gym.make(env_name)
    return env


def handle_reset(env):
    out = env.reset()
    if isinstance(out, tuple) and len(out) == 2:
        obs, _ = out
    else:
        obs = out
    return obs


def handle_step(env, action):
    out = env.step(action)
    if len(out) == 4:
        next_obs, reward, done, info = out
    else:
        next_obs, reward, terminated, truncated, info = out
        done = terminated or truncated
    return next_obs, reward, done, info


def compute_returns_episode(rewards, gamma=0.99):
    R = 0.0
    out = []
    for r in reversed(rewards):
        R = r + gamma * R
        out.insert(0, R)
    return out


def smooth_curve(x, window=5):
    x = np.asarray(x, dtype=np.float32)

    if window <= 1 or x.size == 0:
        return x
    if window > x.size:
        window = x.size

    kernel = np.ones(window, dtype=np.float32) / float(window)

    # Pad with edge values so the ends are not dragged toward zero
    left_pad = window // 2
    right_pad = window - 1 - left_pad
    x_padded = np.pad(x, (left_pad, right_pad), mode="edge")
    return np.convolve(x_padded, kernel, mode="valid")



def get_epsilon(total_steps, args):
    """Linear decay from eps_start to eps_end over eps_decay_steps."""
    if total_steps >= args.eps_decay_steps:
        return args.eps_end
    frac = total_steps / float(args.eps_decay_steps)
    return args.eps_start + frac * (args.eps_end - args.eps_start)


def analyze_results(training_results, eval_results, eval_episodes_per_trial):
    """
    training_results: dict { method_name: [epoch_avg_return, ...] }
    eval_results:     dict { method_name: [r_1, ..., r_(trials * eval_episodes_per_trial)] }
    eval_episodes_per_trial: int, number of eval episodes per trial

    For statistics:
      - Descriptive eval stats are computed over all episodes.
      - Mann–Whitney U tests are run on per-trial mean returns,
        not on stacked per-episode rewards.
    """

    def chunk_means(x, chunk_size):
        """
        x: 1D list/array of length (n_trials * chunk_size)
        returns: list of length n_trials with per-trial means
        """
        x = np.asarray(x, dtype=np.float32)
        if len(x) == 0:
            return []
        assert (
            len(x) % chunk_size == 0
        ), f"Length {len(x)} not divisible by chunk_size {chunk_size}"
        n_trials = len(x) // chunk_size
        return [
            float(x[i * chunk_size : (i + 1) * chunk_size].mean())
            for i in range(n_trials)
        ]

    print("\n" + "=" * 60)
    print("STATISTICAL ANALYSIS")
    print("=" * 60)

    # --------------------------------------------------------
    # Training performance (last 10 epochs)
    # --------------------------------------------------------
    print("\nTraining Performance (Last 10 Epochs):")
    for method, returns in training_results.items():
        if len(returns) >= 10:
            tail = returns[-10:]
        else:
            tail = returns
        mean_r = np.mean(tail) if tail else 0.0
        std_r = np.std(tail) if tail else 0.0
        print("method:", method)
        print("mean_return:", float(mean_r))
        print("std_return:", float(std_r))

    # --------------------------------------------------------
    # Evaluation performance (per-episode)
    # --------------------------------------------------------
    print("\nEvaluation Performance (per-episode distribution):")
    for method, rewards in eval_results.items():
        rewards_arr = np.asarray(rewards, dtype=np.float32)
        mean_r = float(rewards_arr.mean()) if rewards_arr.size > 0 else 0.0
        std_r = float(rewards_arr.std()) if rewards_arr.size > 0 else 0.0
        print(f"  {method:10s}: {mean_r:6.2f} ± {std_r:5.2f}  (N={len(rewards_arr)})")

    # --------------------------------------------------------
    # Per-trial aggregation for statistical tests
    # --------------------------------------------------------
    print("\nPer-trial mean evaluation returns:")
    per_trial_means = {}
    for method, rewards in eval_results.items():
        means = chunk_means(rewards, eval_episodes_per_trial)
        per_trial_means[method] = means
        if len(means) > 0:
            m = float(np.mean(means))
            s = float(np.std(means))
            print(str(method) + "_mean_fin:", m)
            print(str(method) + "_std_fin:", s)
        else:
            print(f"  {method:10s}: no data")

    # --------------------------------------------------------
    # Statistical significance on per-trial means
    # --------------------------------------------------------
    print("\nStatistical Significance on per-trial means (Mann-Whitney U, two-sided):")
    methods = list(per_trial_means.keys())
    for i in range(len(methods)):
        for j in range(i + 1, len(methods)):
            m1, m2 = methods[i], methods[j]
            x = per_trial_means[m1]
            y = per_trial_means[m2]
            if len(x) > 1 and len(y) > 1:
                stat, p = stats.mannwhitneyu(x, y, alternative="two-sided")
                if p < 0.001:
                    sig = "***"
                elif p < 0.01:
                    sig = "**"
                elif p < 0.05:
                    sig = "*"
                else:
                    sig = ""
                print(
                    f"  {m1:10s} vs {m2:10s}: p = {p:.4f} {sig}  "
                    f"(n={len(x)} vs {len(y)})"
                )
            else:
                print(
                    f"  {m1:10s} vs {m2:10s}: insufficient trials for test "
                    f"(n={len(x)} vs {len(y)})"
                )


def plot_results(args, training_results):
    all_data = []
    for method, returns in training_results.items():
        for epoch, value in enumerate(returns, start=1):
            all_data.append({"Method": method, "Epoch": epoch, "Return": value})
    df = pd.DataFrame(all_data)
    df.to_csv("training_results.csv", index=False)

    # --------------------------------------------------------
    # Plot smoothed learning curves: mean ± std across trials
    # --------------------------------------------------------
    plt.figure(figsize=(8, 5))
    fig = plt.gcf()
    fig.patch.set_facecolor("#F2E9E4")  # background outside of plot

    ax = plt.gca()
    ax.set_facecolor("#F2E9E4")  # background inside the plot

    color_map = {"ff": "#CF9E86", "backprop": "#a0522d"}

    default_color = "#a0522d"

    last_epoch = 1

    for method, returns in training_results.items():
        returns_arr = np.asarray(returns, dtype=np.float32)

        if (
            args.trials > 0
            and args.epochs > 0
            and returns_arr.size == args.trials * args.epochs
        ):
            returns_arr = returns_arr.reshape(args.trials, args.epochs)
            mean_curve = returns_arr.mean(axis=0)
            std_curve = returns_arr.std(axis=0)
            epochs_axis = np.arange(1, args.epochs + 1)
        else:
            mean_curve = returns_arr
            std_curve = np.zeros_like(mean_curve)
            epochs_axis = np.arange(1, mean_curve.size + 1)

        last_epoch = max(last_epoch, epochs_axis[-1])

        mean_smooth = smooth_curve(mean_curve, window=5)
        upper_smooth = smooth_curve(mean_curve + std_curve, window=5)
        lower_smooth = smooth_curve(mean_curve - std_curve, window=5)

        color = color_map.get(method, default_color)
        label = (
            "Backprop" if method == "backprop" else "FF" if method == "ff" else method
        )

        plt.plot(epochs_axis, mean_smooth, label=label, color=color, linewidth=2)
        plt.fill_between(
            epochs_axis, lower_smooth, upper_smooth, alpha=0.2, color=color
        )

    text_color = "#a0522d"
    font_scale = 1.2

    plt.xlabel("Epoch", color=text_color, fontsize=12 * font_scale)
    plt.ylabel("Average return", color=text_color, fontsize=12 * font_scale)
    plt.xticks(color=text_color, fontsize=10 * font_scale)
    plt.yticks(color=text_color, fontsize=10 * font_scale)

    ax.set_xlim(1, last_epoch)

    for spine in ax.spines.values():
        spine.set_color(text_color)
    ax.tick_params(colors=text_color)

    legend_elements = []
    for method in training_results.keys():
        color = color_map.get(method, default_color)
        label = (
            "Backprop" if method == "backprop" else "FF" if method == "ff" else method
        )
        line = Line2D([0], [0], color=color, lw=2, label=label)
        legend_elements.append(line)

    legend = plt.legend(handles=legend_elements, fontsize=10 * font_scale, frameon=True)
    legend.get_frame().set_facecolor("#F2E9E4")  # legend background color
    legend.get_frame().set_edgecolor("#a0522d")  # legend frame color
    for text in legend.get_texts():
        text.set_color(text_color)

    plt.grid(True, color=text_color, alpha=0.3)
    plt.tight_layout()
    plt.savefig("1.pdf")



