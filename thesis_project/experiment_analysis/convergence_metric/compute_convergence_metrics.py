import numpy as np
import pandas as pd

def smooth_curve(x, window=5):
    """Simple moving average smoothing."""
    return pd.Series(x).rolling(window=window, min_periods=1).mean().values


def compute_convergence_stats(
    timesteps: np.ndarray,
    rewards: np.ndarray,
    smooth_window: int = 5,
    final_fraction: float = 0.1,
    final_last_k: int = 10,
):
    """
    Compute convergence & post-peak metrics for a single evaluation curve.
    Corresponds directly to Section 4.3.3 in the thesis.
    """

    # ---- 保证是一维 ----
    timesteps = np.atleast_1d(timesteps).astype(float)
    rewards = np.atleast_1d(rewards).astype(float)

    n = len(rewards)
    # 如果 evaluation 次数太少，没法算这些指标，直接 NaN
    if n < 3:
        return dict(
            R_max=np.nan,
            t_peak=np.nan,
            R_end=np.nan,
            delta_post=np.nan,
            s_final=np.nan,
            sigma_final=np.nan,
            n_eval=n,
        )

    # ------------------------
    # 1. smoothing
    # ------------------------
    smooth = smooth_curve(rewards, window=smooth_window)

    # ------------------------
    # 2. find peak
    # ------------------------
    idx_peak = int(np.argmax(smooth))
    R_max = float(smooth[idx_peak])
    t_peak = float(timesteps[idx_peak])

    # ------------------------
    # 3. final performance (last K evals)
    # ------------------------
    K = min(final_last_k, n)
    R_end = float(np.mean(smooth[-K:]))

    # post-peak drop
    delta_post = R_end - R_max

    # ------------------------
    # 4. final-phase slope (last 10% of data)
    # ------------------------
    cut = int(max(2, n * final_fraction))
    final_segment = smooth[-cut:]
    diffs = np.diff(final_segment)
    s_final = float(np.mean(diffs))

    # ------------------------
    # 5. final-phase instability
    # ------------------------
    sigma_final = float(np.std(final_segment))

    return dict(
        R_max=R_max,
        t_peak=t_peak,
        R_end=R_end,
        delta_post=delta_post,
        s_final=s_final,
        sigma_final=sigma_final,
        n_eval=n,
    )
