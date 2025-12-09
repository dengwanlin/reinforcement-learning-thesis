# hyperparameter_interaction_effects/config.py
from pathlib import Path

# ====== 路径配置 ======
# 你的实验日志根目录：下面这个你需要按自己实际情况稍微改一下
# 比如你的 Hopper-v4/a2c/... 如果就是在 experiment_logs 下，就像这样：
LOG_ROOT = Path(
    "/homes/sohawan2/reinforcement-learning-thesis/thesis_project"
)

# 本模块自己的输出目录（会自动创建）
THIS_DIR = Path(__file__).resolve().parent
ANALYSIS_OUT_DIR = THIS_DIR / "results"
ANALYSIS_OUT_DIR.mkdir(parents=True, exist_ok=True)

# 使用的性能指标列名（在汇总表里）
METRIC_COL = "max_eval_return"

# 要分析的超参数对（列名要和汇总表里的字段一致）
HP_PAIRS = {
    "a2c": [
        ("learning_rate", "n_steps"),
        ("learning_rate", "gae_lambda"),
        ("learning_rate", "ent_coef"),
    ],
    "ppo": [
        ("learning_rate", "n_steps"),
        ("learning_rate", "gae_lambda"),
        ("learning_rate", "clip_range"),
        ("ent_coef", "batch_size"),
    ],
}

# 要分析的环境（根据你实际做实验的 env 改）
ENVS = ["CartPole-v1", "LunarLander-v3",
        "LunarLanderContinuous-v3", "Hopper-v4"]
