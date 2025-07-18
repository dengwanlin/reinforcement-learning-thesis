# main.py
import gymnasium as gym

# 创建环境实例（自动从注册表加载）
env = gym.make(
    "GridWorld-v0",
    render_mode="human",  # 开启可视化
    size=10  # 覆盖默认参数为10x10网格
)

# 运行测试
obs, info = env.reset()
print("初始观测:", obs)
print("初始距离:", info["distance"])

for step in range(20):
    action = env.action_space.sample()  # 随机策略
    obs, reward, terminated, truncated, info = env.step(action)

    print(f"\nStep {step + 1}")
    print("动作:", ["右", "上", "左", "下"][action])
    print("新位置:", obs["agent"])
    print("奖励:", reward)
    print("距离:", info["distance"])

    if terminated or truncated:
        print("环境终止，重置...")
        obs, info = env.reset()

env.close()