# grid_world_env.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame


class GridWorldEnv(gym.Env):
    """
    自定义2D网格世界环境
    - 观测空间: 包含代理位置和目标位置的字典
    - 动作空间: 4个方向 (右/上/左/下)
    - 奖励: 到达目标获得+1，其他情况0
    - 终止条件: 当代理到达目标位置
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, size=5, render_mode=None):
        super().__init__()
        self.size = size  # 网格尺寸 (size x size)
        self.window_size = 512  # 渲染窗口像素尺寸

        # 定义动作空间 [右, 上, 左, 下]
        self.action_space = spaces.Discrete(4)

        # 定义观测空间（代理位置 + 目标位置）
        self.observation_space = spaces.Dict({
            "agent": spaces.Box(0, size - 1, shape=(2,), dtype=int),
            "target": spaces.Box(0, size - 1, shape=(2,), dtype=int)
        })

        # 动作到方向映射
        self._action_to_direction = {
            0: np.array([1, 0]),  # 右
            1: np.array([0, 1]),  # 上
            2: np.array([-1, 0]),  # 左
            3: np.array([0, -1]),  # 下
        }

        # 渲染相关初始化
        self.render_mode = render_mode
        self.window = None
        self.clock = None

    def _get_obs(self):
        """获取当前观察值"""
        return {
            "agent": self._agent_location,
            "target": self._target_location
        }

    def _get_info(self):
        """获取调试信息（曼哈顿距离）"""
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            )
        }

    def reset(self, seed=None, options=None):
        """重置环境状态"""
        super().reset(seed=seed)

        # 随机初始化代理位置
        self._agent_location = self.np_random.integers(
            0, self.size, size=2, dtype=int
        )

        # 随机生成目标位置（不与代理重合）
        self._target_location = self._agent_location
        while np.array_equal(self._target_location, self._agent_location):
            self._target_location = self.np_random.integers(
                0, self.size, size=2, dtype=int
            )

        # 初始化渲染
        if self.render_mode == "human":
            self._render_frame()

        return self._get_obs(), self._get_info()

    def step(self, action):
        """执行一个时间步"""
        # 计算新位置
        direction = self._action_to_direction[action]
        self._agent_location = np.clip(
            self._agent_location + direction,
            0,  # 最小坐标
            self.size - 1  # 最大坐标
        )

        # 判断是否终止
        terminated = np.array_equal(
            self._agent_location, self._target_location
        )
        reward = 1 if terminated else 0
        truncated = False  # 本环境不启用提前截断

        # 更新渲染
        if self.render_mode == "human":
            self._render_frame()

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def render(self):
        """渲染环境（PyGame实现）"""
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        """渲染单帧画面"""
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size)
            )
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))  # 白色背景

        # 计算网格单元像素尺寸
        pix_square_size = self.window_size / self.size

        # 绘制目标（红色方块）
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                pix_square_size * self._target_location,
                (pix_square_size, pix_square_size),
            ),
        )

        # 绘制代理（蓝色圆形）
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._agent_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        # 绘制网格线
        for x in range(self.size + 1):
            pygame.draw.line(
                canvas,
                0,  # 黑色
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        # 更新窗口显示
        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array模式
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        """关闭渲染资源"""
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()


# 测试代码（直接运行该文件时执行）
if __name__ == "__main__":
    env = GridWorldEnv(render_mode="human")
    obs, _ = env.reset()

    # 运行10步随机动作演示
    for _ in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(action)
        print(f"Agent Position: {obs['agent']}, Reward: {reward}")
        if terminated:
            obs, _ = env.reset()

    env.close()