#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import gymnasium as gym

def main():
    print("=== Registered Gymnasium Envs ===")
    all_envs = sorted(gym.envs.registry.keys())
    print(f"Total: {len(all_envs)}")
    for env_id in all_envs:
        print(env_id)

if __name__ == "__main__":
    main()
