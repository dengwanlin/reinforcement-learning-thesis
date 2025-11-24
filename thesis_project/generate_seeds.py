#!/usr/bin/env python3
import secrets

N = 5   # the number of seeds to generate
RANGE = 2**31 - 1   # SB3/Torch compatible seed range

seeds = set()
while len(seeds) < N:
    s = secrets.randbelow(RANGE)
    if s not in seeds:
        seeds.add(s)

print("Generated seeds:")
for s in sorted(seeds):
    print(s)
