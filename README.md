# Pursuit-Evasion Game with Obstacle

## Overview
This project solves a planar attacker-defender game with a circular obstacle, finding Nash equilibrium strategies for both players.

## Problem Description
- Attacker tries to reach bottom boundary (target)
- Defender tries to capture attacker (distance < ε)
- Circular obstacle blocks direct path
- Both players have perfect information
- Zero-sum game with payoff J

## Solution Approach
The problem is decomposed into three phases:

1. **Phase 3 (After Obstacle)**: Both players on same side, heading to target
2. **Phase 2 (At Obstacle)**: Critical decision point - 2x2 matrix game
3. **Phase 1 (Before Obstacle)**: Dynamic programming to reach decision point
