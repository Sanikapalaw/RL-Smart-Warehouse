import random
import numpy as np
from collections import defaultdict

# ---------------------------
# Environment (Grid World)
# ---------------------------
GRID = [
    ['D','A','T','A'],
    ['X','X','X','X'],
    ['X','X','X','X'],
    ['X','X','X','X']
]

WORD = "DATA"

ACTIONS = {
    0:(-1,0),1:(1,0),2:(0,-1),3:(0,1),
    4:(-1,-1),5:(-1,1),6:(1,-1),7:(1,1)
}

Q = defaultdict(lambda: np.zeros(8))

alpha = 0.1
epsilon = 0.3
episodes = 1000

# ---------------------------
# Reward Function
# ---------------------------
def reward_fn(x,y,nx,ny,idx,visited):
    n = len(GRID)
    if nx<0 or ny<0 or nx>=n or ny>=n: return -5, idx
    if (nx,ny) in visited: return -3, idx
    if GRID[nx][ny] == WORD[idx]:
        if idx == len(WORD)-1:
            return 100, idx+1
        return 10, idx+1
    return -1, idx

# ---------------------------
# Policy
# ---------------------------
def choose(state):
    if random.random() < epsilon:
        return random.randint(0,7)
    return np.argmax(Q[state])

# ---------------------------
# Training Loop
# ---------------------------
for ep in range(episodes):
    x,y = random.randint(0,3), random.randint(0,3)
    idx = 0
    visited = {(x,y)}

    for step in range(50):
        state = (x,y,idx)
        action = choose(state)

        dx,dy = ACTIONS[action]
        nx,ny = x+dx, y+dy

        r,new_idx = reward_fn(x,y,nx,ny,idx,visited)

        Q[state][action] += alpha * (r - Q[state][action])

        if r > -5:
            x,y,idx = nx,ny,new_idx
            visited.add((x,y))

        if idx == len(WORD):
            print(f"Episode {ep}: Word Found!")
            break

print("Training finished. Agent learned paths.")
