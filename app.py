# app.py
import streamlit as st
import random, string, time
import numpy as np
from collections import defaultdict

# ------------------------
# LEVEL SETTINGS
# ------------------------
LEVELS = {
    "Easy": {"size": 8, "dirs": ["H", "V"]},
    "Hard": {"size": 16, "dirs": ["H", "V", "D1", "D2"]},
    "Advance": {"size": 32, "dirs": ["H", "V", "D1", "D2"]}
}

WORDS = ["DATA", "AI", "MODEL", "RL", "TRAIN", "AGENT"]

ACTIONS = {
    0:(-1,0),1:(1,0),2:(0,-1),3:(0,1),
    4:(-1,-1),5:(-1,1),6:(1,-1),7:(1,1)
}

Q = defaultdict(lambda: np.zeros(8))

alpha, epsilon, episodes = 0.1, 0.3, 1500

# ------------------------
# GRID
# ------------------------
def create_grid(n):
    return [[random.choice(string.ascii_uppercase) for _ in range(n)] for _ in range(n)]

def hide_word(grid, word, dirs):
    n=len(grid)
    dirmap={"H":(0,1),"V":(1,0),"D1":(1,1),"D2":(1,-1)}
    while True:
        dx,dy=dirmap[random.choice(dirs)]
        x=random.randint(0,n-1); y=random.randint(0,n-1)
        if 0<=x+dx*(len(word)-1)<n and 0<=y+dy*(len(word)-1)<n:
            for i in range(len(word)):
                grid[x+dx*i][y+dy*i]=word[i]
            return

# ------------------------
# RL
# ------------------------
def reward_fn(grid,x,y,nx,ny,word,idx,visited):
    n=len(grid)
    if nx<0 or ny<0 or nx>=n or ny>=n: return -5, idx
    if (nx,ny) in visited: return -3, idx
    if grid[nx][ny]==word[idx]:
        if idx==len(word)-1: return 100, idx+1
        return 10, idx+1
    return -1, idx

def choose(state):
    if random.random()<epsilon: return random.randint(0,7)
    return np.argmax(Q[state])

def train(grid, word):
    n=len(grid)
    for _ in range(episodes):
        x=random.randint(0,n-1); y=random.randint(0,n-1)
        idx=0; visited={(x,y)}
        for _ in range(200):
            s=(x,y,idx)
            a=choose(s)
            dx,dy=ACTIONS[a]
            nx,ny=x+dx,y+dy
            r,new_idx=reward_fn(grid,x,y,nx,ny,word,idx,visited)
            Q[s][a]+=alpha*(r-Q[s][a])
            if r>-5:
                x,y,idx=nx,ny,new_idx
                visited.add((x,y))
            if idx==len(word): break

# ------------------------
# STREAMLIT UI
# ------------------------
st.title("🤖 AI Word Search (Reinforcement Learning)")

level = st.selectbox("Choose Level", list(LEVELS.keys()))
if st.button("Start AI"):
    cfg = LEVELS[level]
    grid = create_grid(cfg["size"])

    for w in WORDS:
        hide_word(grid, w, cfg["dirs"])

    word = random.choice(WORDS)
    st.success(f"AI selected word: {word}")

    train(grid, word)

    st.subheader("Grid")
    for row in grid:
        st.write(" ".join(row))

    st.balloons()
    st.success("AI learned how to search! 🎯")
