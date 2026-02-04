import streamlit as st
import numpy as np
import random
import matplotlib.pyplot as plt
import time

st.set_page_config(page_title="RL Smart Warehouse", layout="wide")

# -------------------------
# Q-learning core
# -------------------------
states = [(arm, stock) for arm in range(3) for stock in range(3)]
actions = ["move_left", "move_right", "reorder", "idle"]

Q = {s: {a: 0 for a in actions} for s in states}

alpha, gamma, epsilon = 0.1, 0.9, 0.2

def reward_fn(state, action):
    arm, stock = state
    if action == "reorder" and stock == 0: return 10
    if action == "idle" and stock == 0: return -10
    if action.startswith("move"): return -1
    return 1

def transition(state, action):
    arm, stock = state
    if action == "move_left": arm = max(0, arm-1)
    elif action == "move_right": arm = min(2, arm+1)
    elif action == "reorder": stock = min(2, stock+1)
    elif action == "idle": stock = max(0, stock-1)
    return (arm, stock)

# -------------------------
# UI
# -------------------------
st.title("🤖 RL Smart Warehouse Arm")

if "state" not in st.session_state:
    st.session_state.state = random.choice(states)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Robotic Arm")
    fig, ax = plt.subplots()
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 2)
    ax.set_title("Arm Position")

    def draw_arm(pos):
        ax.clear()
        ax.set_xlim(0, 3)
        ax.set_ylim(0, 2)
        ax.plot([pos+0.5], [1], 'ro', markersize=15)
        ax.text(pos+0.4, 0.5, f"Arm at {pos}")
        st.pyplot(fig)

    draw_arm(st.session_state.state[0])

with col2:
    st.subheader("Inventory")
    st.metric("Stock Level", st.session_state.state[1])

# -------------------------
# Step button
# -------------------------
if st.button("▶ Run RL Step"):
    s = st.session_state.state

    if random.random() < epsilon:
        a = random.choice(actions)
    else:
        a = max(Q[s], key=Q[s].get)

    r = reward_fn(s, a)
    s2 = transition(s, a)

    Q[s][a] += alpha * (r + gamma * max(Q[s2].values()) - Q[s][a])

    st.session_state.state = s2
    st.success(f"Action: {a} | Reward: {r}")
    time.sleep(0.3)
    st.rerun()
