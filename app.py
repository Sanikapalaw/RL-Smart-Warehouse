import streamlit as st
import numpy as np
import pandas as pd
import time

# Simulation Settings
GRID_SIZE = 5
GOAL = (4, 4)

class RobustAgent:
    def __init__(self):
        # Value Function for [Row, Col, Action]
        self.q_table = np.zeros((GRID_SIZE, GRID_SIZE, 4))
        self.alpha = 0.1 # Learning rate
        self.gamma = 0.9

    def step(self, state, action, signal_working=True):
        # Uncertain moves (Uncertainty mentioned in Chapter 3)
        if not signal_working:
            # If signal is broken, 20% chance the car moves in a random direction
            if np.random.rand() < 0.2:
                action = np.random.randint(0, 4)
        
        new_r, new_c = state
        if action == 0 and state[0] > 0: new_r -= 1 # Up
        elif action == 1 and state[0] < GRID_SIZE-1: new_r += 1 # Down
        elif action == 2 and state[1] > 0: new_c -= 1 # Left
        elif action == 3 and state[1] < GRID_SIZE-1: new_c += 1 # Right
        
        # Reward
        if (new_r, new_c) == GOAL: reward = 10
        else: reward = -1
        
        return (new_r, new_c), reward

# --- STREAMLIT GUI ---
st.title("🚗 Self-Driving: Broken Signal Recovery")

if 'car_agent' not in st.session_state:
    st.session_state.car_agent = RobustAgent()
    st.session_state.car_pos = (0, 0)

# The "Signal Status"
signal_status = st.toggle("Traffic Signal Working", value=True)

if not signal_status:
    st.warning("⚠️ Signal is NOT working. Agent is navigating under UNCERTAINTY.")

# Value Function Logic
if st.button("Move AI One Step"):
    state = st.session_state.car_pos
    # Policy: Greedy action based on current Value Function
    action = np.argmax(st.session_state.car_agent.q_table[state[0], state[1]])
    
    # Environment Interaction
    next_state, reward = st.session_state.car_agent.step(state, action, signal_status)
    
    # Policy Improvement (Updating from the "Mistake")
    old_q = st.session_state.car_agent.q_table[state[0], state[1], action]
    max_next_q = np.max(st.session_state.car_agent.q_table[next_state[0], next_state[1]])
    st.session_state.car_agent.q_table[state[0], state[1], action] = old_q + st.session_state.car_agent.alpha * (reward + st.session_state.car_agent.gamma * max_next_q - old_q)
    
    st.session_state.car_pos = next_state
    
    # Draw Grid
    grid = np.full((GRID_SIZE, GRID_SIZE), "⬜")
    grid[GOAL] = "🏁"
    grid[st.session_state.car_pos] = "🏎️"
    st.table(grid)
    
    if next_state == GOAL:
        st.success("Reached Goal despite signal failure!")
        st.session_state.car_pos = (0, 0)
