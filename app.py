import streamlit as st
import numpy as np
import pandas as pd
import time

# --- CONFIGURATION ---
GRID_SIZE = 5
START = (0, 0)
GOAL = (4, 4)
SIGNAL_POS = (2, 2)

class SelfDrivingCar:
    def __init__(self):
        # Value Function: 5x5 grid, 4 possible actions (0:Up, 1:Down, 2:Left, 3:Right)
        self.q_table = np.zeros((GRID_SIZE, GRID_SIZE, 4))
        self.lr = 0.1  # Alpha
        self.gamma = 0.9 # Discount
        self.epsilon = 0.2 # Exploration for uncertainty

    def train_epoch(self, signal_works=True):
        state = START
        total_reward = 0
        
        for _ in range(50): # Max steps to prevent infinite loops
            # Policy Selection (Epsilon-Greedy from Chapter 2)
            if np.random.uniform(0, 1) < self.epsilon:
                action = np.random.randint(0, 4)
            else:
                action = np.argmax(self.q_table[state[0], state[1]])

            # Move Logic with Uncertainty (Signal Not Working)
            if not signal_works and np.random.rand() < 0.3:
                # 30% chance the move is random if signal is broken
                action = np.random.randint(0, 4)

            # Calculate next state
            new_r, new_c = state
            if action == 0 and state[0] > 0: new_r -= 1
            elif action == 1 and state[0] < GRID_SIZE-1: new_r += 1
            elif action == 2 and state[1] > 0: new_c -= 1
            elif action == 3 and state[1] < GRID_SIZE-1: new_c += 1
            
            # Reward Signal (Learning from mistakes)
            if (new_r, new_c) == GOAL:
                reward = 100
            elif (new_r, new_c) == SIGNAL_POS and not signal_works:
                reward = -50 # Penalty for hitting the broken signal area
            else:
                reward = -1 # Small step cost

            # Update Value Function
            old_value = self.q_table[state[0], state[1], action]
            next_max = np.max(self.q_table[new_r, new_c])
            self.q_table[state[0], state[1], action] = old_value + self.lr * (reward + self.gamma * next_max - old_value)
            
            state = (new_r, new_c)
            total_reward += reward
            if state == GOAL: break
            
        return total_reward

# --- STREAMLIT UI ---
st.title("🚗 Training Lab: Epochs & Values")

if 'car' not in st.session_state:
    st.session_state.car = SelfDrivingCar()
    st.session_state.history = []

# Training Interface
st.sidebar.header("Agent Brain")
num_epochs = st.sidebar.number_input("How many Epochs?", min_value=10, max_value=5000, value=100)
signal_status = st.sidebar.checkbox("Signal is Working", value=False)

if st.sidebar.button("Run Training"):
    progress = st.progress(0)
    for i in range(num_epochs):
        reward = st.session_state.car.train_epoch(signal_status)
        st.session_state.history.append(reward)
        if i % (num_epochs // 10) == 0:
            progress.progress(i / num_epochs)
    st.sidebar.success(f"Trained for {num_epochs} Epochs!")

# Results GUI
st.subheader("Training Progress (Learning from Mistakes)")
if st.session_state.history:
    # Plotting how the reward increases as the agent learns
    chart_data = pd.DataFrame(st.session_state.history, columns=["Total Reward"])
    st.line_chart(chart_data)
    st.caption("As the line goes up, the agent is making fewer mistakes (like hitting the signal or getting lost).")

# Visual Grid
st.subheader("Learned Policy Map")
# Display the best move for each square
best_moves = np.argmax(st.session_state.car.q_table, axis=2)
move_icons = {0: "⬆️", 1: "⬇️", 2: "⬅️", 3: "➡️"}
grid_display = [[move_icons[best_moves[r,c]] for c in range(GRID_SIZE)] for r in range(GRID_SIZE)]
st.table(pd.DataFrame(grid_display))
