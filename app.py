import streamlit as st
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
SIGNAL_X = 50.0
GOAL_X = 90.0
ACTIONS = ["BRAKE", "COAST", "ACCELERATE"]

class ContinuousAgent:
    def __init__(self):
        # State: [Position, Velocity, Light] | Actions: 3
        self.q_table = np.zeros((11, 6, 2, 3)) 
        self.lr = 0.2
        self.gamma = 0.9
        self.epsilon = 0.2

    def get_state_bins(self, x, v):
        x_bin = int(min(x, 99) // 10)
        v_bin = int(min(max(v, 0), 4))
        return x_bin, v_bin

    def train_epoch(self, epochs):
        for _ in range(epochs):
            x, v = 0.0, 0.0
            light = np.random.randint(0, 2)
            for _ in range(50):
                xb, vb = self.get_state_bins(x, v)
                action = np.random.randint(0, 3) if np.random.rand() < self.epsilon else np.argmax(self.q_table[xb, vb, light])
                
                # Physics
                v = max(0, min(v + (action - 1) * 1.5 + np.random.normal(0, 0.2), 5))
                x += v
                
                # Reward Logic (Sutton & Barto Chapter 3)
                if x >= GOAL_X: reward = 100
                elif abs(x - SIGNAL_X) < 5 and light == 1 and v > 0.5: reward = -150 # Penalty
                elif abs(x - SIGNAL_X) < 5 and light == 1 and v <= 0.5: reward = 20  # Reward for stopping
                else: reward = -1
                
                nxb, nvb = self.get_state_bins(x, v)
                n_light = np.random.randint(0, 2)
                old_q = self.q_table[xb, vb, light, action]
                self.q_table[xb, vb, light, action] = old_val = old_q + self.lr * (reward + self.gamma * np.max(self.q_table[nxb, nvb, n_light]) - old_q)
                if x >= GOAL_X: break

st.title("🏎️ Self-Driving: Training Monitor")

if 'agent' not in st.session_state:
    st.session_state.agent = ContinuousAgent()

# Training
epochs = st.sidebar.number_input("Train for how many cycles?", 100, 5000, 500)
if st.sidebar.button("Train AI Now"):
    st.session_state.agent.train_epoch(epochs)
    st.sidebar.success("Training Done!")

# Simulation
if st.button("Run & Watch Rewards"):
    x, v = 0.0, 0.0
    history = []
    placeholder = st.empty()
    table_placeholder = st.empty()
    
    for t in range(40):
        light = 1 if 10 < t < 25 else 0 # Red between step 10 and 25
        xb, vb = st.session_state.agent.get_state_bins(x, v)
        
        # Policy: Best Action (Optimal Policy Chapter 3.6)
        action = np.argmax(st.session_state.agent.q_table[xb, vb, light])
        
        # Move
        v = max(0, min(v + (action - 1) * 1.5, 5))
        x += v
        
        # Get immediate reward for the table
        if x >= GOAL_X: current_r = 100
        elif abs(x - SIGNAL_X) < 5 and light == 1 and v > 0.5: current_r = -150
        elif abs(x - SIGNAL_X) < 5 and light == 1 and v <= 0.5: current_r = 20
        else: current_r = -1
        
        history.append({
            "Step": t, 
            "Light": "RED" if light == 1 else "GREEN", 
            "Action": ACTIONS[action], 
            "Reward/Penalty": current_r,
            "Velocity": round(v, 2)
        })

        # Display Visualization
        fig, ax = plt.subplots(figsize=(10, 2))
        ax.axvline(SIGNAL_X, color='red' if light == 1 else 'green', label="Signal")
        ax.scatter([x], [0], color='blue', s=200, marker='s')
        ax.set_xlim(0, 100)
        placeholder.pyplot(fig)
        plt.close()
        
        # Display Table
        table_placeholder.table(pd.DataFrame(history).tail(5))
        
        time.sleep(0.3)
        if x >= GOAL_X: break
