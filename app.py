import streamlit as st
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
MAX_X = 100.0
SIGNAL_X = 50.0
GOAL_X = 90.0
ACTIONS = ["BRAKE", "COAST", "ACCELERATE"]

class ContinuousAgent:
    def __init__(self):
        # State Bins: [Position(10), Velocity(5), Light(2)] 
        # Action: 3 choices
        self.q_table = np.zeros((11, 6, 2, 3)) 
        self.lr = 0.1      # Learning Rate (Alpha)
        self.gamma = 0.9   # Discount Factor
        self.epsilon = 0.2 # Exploration Rate

    def get_state_bins(self, x, v):
        x_bin = int(min(x, MAX_X-1) // 10)
        v_bin = int(min(max(v, 0), 4))
        return x_bin, v_bin

    def train_epoch(self, epochs):
        for _ in range(epochs):
            x, v = 0.0, 0.0
            light = 1 if np.random.rand() > 0.5 else 0 # Randomly start with Red or Green
            
            for _ in range(60): # Max steps per epoch
                x_b, v_b = self.get_state_bins(x, v)
                
                # Epsilon-Greedy Policy (Chapter 2.2)
                if np.random.rand() < self.epsilon:
                    action = np.random.randint(0, 3)
                else:
                    action = np.argmax(self.q_table[x_b, v_b, light])

                # Unpredictable Physics (Uncertainty)
                # Actual acceleration has random noise (skidding/wind)
                noise = np.random.normal(0, 0.5)
                accel = (action - 1) * 1.5 + noise
                
                v = max(0, min(v + accel, 5))
                x += v
                
                # Reward Signal & Penalty (Learning from Mistakes)
                if x >= GOAL_X:
                    reward = 100
                elif abs(x - SIGNAL_X) < 5 and light == 1 and v > 0.5:
                    reward = -150 # PENALTY: Running a Red Light
                elif abs(x - SIGNAL_X) < 5 and light == 1 and v <= 0.5:
                    reward = 10   # REWARD: Stopping correctly at Red
                else:
                    reward = -1   # Step cost

                # Value Function Update (Q-Learning / Policy Improvement)
                nx_b, nv_b = self.get_state_bins(x, v)
                n_light = 1 if np.random.rand() > 0.8 else 0 # Light might change
                old_q = self.q_table[x_b, v_b, light, action]
                max_future_q = np.max(self.q_table[nx_b, nv_b, n_light])
                
                # TD Update Rule
                self.q_table[x_b, v_b, light, action] += self.lr * (reward + self.gamma * max_future_q - old_q)
                
                if x >= GOAL_X: break

# --- STREAMLIT GUI ---
st.title("🏎️ Continuous RL: Policy & Uncertainty")
st.markdown("This car learns to navigate a non-grid world. It must learn the **meaning of the signal** through penalties.")

if 'agent' not in st.session_state:
    st.session_state.agent = ContinuousAgent()

# Sidebar for Training Epochs
st.sidebar.header("Training Lab")
epochs = st.sidebar.number_input("Number of Training Epochs", 100, 10000, 500)
if st.sidebar.button("Train AI Model"):
    with st.spinner("Agent is making mistakes and learning..."):
        st.session_state.agent.train_epoch(epochs)
    st.sidebar.success(f"Trained for {epochs} epochs!")

# Simulation
if st.button("Start AI Drive"):
    x, v = 0.0, 0.0
    placeholder = st.empty()
    
    for t in range(40):
        # Environment Change: Signal turns Red after 10 units of time
        light = 1 if 10 < t < 25 else 0 
        xb, vb = st.session_state.agent.get_state_bins(x, v)
        
        # Policy: Optimal Greedy (Chapter 3.6)
        action = np.argmax(st.session_state.agent.q_table[xb, vb, light])
        
        # Physics with Noise
        noise = np.random.normal(0, 0.2)
        v = max(0, min(v + (action - 1) * 1.5 + noise, 5))
        x += v

        # Drawing the Scene
        fig, ax = plt.subplots(figsize=(10, 2))
        ax.set_xlim(0, 100)
        ax.set_ylim(-1, 1)
        ax.axvline(SIGNAL_X, color='red' if light == 1 else 'green', label="Signal")
        ax.axvline(GOAL_X, color='gold', linewidth=4, label="Goal")
        ax.scatter([x], [0], color='blue', s=250, marker='s', label="Car")
        ax.set_title(f"Step {t} | Action: {ACTIONS[action]} | Velocity: {v:.2f}")
        ax.legend()
        placeholder.pyplot(fig)
        plt.close()

        time.sleep(0.15)
        if x >= GOAL_X:
            st.balloons()
            st.success("Successfully reached the goal!")
            break
