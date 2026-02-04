import streamlit as st
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

# --- CONSTANTS ---
GOAL_X = 100.0
SIGNAL_X = 50.0
MAX_STEPS = 50

class ContinuousCarAgent:
    def __init__(self):
        # Using a simplified Q-table by binning continuous values
        # Bins: [Position (10), Velocity (5), Light (2), Action (3)]
        # Actions: 0: Brake, 1: Coast, 2: Accelerate
        self.q_table = np.zeros((11, 6, 2, 3)) 
        self.lr = 0.1
        self.gamma = 0.95

    def get_bins(self, pos, vel):
        p_bin = int(min(pos, 99) // 10)
        v_bin = int(min(max(vel, 0), 4))
        return p_bin, v_bin

    def train(self, epochs):
        for _ in range(epochs):
            pos, vel = 0.0, 0.0
            light = 0 # 0: Green, 1: Red
            
            for _ in range(MAX_STEPS):
                p_bin, v_bin = self.get_bins(pos, vel)
                action = np.random.randint(0, 3)
                
                # Environment Uncertainty (Unpredictable physics)
                noise = np.random.normal(0, 0.5) 
                acceleration = (action - 1) * 2.0 + noise
                
                new_vel = max(0, vel + acceleration)
                new_pos = pos + new_vel
                
                # Reward Logic (Chapter 3)
                if new_pos >= GOAL_X:
                    reward = 100
                elif abs(new_pos - SIGNAL_X) < 5 and light == 1 and new_vel > 1:
                    reward = -200 # Collision/Signal mistake
                else:
                    reward = -1 # Efficiency penalty

                # Update Value Function
                next_p, next_v = self.get_bins(new_pos, new_vel)
                next_light = 1 if np.random.rand() > 0.8 else 0
                old_val = self.q_table[p_bin, v_bin, light, action]
                next_max = np.max(self.q_table[next_p, next_v, next_light])
                
                self.q_table[p_bin, v_bin, light, action] = \
                    old_val + self.lr * (reward + self.gamma * next_max - old_val)
                
                pos, vel = new_pos, new_vel
                if pos >= GOAL_X: break

# --- GUI ---
st.title("🏎️ Continuous Self-Driving (Unpredictable Env)")
st.write("No grids here. The car uses physics and probability to learn.")

if 'agent' not in st.session_state:
    st.session_state.agent = ContinuousCarAgent()

epochs = st.sidebar.number_input("Training Epochs", 100, 10000, 1000)
if st.sidebar.button("Train Agent"):
    with st.spinner("Learning physics and signal timing..."):
        st.session_state.agent.train(epochs)
    st.sidebar.success("Training Complete!")

if st.button("Run Continuous Simulation"):
    pos, vel = 0.0, 0.0
    history = []
    placeholder = st.empty()
    
    for t in range(MAX_STEPS):
        # Logic: Switch light every 5 steps
        light = 1 if (t // 5) % 2 == 1 else 0
        p_bin, v_bin = st.session_state.agent.get_bins(pos, vel)
        
        # Policy: Best action from Q-table
        action = np.argmax(st.session_state.agent.q_table[p_bin, v_bin, light])
        
        # Physics with Noise
        noise = np.random.normal(0, 0.2)
        vel = max(0, vel + (action - 1) * 2.0 + noise)
        pos += vel
        
        history.append({"Time": t, "Position": pos, "Velocity": vel, "Light": "Red" if light else "Green"})
        
        # Draw Visuals
        fig, ax = plt.subplots(figsize=(10, 2))
        ax.set_xlim(0, 110)
        ax.set_ylim(-1, 1)
        ax.axvline(SIGNAL_X, color='red' if light else 'green', linestyle='--', label="Signal")
        ax.axvline(GOAL_X, color='gold', linewidth=3, label="Goal")
        ax.scatter([pos], [0], color='blue', s=200, marker='s', label="Car")
        ax.legend()
        placeholder.pyplot(fig)
        plt.close()
        
        time.sleep(0.2)
        if pos >= GOAL_X:
            st.success("Target Reached!")
            break

    st.line_chart(pd.DataFrame(history).set_index("Time")[["Velocity", "Position"]])
