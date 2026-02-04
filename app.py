import streamlit as st
import numpy as np
import random

st.title("🚦 Self-Driving Car: Traffic Signal RL")

# --- PARAMETERS ---
ACTIONS = ["Stop", "Drive"]
LIGHTS = ["Green", "Red"]

class TrafficAgent:
    def __init__(self):
        # Q-Table: [Light_State, Action]
        self.q_table = np.zeros((2, 2)) 
        self.lr = 0.1
        self.gamma = 0.9

    def train(self, epochs):
        log = []
        for _ in range(epochs):
            state = random.randint(0, 1) # 0: Green, 1: Red
            action = random.randint(0, 1) # 0: Stop, 1: Drive
            
            # Reward Logic (Learning from mistakes)
            if state == 1 and action == 1: # Red + Drive
                reward = -100 # Penalty
            elif state == 1 and action == 0: # Red + Stop
                reward = 10   # Correct
            elif state == 0 and action == 1: # Green + Drive
                reward = 20   # Efficient
            else: # Green + Stop
                reward = -10  # Waste of time
            
            # Update Value Function (Q-learning)
            self.q_table[state, action] += self.lr * (reward - self.q_table[state, action])
        return self.q_table

# --- UI ---
epochs = st.sidebar.slider("Training Epochs (Learning Cycles)", 10, 1000, 100)

if 'agent' not in st.session_state:
    st.session_state.agent = TrafficAgent()

if st.sidebar.button("Train Car AI"):
    st.session_state.q_table = st.session_state.agent.train(epochs)
    st.sidebar.success("AI Trained!")

# --- THE GAME ---
st.subheader("Test the Driver")
current_light = st.radio("Set Traffic Light:", ["Green", "Red"])
light_idx = 0 if current_light == "Green" else 1

col1, col2 = st.columns(2)

with col1:
    if st.button("Manual: Drive"):
        if current_light == "Red":
            st.error("💥 CRASH! You jumped a red light.")
        else:
            st.success("✅ Smooth driving!")

with col2:
    if st.button("AI: Decide"):
        if 'q_table' in st.session_state:
            # Policy: Choose action with highest Value
            ai_action_idx = np.argmax(st.session_state.q_table[light_idx])
            ai_decision = ACTIONS[ai_action_idx]
            st.write(f"AI chose to: **{ai_decision}**")
            
            if current_light == "Red" and ai_decision == "Drive":
                st.error("AI Crashed! (Needs more epochs to learn)")
            else:
                st.success("AI handled the signal correctly!")
        else:
            st.warning("Train the AI first!")

st.divider()
st.write("### Learned Value Function (Q-Table)")
if 'q_table' in st.session_state:
    st.write(st.session_state.q_table)
    st.caption("Rows: [Green, Red] | Columns: [Stop, Drive]")
