import streamlit as st
import numpy as np
import pandas as pd
import time
import random

# --- SETUP ---
GRID_SIZE = 6
GOAL = (5, 5)
# Multiple Signals: (Row, Col)
SIGNALS = [(1, 2), (3, 4), (2, 2)]

class SelfDrivingGUI:
    def __init__(self):
        # Q-Table (Value Function): [Row, Col, Light_Color, Action]
        # Actions: 0:Up, 1:Down, 2:Left, 3:Right, 4:Wait
        self.q_table = np.zeros((GRID_SIZE, GRID_SIZE, 2, 5)) 
        self.lr = 0.2
        self.gamma = 0.9

    def train(self, epochs):
        for _ in range(epochs):
            state = (0, 0)
            light_color = 0 # 0: Green, 1: Red
            
            for _ in range(30):
                # Random light change to simulate environment uncertainty
                light_color = 1 if random.random() > 0.7 else 0
                
                action = np.random.randint(0, 5)
                new_row, new_col = state
                
                # Apply Move Logic
                if action == 0 and state[0] > 0: new_row -= 1
                elif action == 1 and state[0] < GRID_SIZE-1: new_row += 1
                elif action == 2 and state[1] > 0: new_col -= 1
                elif action == 3 and state[1] < GRID_SIZE-1: new_col += 1
                # Action 4 is "Wait" - position stays same
                
                # Reward Logic: Learning from Mistakes
                if (new_row, new_col) == GOAL:
                    reward = 100
                elif state in SIGNALS and light_color == 1 and action != 4:
                    reward = -100 # Huge penalty for moving on Red
                elif state in SIGNALS and light_color == 1 and action == 4:
                    reward = 20 # Reward for waiting at Red light
                else:
                    reward = -1

                # Update Value Function (Q-Learning)
                old_val = self.q_table[state[0], state[1], light_color, action]
                next_light = 1 if random.random() > 0.7 else 0
                next_max = np.max(self.q_table[new_row, new_col, next_light])
                
                self.q_table[state[0], state[1], light_color, action] = \
                    old_val + self.lr * (reward + self.gamma * next_max - old_val)
                
                state = (new_row, new_col)
                if state == GOAL: break

# --- GUI INTERFACE ---
st.title("🚦 Advanced Multi-Signal Driving Lab")

if 'rl_agent' not in st.session_state:
    st.session_state.rl_agent = SelfDrivingGUI()

st.sidebar.header("Training Settings")
train_epochs = st.sidebar.slider("Epochs (Learning Cycles)", 100, 5000, 500)
if st.sidebar.button("Train AI on Signals"):
    with st.spinner("Learning the meaning of Green and Red..."):
        st.session_state.rl_agent.train(train_epochs)
    st.sidebar.success("Agent Learned!")

def draw_grid(car_pos, current_light_color):
    grid = np.full((GRID_SIZE, GRID_SIZE), "⬜")
    grid[GOAL] = "🏁"
    light_emoji = "🔴" if current_light_color == 1 else "🟢"
    for s in SIGNALS:
        grid[s] = light_emoji
    grid[car_pos] = "🚗"
    return pd.DataFrame(grid)

st.subheader("Live Simulation")
if st.button("Start Intelligent Drive"):
    curr_pos = (0, 0)
    placeholder = st.empty()
    
    for step in range(25):
        # Randomly switch light every few steps
        light_state = 1 if (step // 3) % 2 == 1 else 0
        
        # Policy Improvement: Select best action based on current state AND light color
        action = np.argmax(st.session_state.rl_agent.q_table[curr_pos[0], curr_pos[1], light_state])
        
        move_name = ["Up", "Down", "Left", "Right", "WAITING (Red Light)"][action]
        
        if action != 4:
            new_r, new_c = curr_pos
            if action == 0 and curr_pos[0] > 0: new_r -= 1
            elif action == 1 and curr_pos[0] < GRID_SIZE-1: new_r += 1
            elif action == 2 and curr_pos[1] > 0: new_c -= 1
            elif action == 3 and curr_pos[1] < GRID_SIZE-1: new_c += 1
            curr_pos = (new_r, new_c)
        
        with placeholder.container():
            st.table(draw_grid(curr_pos, light_state))
            st.write(f"Step: {step+1} | Light: {'RED' if light_state == 1 else 'GREEN'} | Action: **{move_name}**")
        
        time.sleep(0.6)
        if curr_pos == GOAL:
            st.balloons()
            st.success("Goal Reached Safely!")
            break
