import streamlit as st

import numpy as np

import pandas as pd

import time



# --- SETUP ---

GRID_SIZE = 6

GOAL = (5, 5)

TRAFFIC_LIGHT = (2, 2) # A "red light" obstacle



class SelfDrivingGUI:

    def __init__(self):

        # Q-Table (Value Function): [Row, Col, Action(Up, Down, Left, Right)]

        self.q_table = np.zeros((GRID_SIZE, GRID_SIZE, 4))

        self.lr = 0.2

        self.gamma = 0.9



    def train(self, epochs):

        for _ in range(epochs):

            state = (0, 0) # Start at top-left

            for _ in range(20): # Max steps per epoch

                action = np.random.randint(0, 4)

                # Apply move

                new_row, new_col = state

                if action == 0 and state[0] > 0: new_row -= 1

                elif action == 1 and state[0] < GRID_SIZE-1: new_row += 1

                elif action == 2 and state[1] > 0: new_col -= 1

                elif action == 3 and state[1] < GRID_SIZE-1: new_col += 1

                

                # Reward Logic (Learning from Mistakes)

                if (new_row, new_col) == GOAL:

                    reward = 100

                elif (new_row, new_col) == TRAFFIC_LIGHT:

                    reward = -50 # Penalty for "running the light"

                else:

                    reward = -1 # Small movement penalty

                

                # Update Value Function (Q-Learning)

                old_val = self.q_table[state[0], state[1], action]

                next_max = np.max(self.q_table[new_row, new_col])

                self.q_table[state[0], state[1], action] = old_val + self.lr * (reward + self.gamma * next_max - old_val)

                

                state = (new_row, new_col)

                if state == GOAL: break



# --- GUI INTERFACE ---

st.title("🚗 GUI Self-Driving Lab")

st.markdown("Watching the agent learn from **Value Functions** and **Policy Improvement**.")



if 'rl_agent' not in st.session_state:

    st.session_state.rl_agent = SelfDrivingGUI()



# Sidebar Training

st.sidebar.header("Training Controls")

train_epochs = st.sidebar.slider("Epochs (Learning Cycles)", 10, 2000, 100)

if st.sidebar.button("Train AI Model"):

    with st.spinner("Agent exploring unknown environment..."):

        st.session_state.rl_agent.train(train_epochs)

    st.sidebar.success("Training Finished!")



# Drawing the Map

def draw_grid(car_pos):

    grid = np.full((GRID_SIZE, GRID_SIZE), "⬜")

    grid[GOAL] = "🏁"

    grid[TRAFFIC_LIGHT] = "🚦"

    grid[car_pos] = "🚗"

    return pd.DataFrame(grid)



# Game Control

st.subheader("Live Simulation")

if st.button("Run AI Policy"):

    curr_pos = (0, 0)

    placeholder = st.empty()

    

    for step in range(15):

        # Policy Improvement: Select best action from Value Function

        action = np.argmax(st.session_state.rl_agent.q_table[curr_pos[0], curr_pos[1]])

        

        # Calculate new position

        new_r, new_c = curr_pos

        if action == 0 and curr_pos[0] > 0: new_r -= 1

        elif action == 1 and curr_pos[0] < GRID_SIZE-1: new_r += 1

        elif action == 2 and curr_pos[1] > 0: new_c -= 1

        elif action == 3 and curr_pos[1] < GRID_SIZE-1: new_c += 1

        

        curr_pos = (new_r, new_c)

        

        # Update GUI

        with placeholder.container():

            st.table(draw_grid(curr_pos))

            st.write(f"Step: {step+1} | Current Move: {['Up', 'Down', 'Left', 'Right'][action]}")

        

        time.sleep(0.5)

        if curr_pos == GOAL:

            st.balloons()

            st.success("Goal Reached!")

            break

else:

    st.table(draw_grid((0, 0)))



st.divider()

st.subheader("The 'Brain' (Value Function Heatmap)")

st.write("This shows which areas the car thinks are 'Good' (High numbers) vs 'Dangerous' (Low numbers).")

st.write(np.max(st.session_state.rl_agent.q_table, axis=2))
