import streamlit as st
import numpy as np
import pandas as pd
import random

# --- GAME CONFIGURATION ---
SCM_WORDS = ["LOGISTICS", "DEMAND", "STOCK", "SUPPLY", "TRUCK"]
GRID_SIZE = 10

def create_grid():
    grid = np.full((GRID_SIZE, GRID_SIZE), "-")
    for word in SCM_WORDS:
        placed = False
        while not placed:
            direction = random.choice(['H', 'V'])
            row = random.randint(0, GRID_SIZE - 1)
            col = random.randint(0, GRID_SIZE - 1)
            
            if direction == 'H' and col + len(word) <= GRID_SIZE:
                if all(grid[row, col+i] == "-" or grid[row, col+i] == word[i] for i in range(len(word))):
                    for i in range(len(word)): grid[row, col+i] = word[i]
                    placed = True
            elif direction == 'V' and row + len(word) <= GRID_SIZE:
                if all(grid[row+i, col] == "-" or grid[row+i, col] == word[i] for i in range(len(word))):
                    for i in range(len(word)): grid[row+i, col] = word[i]
                    placed = True
    
    # Fill remaining with random letters
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    for r in range(GRID_SIZE):
        for c in range(GRID_SIZE):
            if grid[r, c] == "-":
                grid[r, c] = random.choice(letters)
    return grid

# --- RL TRAINING (LEARNING FROM MISTAKES) ---
def train_agent(epochs):
    # Q-Table: [Row, Col, Action]
    q_table = np.zeros((GRID_SIZE, GRID_SIZE, 4)) 
    lr = 0.1
    gamma = 0.9
    
    progress = st.progress(0)
    for epoch in range(epochs):
        r, c = random.randint(0, 9), random.randint(0, 9)
        # Simplified learning: Agent gets reward for hitting any cell that is part of an SCM word
        # In a real MDP, this would be a sequence, but for a "Normal Game" we simplify the goal
        for _ in range(20):
            action = random.randint(0, 3) # 0:U, 1:D, 2:L, 3:R
            new_r, new_c = r, c
            if action == 0 and r > 0: new_r -= 1
            elif action == 1 and r < 9: new_r += 1
            elif action == 2 and c > 0: new_c -= 1
            elif action == 3 and c < 9: new_c += 1
            
            # Learning from mistake: Penalty for non-word cells
            reward = 5 if st.session_state.raw_grid[new_r, new_c] in "LOGISTICSDEMANDSTOCKSUPPLYTRUCK" else -2
            
            # Update Q-value
            q_table[r, c, action] += lr * (reward + gamma * np.max(q_table[new_r, new_c]) - q_table[r, c, action])
            r, c = new_r, new_c
            
        if epoch % (epochs//10) == 0:
            progress.progress(epoch/epochs)
    progress.empty()
    return q_table

# --- UI ---
st.title("🔍 SCM Word Search: RL Edition")
st.write("Train the agent to recognize SCM keywords in the grid.")

if 'raw_grid' not in st.session_state:
    st.session_state.raw_grid = create_grid()
    st.session_state.found = []

# 1. Training Section
st.header("1. Training the Brain")
epochs = st.select_slider("Select Training Intensity (Epochs)", options=[100, 500, 1000, 5000])

if st.button("Train Agent"):
    st.session_state.q_table = train_agent(epochs)
    st.success("Agent trained! It now 'prefers' cells that contribute to SCM words.")

# 2. Game Section
st.header("2. Find the Words")
st.write(f"Targets: {', '.join(SCM_WORDS)}")

# Display Grid
grid_df = pd.DataFrame(st.session_state.raw_grid)

def highlight_found(val):
    return 'background-color: yellow' if val in st.session_state.found else ''

st.table(grid_df)

# Input for User
user_word = st.text_input("Type a word you found:").upper()
if st.button("Submit Word"):
    if user_word in SCM_WORDS and user_word not in st.session_state.found:
        st.session_state.found.append(user_word)
        st.success(f"Correct! {user_word} is key to SCM.")
    else:
        st.error("Not an SCM word or already found. Try again!")

if len(st.session_state.found) == len(SCM_WORDS):
    st.balloons()
    st.header("🏆 You found all SCM terms!")
