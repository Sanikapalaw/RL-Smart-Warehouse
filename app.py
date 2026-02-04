import streamlit as st
import numpy as np
import pandas as pd
import time

st.set_page_config(page_title="RL Inventory Game", layout="wide")

# --- RL AGENT CLASS (Q-Learning from Chapter 3) ---
class InventoryAgent:
    def __init__(self, n_states=21, n_actions=11):
        self.q_table = np.zeros((n_states, n_actions)) # 0 to 20 units
        self.lr = 0.1
        self.gamma = 0.9

    def get_reward(self, stock, order, demand):
        # SCM Logic: Profit - Storage Cost - Stockout Penalty
        sold = min(stock, demand)
        revenue = sold * 10
        storage_cost = stock * 1
        penalty = 0 if stock >= demand else (demand - stock) * 15
        return revenue - storage_cost - penalty

    def train(self, epochs):
        progress_bar = st.progress(0)
        for epoch in range(epochs):
            state = 10 # Start with 10 units
            for _ in range(30): # 30 days per epoch
                action = np.random.randint(0, 11) # Explore
                demand = np.random.poisson(5)
                reward = self.get_reward(state, action, demand)
                
                # Next state logic (MDP Transition)
                next_state = max(0, min(20, state - demand + action))
                
                # Q-Learning Formula (Learning from mistakes)
                old_value = self.q_table[state, action]
                next_max = np.max(self.q_table[next_state])
                self.q_table[state, action] = old_value + self.lr * (reward + self.gamma * next_max - old_value)
                state = next_state
            
            if epoch % (epochs // 10) == 0:
                progress_bar.progress(epoch / epochs)
        progress_bar.empty()

# --- STREAMLIT UI ---
st.title("📦 The SCM Training Lab")
st.write("Train an AI on the **Finite MDP model** (Chapter 3) to manage warehouse inventory.")

# 1. Training Phase
st.header("Step 1: Train the AI")
epochs = st.slider("Select Training Epochs (Learning Cycles)", 100, 5000, 1000)

if 'agent' not in st.session_state:
    st.session_state.agent = None

if st.button("Start Training AI"):
    with st.spinner("Agent is learning from mistakes..."):
        st.session_state.agent = InventoryAgent()
        st.session_state.agent.train(epochs)
        st.success(f"Training Complete! The AI has processed {epochs} cycles.")

# 2. The Game
if st.session_state.agent:
    st.divider()
    st.header("Step 2: Human vs. Trained AI")
    
    if 'day' not in st.session_state:
        st.session_state.day = 1
        st.session_state.h_stock = 10
        st.session_state.ai_stock = 10
        st.session_state.h_score = 0
        st.session_state.ai_score = 0

    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Your Stock", st.session_state.h_stock)
        order = st.number_input("How many units to order today?", 0, 10, 5)
        if st.button("Confirm Order"):
            demand = np.random.poisson(5)
            
            # Human Turn
            h_reward = st.session_state.agent.get_reward(st.session_state.h_stock, order, demand)
            st.session_state.h_score += h_reward
            st.session_state.h_stock = max(0, min(20, st.session_state.h_stock - demand + order))
            
            # AI Turn (Uses learned Q-Table)
            ai_action = np.argmax(st.session_state.agent.q_table[st.session_state.ai_stock])
            ai_reward = st.session_state.agent.get_reward(st.session_state.ai_stock, ai_action, demand)
            st.session_state.ai_score += ai_reward
            st.session_state.ai_stock = max(0, min(20, st.session_state.ai_stock - demand + ai_action))
            
            st.session_state.day += 1
            st.write(f"**Day {st.session_state.day-1} Results:** Demand was {demand}. AI ordered {ai_action}.")

    with col2:
        st.metric("AI Score", f"${st.session_state.ai_score}")
        st.metric("Your Score", f"${st.session_state.h_score}")

    if st.session_state.day > 10:
        st.balloons()
        st.write("### Final Results")
        if st.session_state.ai_score > st.session_state.h_score:
            st.error("The AI learned better! It minimized storage and stockout costs.")
        else:
            st.success("You beat the AI!")
