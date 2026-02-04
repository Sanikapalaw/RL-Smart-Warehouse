import streamlit as st
import numpy as np
import pandas as pd

st.set_page_config(page_title="SCM Bandit Game", layout="wide")

# --- Title and Intro ---
st.title("📦 SCM Procurement: The Multi-Armed Bandit Game")
st.markdown("""
Based on **Chapter 2 of Sutton & Barto**, this game simulates the **k-armed Bandit problem**.
**Your Task:** Choose the best supplier to maximize total profit. 
Some suppliers are consistent; others are risky but high-reward!
""")

# --- Game Logic / Model Parameters ---
if 'round' not in st.session_state:
    st.session_state.round = 1
    st.session_state.total_profit = 0
    st.session_state.history = []
    # Hidden True Means (The "Reality" the player tries to learn)
    st.session_state.suppliers = {
        "Supplier A (Local)": {"mean": 50, "std": 5},    # Stable, Low Reward
        "Supplier B (Global)": {"mean": 70, "std": 20},  # Risky, Medium Reward
        "Supplier C (Startup)": {"mean": 90, "std": 50}  # Very Risky, High Reward
    }

# --- Sidebar Stats ---
st.sidebar.header("Dashboard")
st.sidebar.metric("Current Round", f"{st.session_state.round} / 20")
st.sidebar.metric("Total Profit", f"${st.session_state.total_profit:,.0f}")

if st.session_state.round <= 20:
    st.subheader(f"Round {st.session_state.round}: Select a Supplier")
    
    cols = st.columns(3)
    
    for i, (name, stats) in enumerate(st.session_state.suppliers.items()):
        if cols[i].button(f"Order from {name}"):
            # Generate reward based on normal distribution (Chapter 2.1)
            reward = np.random.normal(stats['mean'], stats['std'])
            st.session_state.total_profit += reward
            st.session_state.history.append({"Round": st.session_state.round, "Supplier": name, "Profit": reward})
            st.session_state.round += 1
            st.rerun()

# --- Visualization (The Learning Process) ---
if st.session_state.history:
    df = pd.DataFrame(st.session_state.history)
    
    st.divider()
    st.subheader("Profit Performance")
    st.line_chart(df.pivot(index='Round', columns='Supplier', values='Profit'))

    # Show Average Reward per 'Arm' (Chapter 2.4 - Action-value Methods)
    st.subheader("Estimated Supplier Value ($Q_t$)")
    avg_rewards = df.groupby('Supplier')['Profit'].mean()
    st.bar_chart(avg_rewards)
    
    st.write("This bar chart shows your **Action-Value Estimate ($Q_t(a)$)**. In Chapter 2, RL agents use this to decide which supplier to 'exploit'.")

# --- Reset Game ---
if st.session_state.round > 20:
    st.success(f"Game Over! Final Profit: ${st.session_state.total_profit:,.0f}")
    if st.button("Restart Simulation"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
