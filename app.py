import streamlit as st
import random

st.set_page_config(page_title="Treasure Rooms AI", layout="centered")
st.title("🎮 Treasure Rooms AI (Associative Search)")

rooms = [0,1,2]
doors = [0,1,2]

if "Q" not in st.session_state:
    st.session_state.Q = {r:{d:0.0 for d in doors} for r in rooms}
    st.session_state.true_best = {0:1, 1:2, 2:0}
    st.session_state.epsilon = 0.3
    st.session_state.alpha = 0.1

room = st.selectbox("Choose a Room", rooms)

if st.button("🤖 AI Choose Door"):
    Q = st.session_state.Q

    if random.random() < st.session_state.epsilon:
        door = random.choice(doors)
        mode = "Exploring"
    else:
        door = max(Q[room], key=Q[room].get)
        mode = "Exploiting"

    reward = 1 if door == st.session_state.true_best[room] else 0
    Q[room][door] += st.session_state.alpha * (reward - Q[room][door])

    st.success(f"AI chose Door {door} ({mode})")
    st.write("Reward:", "💎 Treasure!" if reward==1 else "❌ Empty")

st.subheader("🧠 Learned Preferences")
for r in rooms:
    st.write(f"Room {r}:", st.session_state.Q[r])
