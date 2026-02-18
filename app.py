import streamlit as st
import requests

API_URL = "http://172.17.212.129:8000"

st.set_page_config(page_title="AI Debate Arena", page_icon="⚖️")

if "joined" not in st.session_state:
    st.session_state.joined = False


def post(endpoint, data):
    try:
        res = requests.post(f"{API_URL}/{endpoint}", json=data)
        return res.json()
    except:
        return {"error": "Backend error"}


def get_state(room):
    try:
        res = requests.get(f"{API_URL}/state/{room}")
        return res.json()
    except:
        return {"error": "Backend error"}


st.title("⚖️ AI Debate Arena")

if not st.session_state.joined:

    name = st.text_input("Your Name")
    room = st.text_input("Room Code")

    if st.button("Join"):

        data = post("join", {"room": room, "name": name})

        if "error" in data:
            st.error(data["error"])
        else:
            st.session_state.name = name
            st.session_state.room = room
            st.session_state.joined = True
            st.rerun()

else:

    room = st.session_state.room
    name = st.session_state.name

    state = get_state(room)

    st.subheader(f"📝 Topic: {state['topic']}")

    role = state["roles"][name]
    st.info(f"🎭 You are debating: {role.upper()}")

    st.divider()

    for m in state["messages"]:

        st.write(f"**{m['name']}**: {m['text']}  (+{m['score']})")

        st.caption(
            f"Claim: {m['stats']['claim']} | "
            f"Logic: {m['stats']['logic']} | "
            f"Civility: {m['stats']['civility']} | "
            f"Relevance: {m['stats']['relevance']}"
        )

        st.divider()

    current = state["players"][state["turn"]]

    st.write("### Scoreboard")

    for p in state["scores"]:
        st.write(f"{p}: {round(state['scores'][p],2)}")

    st.divider()

    if current == name:

        text = st.text_area("Your Argument")

        if st.button("Send"):

            if not text.strip():
                st.warning("Type something first")
                st.stop()

            res = post("send", {
            "room": room,
            "name": name,
            "text": text
            })

            st.write("SEND DEBUG:", res)   # 👈 IMPORTANT

            if "error" in res:
                st.error(res["error"])
                st.stop()

            st.success("Sent!")
            st.rerun()


    else:
        st.warning("Waiting for opponent...")

        if st.button("🔄 Refresh"):
            st.rerun()