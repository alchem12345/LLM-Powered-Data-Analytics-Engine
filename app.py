import streamlit as st
from agent_logic import get_agent
from langchain_core.messages import AIMessage,HumanMessage

st.set_page_config(page_title="VeriScholar AI", layout="wide")
st.title("VerischolarAI : student analytics")

# start the session states / frames in a still frame website
if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_fig" not in st.session_state:
    st.session_state.last_fig = None

st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stStatus { border-radius: 10px; border: 1px solid #ddd; }
    </style>
    """, unsafe_allow_html=True)



st.title("🎓 VeriScholar AI")
st.caption("Advanced Student Performance Analytics Engine")

for message in st.session_state.messages:
    role = "user" if isinstance(message, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(message.content)

# mian loop for the chat
if prompt := st.chat_input("Ex: What is the correlation between study hours and final grades?"):
    st.session_state.messages.append(HumanMessage(content=prompt))
    with st.chat_message("user"):
        st.markdown(prompt)
        with st.chat_message("assistant"):
            #using st.status the show the process of thinking to the user
            with st.status("🔍 Analyzing Student Data...", expanded=True) as status:
                try:
                    # getting / creting the agent into this code
                    executor = get_agent()

                    response = executor.invoke({
                        "input": prompt,
                        "chat_history": st.session_state.messages[:-1]
                    })

                    output = response["output"]
                    status.update(label="Analysis Complete!", state="complete", expanded=False)

                except Exception as e:
                    output = f"⚠️ An error occurred during analysis: {str(e)}"
                    status.update(label="Analysis Failed", state="error")

            # Dispaling the final ansswrr
            st.markdown(output)
            st.session_state.messages.append(AIMessage(content=output))

            if st.session_state.last_fig:
                st.rerun()  # Refresh to show chart in sidebar



