import streamlit as st
import requests

st.set_page_config(
    page_title="chatting",
    page_icon="👋",
)

base_url = "http://127.0.0.1:8000"

if "messages" not in st.session_state:
    st.session_state.messages = []

prompt = st.chat_input("Enter You Question")

for message in st.session_state.messages:
    if message is not None:
        with st.chat_message(message["role"]):
            st.markdown(message["message"])

if prompt is not None:
    with st.chat_message("human"):
        st.markdown(prompt)

    st.session_state.messages.append(
        {
        "role": "human", 
        "message": prompt
        }
    )

    try:
        payload = {
            "question": prompt,
            "top_k" : 3
        }
        response = requests.post(url=f"{base_url}/question", json=payload)

        with st.chat_message("ai"):
            response_dict = response.json()
            llm_output = response_dict["LLM Output"]
            st.markdown(llm_output)

        st.session_state.messages.append(
            {
                "role": "ai",
                "message": llm_output
            }
        )
    except requests.exceptions.ConnectionError:
        st.error("Cannot Connect to Backend server")    
        




        