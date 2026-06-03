import streamlit as st
import requests

st.set_page_config(
   page_title="Home Page",
   page_icon="🏠"
)

base_url = "http://127.0.0.1:8000"

st.title("First LLM Application")

uploaded_files = st.file_uploader("Choose Your .pdf file", type="pdf", accept_multiple_files=True)


if st.button("upload"):

    if uploaded_files:
      files_payload = [
         ("files", (file.name, file.getvalue(), "application/pdf"))
         for file in uploaded_files
      ]

      try:
         reponse = requests.post(url=f"{base_url}/upload", files=files_payload)
         st.switch_page("pages/chat_with_llm.py")
      except requests.exceptions.ConnectionError:
         st.write("Could Not Connect to Backend Server Try Again later")

    else:
        st.write("Please Select the file before uploading")

      
         

    




