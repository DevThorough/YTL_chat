import streamlit as st
import subprocess
import sys

index_name = st.text_input("Enter index name")
folder_name = st.text_input("Enter folder name")

if st.button("Run Script"):
    if index_name and folder_name:
        result = subprocess.run([sys.executable, "ragLang.py", index_name, folder_name], capture_output=True, text=True)
        st.text("Output:")
        st.code(result.stdout)
        if result.stderr:
            st.error(result.stderr)
    else:
        st.error("Please provide both index name and folder name")
