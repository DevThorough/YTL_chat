import streamlit as st
import subprocess
import sys


channel_handle = st.text_input("Enter YouTube channel handle(begins with @): ")

if channel_handle:
    channel_lookup = subprocess.run([sys.executable, "youtube_lore.py", channel_handle, "0"], capture_output=True, text=True)
    st.code(channel_lookup.stdout)

    choice = st.selectbox("Is this the correct channel?",(" ","Yes", "No"))
    if choice == "Yes":
        final_result = subprocess.run([sys.executable, "youtube_lore.py", channel_handle, "1"], capture_output=True, text=True)
        st.code(final_result.stdout)
    elif choice == "No":
        st.write("Please try another search.")

# if st.button("Run Script"):
#     if index_name and folder_name:
#         result = subprocess.run([sys.executable, "ragLang.py", index_name, folder_name], capture_output=True, text=True)
#         st.text("Output:")
#         st.code(result.stdout)
#         if result.stderr:
#             st.error(result.stderr)
#     else:
#         st.error("Please provide both index name and folder name")
