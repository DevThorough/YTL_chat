import streamlit as st
import subprocess
import sys
import re


channel_handle = st.text_input("Enter YouTube channel handle(begins with @): ")

if channel_handle:
    channel_lookup = subprocess.run([sys.executable, "youtube_lore.py", channel_handle, "0"], capture_output=True, text=True)
    st.code(channel_lookup.stdout)

    choice = st.selectbox("Is this the correct channel?",(" ","Yes", "No"))
    if choice == "No":
        st.write("Please try another search.") 
    elif choice == "Yes":
        st.write("Please wait. Currently downloading captions.") 
        download_output = subprocess.run([sys.executable, "youtube_lore.py", channel_handle, "1"], capture_output=True, text=True)
        st.code(download_output.stdout)
        
        pattern = r"Captions saved to ([^/]+)"
        match = re.search(pattern, download_output.stdout)
        
        if match:
            folder_name = match.group(1).strip()
            st.success(f"Success! Folder name: {folder_name}")
            
            ai_output = subprocess.run([sys.executable, "ragLang.py", "new-test", folder_name], capture_output=True, text=True)
            st.code(ai_output.stdout)
            
        else:
            st.warning("Error: Folder name not found in the output.")
        
    """
    TO DO LIST:
    - Edit ragLang to have a continuous query output loop
    - Allow user to enter their own query
    
    Additional Options:
    - Already existing folder and index
    - Verbose toggle
    - Display thumbnail and terminal output in a more appealing way
    """

        
        
        

# if st.button("Run Script"):
#     if index_name and folder_name:
#         result = subprocess.run([sys.executable, "ragLang.py", index_name, folder_name], capture_output=True, text=True)
#         st.text("Output:")
#         st.code(result.stdout)
#         if result.stderr:
#             st.error(result.stderr)
#     else:
#         st.error("Please provide both index name and folder name")
