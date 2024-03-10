import streamlit as st
from time import sleep

st.title('ShadowCon\'s RAG Using Llama2 and Streamlit')

st.write("Choose a PDF File")
uploaded_file = st.file_uploader("PDF File",
                 type='pdf',
                 accept_multiple_files=False,
                 on_change=None,)

if uploaded_file is not None:
    st.write("File Uploaded")
    sleep(5)
    st.write("Processing the file")

    # Processing the file
    st.write(f"File Processed: {uploaded_file.name}")