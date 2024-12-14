# Streamlit app UI (all backend code is in functions.py)
""" 
This is a Streamlit app that allows users to upload a PDF document and generate a structured response.
the outline for this app is in this file, it uses the calculations from the functions.py file and displays the results in a Streamlit app
"""

import streamlit as st  # streamlit
from functions import * # import all functions from functions.py
import base64 # for encoding and decoding

# Initialize the API key in session state if it doesn't exist
if 'api_key' not in st.session_state:
    st.session_state.api_key = ''

def display_pdf(uploaded_file): # after file is uploaded display it
    # Read file as bytes:
    bytes_data = uploaded_file.getvalue()
    # Convert to Base64
    base64_pdf = base64.b64encode(bytes_data).decode('utf-8')
    # Embed PDF in HTML and display on screen
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>'
    # Display file in Streamlit
    st.markdown(pdf_display, unsafe_allow_html=True)

# Main app page
def load_streamlit_page(): 
    st.set_page_config(layout="wide", page_title="LLM Tool")
    # Design page layout with 2 columns: File uploader on the left, and other interactions on the right.
    col1, col2 = st.columns([0.5, 0.5], gap="large")
    with col1: # left column
        # stuff for left column on page
        st.header("Input your OpenAI API key")
        st.text_input('OpenAI API key', type='password', key='api_key',
                    label_visibility="collapsed", disabled=False)
        st.header("Upload file")
        uploaded_file = st.file_uploader("Please upload your PDF document:", type= "pdf")

    return col1, col2, uploaded_file # export col1 and col2 to be used in creating the app


# Make a streamlit page using the load function above
col1, col2, uploaded_file = load_streamlit_page()  # call the load function and load the vvalues into the variables

# if we have a file Process the input from it 
if uploaded_file is not None:
    with col2: # right column
        display_pdf(uploaded_file) # display the uploaded file
        
    # Load in the documents
    documents = get_pdf_text(uploaded_file) # from our functions.py we get the pdf text using pypdfloader
    # creating a vector store to store the text emmbeddings
    st.session_state.vector_store = create_vectorstore_from_texts(documents, 
                                                                  api_key=st.session_state.api_key,
                                                                  file_name=uploaded_file.name)
    st.write("Input Processed") # finish statement

# Generate answer using the vector store
with col1: # left column
    if st.button("Generate table"): # if the button is clicked
        with st.spinner("Generating answer"): # display a spinner that will wait until the answer is generated
            # answer if now generated and ready
            
            # Load into the variable answer vectorstore, query and api key into the query document function which dose all the rag chain processing 
            answer = query_document(vectorstore = st.session_state.vector_store, 
                                    query = "Give me the title, summary, publication date, and authors of the research paper.",
                                    api_key = st.session_state.api_key)
                            
            # display the answer
            placeholder = st.empty() # create an empty placeholder to display the answer
            placeholder = st.write(answer) # display the answer in the placeholder
            
# * done