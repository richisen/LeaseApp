import streamlit as st
import requests

# FastAPI Backend URL
BACKEND_URL = "http://localhost:8000"  # Change this based on your backend's host/port

# Streamlit UI
st.title("Document Upload & Financial Risk Analysis")

# File Upload
uploaded_file = st.file_uploader("Upload your financial document", type=["pdf", "txt", "csv"])

if uploaded_file is not None:
    # Send file to backend
    files = {"file": uploaded_file}
    try:
        response = requests.post(f"{BACKEND_URL}/upload/", files=files)
        if response.status_code == 200:
            st.success("File uploaded and indexed successfully!")
        else:
            st.error(f"Error: {response.json().get('detail')}")
    except requests.exceptions.RequestException as e:
        st.error(f"File upload failed: {str(e)}")

# User input for question
question = st.text_input("Ask a question related to the document:")

if question:
    try:
        # Ensure that the question is sent in the correct format
        payload = {"question": question}
        st.write(f"Sending question: {question}")  # Log the question being sent
        response = requests.post(f"{BACKEND_URL}/enhanced-response/", json=payload)

        if response.status_code == 200:
            result = response.json()
            if 'error' in result:
                st.error(result['error'])
            else:
                st.write("FinBERT Analysis: ", result.get("finbert_analysis"))
                st.write("Gemini Response: ", result.get("response"))
        else:
            st.error(f"Error: {response.json().get('detail')}")
    except requests.exceptions.RequestException as e:
        st.error(f"Enhanced response failed: {str(e)}")
