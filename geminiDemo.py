# Example (Python)
import google.generativeai as genai
import os
import streamlit as st


def gemini_demo():
# It's best practice to load your API key from an environment variable
# For example: export GOOGLE_API_KEY='YOUR_API_KEY'
    genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))

# Example (Python - Text Generation)
    model = genai.GenerativeModel('gemini-2.0-flash')
    response = model.generate_content("Tell me a story about a brave knight.")
    print(response.text)
    st.text_area(response.text)