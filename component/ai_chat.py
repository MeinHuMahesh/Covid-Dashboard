import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from pandasai import SmartDataframe
from pandasai_litellm import LiteLLM

load_dotenv()

@st.cache_resource
def get_llm_and_df(df: pd.DataFrame):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        st.error(
            "Gemini API key is missing! Please set the GEMINI_API_KEY environment variable."
        )
        return

    # Initialize your custom LLM wrapper
    llm = LiteLLM(provider="google_ai",model="gemini-2.0-flash",api_key=api_key)

    # Set up the SmartDataframe with this LLM
    smart_df = SmartDataframe(
        df,
        config={
            "llm": llm,
            "enable_code_output": False,
            "use_error_correction_framework": False,
            "enable_cache": False,
            "return_direct_answer": True,
            "verbose": True,
        }
    )
    return smart_df


def ai_chat_interface(df):
    st.markdown("## 🧠 Ask Questions on COVID Data with Gemini AI")
    smart_df=get_llm_and_df(df)
    if "history" not in st.session_state:
        st.session_state.history = []

    for msg in st.session_state.history:
        st.chat_message(msg["role"]).markdown(msg["content"])
    with st.form("ai_chat_form"):
        question = st.text_input("Ask a question:")
        submit = st.form_submit_button("Ask")

        if submit and question:
            st.session_state.history.append({"role":"user","content":question})
            st.chat_message("user").markdown(question)
    # Input box for user question
            with st.spinner("Thinking..."):
                try:
                    response = smart_df.chat(question)
                    if isinstance(response, pd.DataFrame):
                        response = response.head(50)
                        resp_text = response.to_markdown(index=False)
                    else:
                        resp_text = str(response)[:5000]

                except Exception as e:
                    resp_text = f"Error: {e}"
                st.session_state.history.append({"role":"assistant","content":resp_text})
                st.chat_message("assistant").markdown(resp_text)