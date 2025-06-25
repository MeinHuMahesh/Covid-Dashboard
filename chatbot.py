from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import os
from dotenv import load_dotenv
from pandasai import SmartDataframe
from pandasai_litellm import LiteLLM

load_dotenv()
import pandas as pd
def load_data():
    # url = "https://covid.ourworldindata.org/data/owid-covid-data.csv"
    df = pd.read_csv("owid-covid-data.csv")
    df["date"]=pd.to_datetime(df["date"],dayfirst=True)
    df["year"] = df["date"].dt.year
    return df
df = load_data()

columns_needed = [
    "location", "date", "new_cases", "total_cases", "total_deaths",
    "total_vaccinations", "total_tests", "new_deaths", "iso_code", "population"
]
df = df[columns_needed]

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    st.error("Gemini API key is missing! Please set the GEMINI_API_KEY environment variable.")


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

app =FastAPI(title="Chat Backend")

class QuestionRequest(BaseModel):
    question:str

@app.post("/chat")
def chat(request: QuestionRequest):
    try:
        response = smart_df.chat(request.question)
        # PandasAI returns either text or DataFrames
        if isinstance(response, pd.DataFrame):
            return {"answer": response.to_dict(orient="records")}
        return {"answer": str(response)}
    except Exception as e:
        return {"error": str(e)}
@app.get("/ping")
def ping():
    return {"status": "ok"}
@app.get("/count_rows")
def count_rows():
    sql_query = "SELECT COUNT(*) AS num_rows FROM table_b67a3b861e99c6f712f536dae44748d8"
    df = execute_sql_query(sql_query)
    return {"num_rows": int(df.iloc[0, 0])}
