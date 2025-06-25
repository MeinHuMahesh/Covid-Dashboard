from component.data_loader import load_data
from component.ai_chat import ai_chat_interface

df = load_data()

columns_needed = [
    "location", "date", "new_cases", "total_cases", "total_deaths",
    "total_vaccinations", "total_tests", "new_deaths","year",
    "gdp_per_capita", "human_development_index", "iso_code", "population"
]
df = df[columns_needed]

ai_chat_interface(df)