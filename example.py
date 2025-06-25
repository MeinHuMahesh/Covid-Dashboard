import pandas as pd
def load_data():
    # url = "https://covid.ourworldindata.org/data/owid-covid-data.csv"
    df = pd.read_csv("owid-covid-data.csv")
    df["date"]=pd.to_datetime(df["date"],dayfirst=True)
    df["year"] = df["date"].dt.year
    return df

df =load_data()
columns_needed = [
    "location", "date", "new_cases", "total_cases", "total_deaths",
    "total_vaccinations", "total_tests", "new_deaths", "iso_code", "population"
]
df = df[columns_needed]
total_mem = df.memory_usage(deep=True).sum() / (1024 ** 2)  # in MB
print(f"Total DataFrame memory usage: {total_mem:.2f} MB")

# Check per-column usage
print(df.memory_usage(deep=True))