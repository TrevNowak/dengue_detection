import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import trim_mean
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from meteostat import Daily, Stations
from datetime import datetime

df = pd.read_csv("data/dengue_data_sg.csv")
filtered_df = df[df['T_res'].str.contains('Week')]
cols_to_drop = [
    'adm_0_name',
    'adm_1_name',
    'adm_2_name',
    'full_name',
    'ISO_A0',
    'FAO_GAUL_code',
    'RNE_iso_code',
    'IBGE_code',
    'UUID'
]
df_cleaned = filtered_df.drop(columns=cols_to_drop)
start = datetime(2001, 12, 30)
end = datetime(2022, 12, 24)

station_id = "48698"
data = Daily(station_id, start, end)
daily = data.fetch()

daily = daily.reset_index()
daily['time'] = pd.to_datetime(daily['time'])
daily.set_index('time', inplace=True)

weekly_weather = daily.resample('W-SUN').agg({
    'tavg': 'mean',
    'prcp': 'sum'
}).reset_index()

weekly_weather.columns = ['Week', 'Temperature_C', 'Rainfall_mm']

weekly_weather["Week_Num"] = weekly_weather["Week"].dt.isocalendar().week
weekly_weather["Seasonality"] = np.sin(2 * np.pi * weekly_weather["Week_Num"] / 52)

df_cleaned['Week'] = pd.to_datetime(df_cleaned['calendar_start_date'])
weekly_weather['Week'] = pd.to_datetime(weekly_weather['Week'])

df_merged = pd.merge(df_cleaned, weekly_weather, on='Week', how='left')
df_merged["Rainfall_mm_lag1"] = df_merged["Rainfall_mm"].shift(1)
df_merged["Rainfall_mm_lag2"] = df_merged["Rainfall_mm"].shift(2)
df_merged["Rainfall_mm_lag3"] = df_merged["Rainfall_mm"].shift(3)

df_final = df_merged.dropna()
df_final = df_final.sort_values(by="Week")


baseline_weeks = 7   # how many past weeks to consider
z_threshold = 0.8    # standard deviation threshold
min_case_threshold = 100  # optional: avoid triggering on low base levels

# --- Rolling Trimmed Mean and Std Dev ---
df_final["Mean_Past_Cases"] = df_final["dengue_total"].rolling(baseline_weeks).apply(
    lambda x: trim_mean(x, 0.1), raw=True)
df_final["Std_Past_Cases"] = df_final["dengue_total"].rolling(baseline_weeks).std()

# --- Handle zero/std edge cases ---
mean_std = df_final["Std_Past_Cases"].mean()
df_final["Std_Past_Cases"] = df_final["Std_Past_Cases"].replace(0, np.nan).fillna(mean_std)

# --- Z-Score Calculation ---
df_final["Z_Score"] = (df_final["dengue_total"] - df_final["Mean_Past_Cases"]) / df_final["Std_Past_Cases"]

# --- Flag Outbreaks Based on Z-score + Cases ---
df_final["EARS_Outbreak"] = df_final.apply(
    lambda row: "Yes" if row["Z_Score"] > z_threshold and row["dengue_total"] > min_case_threshold else "No", axis=1)

plt.figure(figsize=(12, 5))
plt.plot(df_final["Week"], df_final["dengue_total"], label="Actual Dengue Cases", color="red")
plt.axhline(y=df_final["Mean_Past_Cases"].mean(), color="black", linestyle="--", label="Baseline Mean")
plt.scatter(df_final[df_final["EARS_Outbreak"] == "Yes"]["Week"],
            df_final[df_final["EARS_Outbreak"] == "Yes"]["dengue_total"],
            color="blue", label="EARS Outbreaks", zorder=3, s=40)
plt.xlabel("Week")
plt.ylabel("Dengue Cases")
plt.title("EARS-Based Dengue Outbreak Detection (Real Data)")
plt.legend()
plt.tight_layout()
plt.show()

print("✅ Total Outbreaks Detected by EARS:", df_final["EARS_Outbreak"].value_counts())
print(df_final[df_final["EARS_Outbreak"] == "Yes"][["Week", "dengue_total", "Z_Score"]])