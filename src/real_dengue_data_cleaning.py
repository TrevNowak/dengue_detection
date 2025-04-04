import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.api as sm
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

# --- Step 2: Use Changi Airport Station (48698) ---
station_id = "48698"
data = Daily(station_id, start, end)
daily = data.fetch()

# --- Step 3: Convert and Resample to Weekly ---
daily = daily.reset_index()
daily['time'] = pd.to_datetime(daily['time'])
daily.set_index('time', inplace=True)

# Resample to weekly: mean temperature, sum rainfall (Sunday-ending weeks)
weekly_weather = daily.resample('W-SUN').agg({
    'tavg': 'mean',
    'prcp': 'sum'
}).reset_index()

# Rename columns
weekly_weather.columns = ['Week', 'Temperature_C', 'Rainfall_mm']

# --- Step 4: Add Seasonality Feature ---
weekly_weather["Week_Num"] = weekly_weather["Week"].dt.isocalendar().week
weekly_weather["Seasonality"] = np.sin(2 * np.pi * weekly_weather["Week_Num"] / 52)

# Step 1: Ensure date types are correct
df_cleaned['Week'] = pd.to_datetime(df_cleaned['calendar_start_date'])
weekly_weather['Week'] = pd.to_datetime(weekly_weather['Week'])

# Step 2: Merge on Week
df_merged = pd.merge(df_cleaned, weekly_weather, on='Week', how='left')

# Step 3: Drop duplicate or unnecessary columns if needed
# Example: df_merged.drop(columns=['calendar_start_date', 'calendar_end_date'], inplace=True)

# Step 4: Preview
df_merged["Rainfall_mm_lag1"] = df_merged["Rainfall_mm"].shift(1)
df_merged["Rainfall_mm_lag2"] = df_merged["Rainfall_mm"].shift(2)
df_merged["Rainfall_mm_lag3"] = df_merged["Rainfall_mm"].shift(3)

print(df_merged.head())
print(df_merged.columns.tolist())
