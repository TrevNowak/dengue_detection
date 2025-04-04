import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score, root_mean_squared_error, mean_absolute_error
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

outbreak_threshold = np.percentile(df_final["dengue_total"], 90)
df_final["Outbreak_Status"] = df_final["dengue_total"].apply(lambda x: "Yes" if x > outbreak_threshold else "No")

#Define severe outbreak to be top 5% cases AND high rainfall (80)
severe_threshold = np.percentile(df_final["dengue_total"], 95)
df_final["Severe_Outbreak_Status"] = df_final.apply(
    lambda row: "Yes" if row["dengue_total"] > severe_threshold and row["Rainfall_mm"] > 80 else "No", axis=1
)
df_final["Outbreak_Binary"] = df_final["Outbreak_Status"].map({"Yes": 1, "No": 0})
df_final["Severe_Outbreak_Binary"] = df_final["Severe_Outbreak_Status"].map({"Yes": 1, "No": 0})

y_true = df_final["Severe_Outbreak_Binary"]
y_pred = df_final["Outbreak_Binary"]

conf_matrix = confusion_matrix(y_true, y_pred)
classification_rep = classification_report(y_true, y_pred)
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, zero_division=0)
recall = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)

print("Confusion Matrix:\n", conf_matrix)
print("\nClassification Report:\n", classification_rep)
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")

