import pandas as pd
import numpy as np
import statsmodels.api as sm
from meteostat import Daily, Stations
from datetime import datetime
from sklearn.metrics import root_mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt


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
train_df = df_final[df_final["Week"] < "2019-01-01"]
test_df = df_final[df_final["Week"] >= "2019-01-01"]

features = [
    "Rainfall_mm_lag1", "Rainfall_mm_lag2", "Rainfall_mm_lag3",
    "Temperature_C", "Seasonality"
]

X_train = sm.add_constant(train_df[features].astype(float))
y_train = train_df["dengue_total"].astype(float)

X_test = sm.add_constant(test_df[features].astype(float))
y_test = test_df["dengue_total"].astype(float)

#Train on training set
model = sm.GLM(y_train, X_train, family=sm.families.NegativeBinomial())
result = model.fit()
test_df["Predicted_Cases"] = result.predict(X_test)

rmse = root_mean_squared_error(y_test, test_df["Predicted_Cases"])
mae = mean_absolute_error(y_test, test_df["Predicted_Cases"])

print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")

plt.figure(figsize=(12, 5))
plt.plot(test_df["Week"], y_test, label="Actual", color="red")
plt.plot(test_df["Week"], test_df["Predicted_Cases"], label="Predicted", color="blue")
plt.title("Test Set: Actual vs Predicted Dengue Cases")
plt.xlabel("Week")
plt.ylabel("Cases")
plt.legend()
plt.tight_layout()
plt.show()
