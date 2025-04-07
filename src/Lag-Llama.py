import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from datetime import datetime
from meteostat import Daily

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

#Merge on Week
df_merged = pd.merge(df_cleaned, weekly_weather, on='Week', how='left')
#Add lagged features
df_merged["Rainfall_mm_lag1"] = df_merged["Rainfall_mm"].shift(1)
df_merged["Rainfall_mm_lag2"] = df_merged["Rainfall_mm"].shift(2)
df_merged["Rainfall_mm_lag3"] = df_merged["Rainfall_mm"].shift(3)
df_final = df_merged.dropna()

# === Scaling and Target Setup ===
scaler = MinMaxScaler()
df_numeric = df_final.select_dtypes(include=[np.number])

# Define target column
target_column = "dengue_total"
target_index = df_numeric.columns.get_loc(target_column)

# Scale
scaled = scaler.fit_transform(df_numeric)

# === Create sequences ===
def create_sequences(data, seq_len=52):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len][target_index])
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

X, y = create_sequences(scaled, seq_len=52)
train_size = int(0.8 * len(X))
X_train, y_train = X[:train_size], y[:train_size]
X_test, y_test = X[train_size:], y[train_size:]

# === Transformer Model ===
class SimpleTransformer(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        x = self.input_proj(x)
        x = x.permute(1, 0, 2)
        out = self.transformer(x)
        return self.fc(out[-1]).squeeze()

# === Train ===
model = SimpleTransformer(input_dim=X.shape[2])
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.L1Loss()  # Use MAE loss instead of MSE

for epoch in range(50):
    model.train()
    optimizer.zero_grad()
    out = model(X_train)
    loss = criterion(out, y_train)
    loss.backward()
    optimizer.step()
    if epoch % 5 == 0:
        print(f"Epoch {epoch+1}: Train Loss = {loss.item():.4f}")

# === Evaluate ===
model.eval()
with torch.no_grad():
    pred = model(X_test).numpy()
    true = y_test.numpy()

features_count = scaled.shape[1]

# Inverse scale
pred_scaled = np.zeros((len(pred), features_count))
pred_scaled[:, target_index] = pred.flatten()
pred_cases = scaler.inverse_transform(pred_scaled)[:, target_index]

true_scaled = np.zeros((len(true), features_count))
true_scaled[:, target_index] = true.flatten()
actual_cases = scaler.inverse_transform(true_scaled)[:, target_index]

# Metrics
mae = mean_absolute_error(actual_cases, pred_cases)
rmse = np.sqrt(mean_squared_error(actual_cases, pred_cases))
print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}")

# Outbreak Detection
residuals = actual_cases - pred_cases
threshold = np.percentile(residuals, 95)
anomalies = residuals > threshold

# Plot
weeks = df_final["Week"].iloc[-len(actual_cases):]

plt.figure(figsize=(14, 5))
plt.plot(weeks, actual_cases, label="Actual Cases", linewidth=2)
plt.plot(weeks, pred_cases, label="Predicted Cases", linewidth=2)
plt.scatter(weeks[anomalies], actual_cases[anomalies], color='red', label="Detected Outbreaks", zorder=5)
plt.title("Predicted vs Actual Dengue Cases (Test Set)")
plt.xlabel("Week")
plt.ylabel("Dengue Case Count")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
