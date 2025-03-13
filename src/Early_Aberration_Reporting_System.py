import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import trim_mean

# Load simulated dengue data from SEIR model
df = pd.read_csv("data/simulated_dengue_data.csv")

# Ensure data is sorted by week
df = df.sort_values(by=["Week"])

# Parameters for EARS
baseline_weeks = 7  # Use the past 5 weeks as baseline for comparison
z_threshold = 0.8  # Further lowered standard deviation threshold for detecting outbreaks

# Compute rolling trimmed mean and standard deviation for historical cases
df["Mean_Past_Cases"] = df["Simulated_Dengue_Cases"].rolling(baseline_weeks).apply(
    lambda x: trim_mean(x, 0.1), raw=True)
df["Std_Past_Cases"] = df["Simulated_Dengue_Cases"].rolling(baseline_weeks).std()

# Ensure standard deviation is not too large or zero
df["Std_Past_Cases"] = df["Std_Past_Cases"].replace(0, np.nan).fillna(df["Std_Past_Cases"].mean())

# Compute EARS Z-score (Standardized Score)
df["Z_Score"] = (df["Simulated_Dengue_Cases"] - df["Mean_Past_Cases"]) / df["Std_Past_Cases"]

# Identify Outbreaks using EARS Threshold
min_case_threshold = 100
df["EARS_Outbreak"] = df.apply(lambda row: "Yes" if row["Z_Score"] > z_threshold and row["Simulated_Dengue_Cases"] > min_case_threshold else "No", axis=1)

# Save results
df.to_csv("ears_outbreaks.csv", index=False)

# Debugging: Print the last 20 weeks of Z-scores to check values
print(df[["Week", "Simulated_Dengue_Cases", "Mean_Past_Cases", "Std_Past_Cases", "Z_Score", "EARS_Outbreak"]].tail(20))

# Plot the detected outbreaks
plt.figure(figsize=(10, 5))
plt.plot(df["Week"], df["Simulated_Dengue_Cases"], label="Simulated Dengue Cases", color="red")
plt.axhline(y=df["Mean_Past_Cases"].mean(), color="black", linestyle="--", label="Historical Mean")
plt.scatter(df[df["EARS_Outbreak"] == "Yes"]["Week"],
            df[df["EARS_Outbreak"] == "Yes"]["Simulated_Dengue_Cases"],
            color="blue", label="EARS Detected Outbreaks", zorder=3)
plt.xlabel("Week")
plt.ylabel("Number of Cases")
plt.title("EARS-Based Dengue Outbreak Detection")
plt.legend()
plt.show()

# Print the number of outbreaks detected
print("Total Outbreaks Detected by EARS:", df["EARS_Outbreak"].value_counts())
print(df[df["EARS_Outbreak"] == "Yes"][["Week", "Simulated_Dengue_Cases", "Z_Score"]])
