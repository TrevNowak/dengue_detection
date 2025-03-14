import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import trim_mean

df = pd.read_csv("data/simulated_dengue_data.csv")

df = df.sort_values(by=["Week"])

#Para
baseline_weeks = 7  #baseline( past 7 weeks)
z_threshold = 0.8 

#rolling trimmed mean and s
df["Mean_Past_Cases"] = df["Simulated_Dengue_Cases"].rolling(baseline_weeks).apply(
    lambda x: trim_mean(x, 0.1), raw=True)
df["Std_Past_Cases"] = df["Simulated_Dengue_Cases"].rolling(baseline_weeks).std()

#Ensure std is not too large or zero
df["Std_Past_Cases"] = df["Std_Past_Cases"].replace(0, np.nan).fillna(df["Std_Past_Cases"].mean())

#z-score (Standardized))
df["Z_Score"] = (df["Simulated_Dengue_Cases"] - df["Mean_Past_Cases"]) / df["Std_Past_Cases"]

min_case_threshold = 100
df["EARS_Outbreak"] = df.apply(lambda row: "Yes" if row["Z_Score"] > z_threshold and row["Simulated_Dengue_Cases"] > min_case_threshold else "No", axis=1)

df.to_csv("ears_outbreaks.csv", index=False)

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

print("Total Outbreaks Detected by EARS:", df["EARS_Outbreak"].value_counts())
print(df[df["EARS_Outbreak"] == "Yes"][["Week", "Simulated_Dengue_Cases", "Z_Score"]])
