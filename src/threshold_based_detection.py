import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score

df = pd.read_csv("data/simulated_dengue_data.csv")

outbreak_threshold = np.percentile(df["Simulated_Dengue_Cases"], 90)
df["Outbreak_Status"] = df["Simulated_Dengue_Cases"].apply(lambda x: "Yes" if x > outbreak_threshold else "No")

severe_outbreak_threshold = np.percentile(df["Simulated_Dengue_Cases"], 95)
df["Severe_Outbreak_Status"] = df.apply(
    lambda row: "Yes" if row["Simulated_Dengue_Cases"] > severe_outbreak_threshold and row["Rainfall_mm"] > 250 else "No", axis=1)

df["Outbreak_Binary"] = df["Outbreak_Status"].apply(lambda x: 1 if x == "Yes" else 0)
df["Severe_Outbreak_Binary"] = df["Severe_Outbreak_Status"].apply(lambda x: 1 if x == "Yes" else 0)

y_true = df["Severe_Outbreak_Binary"]
y_pred = df["Outbreak_Binary"]

conf_matrix = confusion_matrix(y_true, y_pred)
classification_rep = classification_report(y_true, y_pred)
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print("Confusion Matrix:\n", conf_matrix)
print("\nClassification Report:\n", classification_rep)
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

df.to_csv("threshold_based_outbreaks.csv", index=False)

plt.figure(figsize=(10, 5))
plt.plot(df["Week"], df["Simulated_Dengue_Cases"], label="Simulated Dengue Cases", color="red")
plt.axhline(y=outbreak_threshold, color="black", linestyle="--", label="Outbreak Threshold")
plt.scatter(df[df["Outbreak_Status"] == "Yes"]["Week"], 
            df[df["Outbreak_Status"] == "Yes"]["Simulated_Dengue_Cases"], 
            color="blue", label="Outbreak Detected", zorder=3)
plt.xlabel("Week")
plt.ylabel("Number of Cases")
plt.title("Threshold-Based Dengue Outbreak Detection")
plt.legend()
plt.show()
