import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.api as sm

df = pd.read_csv("data/simulated_dengue_data.csv")

X = df[[
    "Rainfall_mm_lag1", "Rainfall_mm_lag2", "Rainfall_mm_lag3",
    "Temperature_C", "Humidity_%", "Fogging", "Larvicide_Use", "Seasonality",
    "Population_Density", "Rainfall_Temperature_Interaction",
    "Fogging_Cases_Interaction", "Larvicide_Cases_Interaction"
]]
X = sm.add_constant(X) 
y = df["Simulated_Dengue_Cases_Noised"]

#Drop NaN values caused by lagging
df_cleaned = df.dropna()

X = df_cleaned[[ 
    "Rainfall_mm_lag1", "Rainfall_mm_lag2", "Rainfall_mm_lag3",
    "Temperature_C", "Humidity_%", "Fogging", "Larvicide_Use", "Seasonality",
    "Population_Density", "Rainfall_Temperature_Interaction",
    "Fogging_Cases_Interaction", "Larvicide_Cases_Interaction"
]]

X = sm.add_constant(X)
y = df_cleaned["Simulated_Dengue_Cases_Noised"] 

model = sm.GLM(y, X, family=sm.families.NegativeBinomial())
result = model.fit()

print(result.summary())
