"""
Generates EDA charts from WESAD recordings and Open-Meteo API model data:
1. Recorded WESAD Subject S2 physiological signals
2. Ljubljana environmental data from Open-Meteo APIs (not station measurements)
Author: Kartikeya Singh (Member 1)
"""
import pandas as pd
import matplotlib.pyplot as plt

# 1. Load genuine WESAD recording
df_wesad = pd.read_csv("data/processed/wesad_s2_real.csv")

# 2. Load Ljubljana Open-Meteo API model data
df_ljub = pd.read_csv("data/processed/ljubljana_real.csv")

# Plot 1: Real WESAD Subject S2 Vitals (Empatica E4)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
ax1.plot(df_wesad.index, df_wesad["heart_rate_bpm"], color="#2980b9", label="Recorded HR (bpm)")
ax1.set_ylabel("Heart Rate (bpm)")
ax1.set_title("WESAD Subject S2: Recorded Physiological Signals (Empatica E4)", fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.legend(loc="upper right")

ax2.plot(df_wesad.index, df_wesad["eda_microsiemens"], color="#8e44ad", label="Recorded EDA (microSiemens)")
ax2.set_xlabel("Recording Timeline (Seconds)")
ax2.set_ylabel("EDA (microSiemens)")
ax2.grid(True, alpha=0.3)
ax2.legend(loc="upper right")
plt.tight_layout()
plt.savefig("data/eda/plots/genuine_wesad_s2_vitals.png", dpi=200)
plt.close()

# Plot 2: Ljubljana Open-Meteo Environmental Data
fig, ax1 = plt.subplots(figsize=(10, 5))
ax2 = ax1.twinx()

ax1.plot(range(len(df_ljub)), df_ljub["pm25_ug_m3"], color="#e74c3c", label="PM2.5 (ug/m3)")
ax2.plot(range(len(df_ljub)), df_ljub["ambient_temperature_c"], color="#f39c12", linestyle="--", label="Temperature (C)")

ax1.set_xlabel("Timeline (API Hours)")
ax1.set_ylabel("PM2.5 (ug/m3)", color="#e74c3c")
ax2.set_ylabel("Temperature (C)", color="#f39c12")
plt.title("Ljubljana: Open-Meteo Air Quality & Weather Model Data", fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("data/eda/plots/genuine_ljubljana_environmental.png", dpi=200)
plt.close()

print("[SUCCESS] Generated EDA charts from WESAD recordings and Open-Meteo API data.")
