import glob
import os
import sys
from pathlib import Path

sys.path.append("src")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from features import filter_and_clean, load_data

ROOT = Path(__file__).resolve().parent
os.makedirs(ROOT / "outputs/plots", exist_ok=True)

paths = sorted(glob.glob(str(ROOT / "data/raw/monthly_processed_data/*.parquet")))
df = load_data(paths)
df = filter_and_clean(df)
print(f"Bereinigt: {df.shape[0]:,} Zeilen\n")

# Plot 1 – Verspätungsverteilung
fig, ax = plt.subplots(figsize=(10, 4))
sns.histplot(df["delay_in_min"], bins=60, kde=True, ax=ax, color="steelblue")
ax.set_title("Verteilung der Verspätungen (Deutsche Bahn)")
ax.set_xlabel("Verspätung (Minuten)")
plt.tight_layout()
plt.savefig(ROOT / "outputs/plots/delay_distribution.png", dpi=150)
plt.close()
print("saved: delay_distribution.png")

# Plot 2 – Nach Stunde
df["hour"] = pd.to_datetime(df["departure_planned_time"]).dt.hour
fig, ax = plt.subplots(figsize=(10, 4))
df.groupby("hour")["delay_in_min"].mean().plot(
    kind="bar", color="steelblue", ax=ax
)
ax.set_title("Ø Verspätung pro Stunde")
ax.set_xlabel("Stunde")
ax.set_ylabel("Ø Verspätung (min)")
plt.tight_layout()
plt.savefig(ROOT / "outputs/plots/delay_by_hour.png", dpi=150)
plt.close()
print("saved: delay_by_hour.png")

# Plot 3 – Nach Wochentag
days = ["Mo", "Di", "Mi", "Do", "Fr", "Sa", "So"]
df["weekday"] = pd.to_datetime(df["departure_planned_time"]).dt.dayofweek
fig, ax = plt.subplots(figsize=(10, 4))
df.groupby("weekday")["delay_in_min"].mean().rename(
    index=dict(enumerate(days))
).plot(kind="bar", color="steelblue", ax=ax)
ax.set_title("Ø Verspätung pro Wochentag")
ax.set_ylabel("Ø Verspätung (min)")
plt.tight_layout()
plt.savefig(ROOT / "outputs/plots/delay_by_weekday.png", dpi=150)
plt.close()
print("saved: delay_by_weekday.png")

# Plot 4 – Top 10 Bahnhöfe
fig, ax = plt.subplots(figsize=(8, 5))
(
    df.groupby("station_name")["delay_in_min"]
    .mean()
    .sort_values(ascending=False)
    .head(10)
    .sort_values()
    .plot(kind="barh", color="steelblue", ax=ax)
)
ax.set_title("Top 10 Bahnhöfe – Ø Verspätung")
ax.set_xlabel("Ø Verspätung (min)")
plt.tight_layout()
plt.savefig(ROOT / "outputs/plots/top_stations.png", dpi=150)
plt.close()
print("saved: top_stations.png")

print("\nEDA abgeschlossen.")
