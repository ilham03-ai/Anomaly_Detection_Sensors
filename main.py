import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest

# Generate synthetic sensor data
np.random.seed(42)
data_normal = np.random.normal(loc=50, scale=5, size=100).reshape(-1, 1)
data_anomalies = np.random.normal(loc=80, scale=3, size=5).reshape(-1, 1)
data = np.vstack((data_normal, data_anomalies))

df = pd.DataFrame(data, columns=["Sensor_Value"])

# Train Isolation Forest model
model = IsolationForest(contamination=0.05, random_state=42)
df["Anomaly"] = model.fit_predict(df[["Sensor_Value"]])

# Plot results
plt.figure(figsize=(10, 5))
sns.scatterplot(x=df.index, y=df["Sensor_Value"], hue=df["Anomaly"], palette={1: "blue", -1: "red"})
plt.title("Anomaly Detection in Sensor Data")
plt.xlabel("Time")
plt.ylabel("Sensor Value")
plt.legend(["Normal", "Anomaly"])
plt.show()