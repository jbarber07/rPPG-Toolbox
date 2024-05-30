import matplotlib.pyplot as plt
import pandas as pd

# Data from the experiments
data = {
    "Luminosity Experiment": ["exp_1", "exp_105", "exp_11", "exp_09", "exp_095"],
    "PHYSNET HR_pred_1": [50.9765625, 50.09765625, 51.85546875, 50.9765625, 84.375],
    "PHYSNET HR_pred_2": [51.85546875, 51.85546875, 51.85546875, 51.85546875, 65.0390625],
    "DEEPPHYS HR_pred_1": [100.1953125, 100.1953125, 100.1953125, 100.1953125, 100.1953125],
    "DEEPPHYS HR_pred_2": [59.765625, 59.765625, 59.765625, 59.765625, 59.765625],
}

# Create a DataFrame
df = pd.DataFrame(data)

# Plotting the results
plt.figure(figsize=(14, 8))

# PHYSNET HR Predictions
plt.subplot(2, 1, 1)
plt.plot(df["Luminosity Experiment"], df["PHYSNET HR_pred_1"], marker='o', label='PHYSNET HR_pred_1')
plt.plot(df["Luminosity Experiment"], df["PHYSNET HR_pred_2"], marker='o', label='PHYSNET HR_pred_2')
plt.title("PHYSNET HR Predictions")
plt.xlabel("Luminosity Experiment")
plt.ylabel("HR Prediction")
plt.legend()
plt.grid(True)

# DEEPPHYS HR Predictions
plt.subplot(2, 1, 2)
plt.plot(df["Luminosity Experiment"], df["DEEPPHYS HR_pred_1"], marker='o', label='DEEPPHYS HR_pred_1')
plt.plot(df["Luminosity Experiment"], df["DEEPPHYS HR_pred_2"], marker='o', label='DEEPPHYS HR_pred_2')
plt.title("DEEPPHYS HR Predictions")
plt.xlabel("Luminosity Experiment")
plt.ylabel("HR Prediction")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
