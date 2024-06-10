import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Organizing the new data into a DataFrame, including PhysNet_Light
new_data = {
    'Method': [
        'PHYSNET', 'PHYSNET', 'PHYSNET', 'PHYSNET', 'PHYSNET',
        'PHYSNET_Light', 'PHYSNET_Light', 'PHYSNET_Light', 'PHYSNET_Light', 'PHYSNET_Light',
        'DEEPPHYS', 'DEEPPHYS', 'DEEPPHYS', 'DEEPPHYS', 'DEEPPHYS',
        'ICA', 'ICA', 'ICA', 'ICA', 'ICA',
        'POS', 'POS', 'POS', 'POS', 'POS',
        'CHROM', 'CHROM', 'CHROM', 'CHROM', 'CHROM',
        'GREEN', 'GREEN', 'GREEN', 'GREEN', 'GREEN',
        'LGI', 'LGI', 'LGI', 'LGI', 'LGI',
        'PBV', 'PBV', 'PBV', 'PBV', 'PBV'
    ],
    'Luminosity Experiment': [
        'exp_1', 'exp_105', 'exp_11', 'exp_09', 'exp_095',
        'exp_1', 'exp_105', 'exp_11', 'exp_09', 'exp_095',
        'exp_1', 'exp_105', 'exp_11', 'exp_09', 'exp_095',
        'exp_1', 'exp_105', 'exp_11', 'exp_09', 'exp_095',
        'exp_1', 'exp_105', 'exp_11', 'exp_09', 'exp_095',
        'exp_1', 'exp_105', 'exp_11', 'exp_09', 'exp_095',
        'exp_1', 'exp_105', 'exp_11', 'exp_09', 'exp_095',
        'exp_1', 'exp_105', 'exp_11', 'exp_09', 'exp_095',
        'exp_1', 'exp_105', 'exp_11', 'exp_09', 'exp_095'
    ],
    'Subject 1': [
        50.9765625, 50.09765625, 51.85546875, 50.9765625, 84.375,
        68.5546875, 45.7031, 66.796875, 89.6484375, 82.6171,
        100.1953125, 100.1953125, 100.1953125, 100.1953125, 100.1953125,
        72.94921875, 92.28515625, 59.765625, 65.0390625, 63.28125,
        77.34375, 65.0390625, 59.765625, 71.19140625, 78.22265625,
        69.43359375, 65.0390625, 59.765625, 71.19140625, 79.98046875,
        92.28515625, 60.64453125, 92.28515625, 69.43359375, 69.43359375,
        77.34375, 65.0390625, 59.765625, 65.0390625, 78.22265625,
        60.64453125, 59.765625, 59.765625, 65.0390625, 63.28125
    ],
    'Subject 2': [
        51.85546875, 51.85546875, 51.85546875, 51.85546875, 65.0390625,
        np.nan, np.nan, np.nan, np.nan, np.nan,
        59.765625, 59.765625, 59.765625, 59.765625, 59.765625,
        59.765625, 79.98046875, 90.52734375, 65.91796875, 67.67578125,
        86.1328125, 86.1328125, 90.52734375, 72.94921875, 67.67578125,
        71.19140625, 79.98046875, 72.94921875, 84.375, 71.19140625,
        72.0703125, 79.98046875, 72.0703125, 72.0703125, 72.0703125,
        71.19140625, 71.19140625, 90.52734375, 71.1914062, 67.67578125,
        86.1328125, 81.73828125, 90.52734375, 71.19140625, 67.67578125
    ]
}

new_df = pd.DataFrame(new_data)

# Calculate the Mean Absolute Error (MAE) for each method and experiment
new_df['MAE'] = np.abs(new_df['Subject 1'] - new_df['Subject 2'])
new_df['MAE'] = new_df['MAE'].fillna(0)

# Mapping the luminosity experiments to more readable labels
luminosity_labels = {
    'exp_1': 'Reference',
    'exp_105': '+5%',
    'exp_11': '+10%',
    'exp_09': '-10%',
    'exp_095': '-5%'
}

# Updating the DataFrame with the new labels
new_df['Luminosity Experiment'] = new_df['Luminosity Experiment'].map(luminosity_labels)

# Calculate the MAE for each method and experiment relative to the reference value
reference_values = new_df[new_df['Luminosity Experiment'] == 'Reference'].set_index('Method')['MAE'].to_dict()

# Adjusting the MAE to show the variation from the reference value
new_df['MAE Variation'] = new_df.apply(lambda row: row['MAE'] - reference_values[row['Method']], axis=1)

# Sorting the DataFrame to have the correct order on the x-axis
order = ['-10%', '-5%', 'Reference', '+5%', '+10%']
new_df['Luminosity Experiment'] = pd.Categorical(new_df['Luminosity Experiment'], categories=order, ordered=True)
new_df = new_df.sort_values('Luminosity Experiment')

# Create the plot with updated labels and MAE variations
plt.figure(figsize=(14, 8))
for method in new_df['Method'].unique():
    subset = new_df[new_df['Method'] == method]
    plt.plot(subset['Luminosity Experiment'], subset['MAE Variation'], marker='o', label=method)

plt.xlabel('Luminosity Experiment')
plt.ylabel('MAE Variation (relative to Reference)')
plt.title('MAE Variation of Heart Rate Predictions for Each Method Under Different Luminosity Experiments')
plt.xticks(order)
plt.legend(title='Method')
plt.grid(True)
plt.show()
