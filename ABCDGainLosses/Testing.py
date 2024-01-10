import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# importing the dataset
dataset = pd.read_csv('./TESTING.csv')

# clean the dataset
# remove the first 6 rows
dataset = dataset.iloc[6:]
# remove the last 2 rows
dataset = dataset.iloc[:-2]
# remove the \' from values in the first column
dataset.iloc[:, 0] = dataset.iloc[:, 0].str.replace('\'', '')
# convert the first column to float
dataset.iloc[:, 0] = dataset.iloc[:, 0].astype(float)
dataset.iloc[:, 1] = dataset.iloc[:, 1].astype(float)

# draw a line plot of the data
plt.figure(figsize=(15, 10))
plt.plot(dataset.iloc[:, 0], dataset.iloc[:, 1], color='red')
plt.title('Testing Data')
plt.xlabel('Frequency')
plt.ylabel('Gain')
plt.show()