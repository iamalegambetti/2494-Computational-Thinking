# Import Dependencies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# Load Data
print('Loading Data..')
df = pd.read_excel('peakpower.xlsx')
print(df.head())

# Plot Data
print('Plotting Data..')
plt.figure(figsize = (10, 6))
plt.scatter(df['Daily High Temperature'].values, df['Peak Load'].values)
plt.xlabel('Temperature', fontsize = 14)
plt.ylabel('Peak Power', fontsize = 14)
plt.title('Temperature and Peak Power Relationship', fontsize = 24)
plt.show()

# Fit the Best Model
print('Regression..')
x = df['Daily High Temperature'].values
y = df['Peak Load'].values
coefs = np.polyfit(x, y, 2)

# Get Fitted Values
fitted_values = coefs[2] + coefs[1] * x + coefs[0] * x ** 2
r2 = r2_score(y, fitted_values)

print('By choosing a 2nd Degree Polinomial the R2 Score is: ')
print(f'The R squared coefficient is {round(r2, 2)}.')

print('Predicting Peak Power for a Day with Temperature of 38°:')
pred38 = coefs[2] + coefs[1] * 38 + coefs[0] * 38 ** 2
print(round(pred38, 2), '°.')
