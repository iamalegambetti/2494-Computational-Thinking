# Alessandro Gambetti 40755

# Import Dependencies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# Load Data for Exercise 1
df = pd.read_excel('movies.xlsx')

# Scatter Plot
def scatter_plot(df, x, y):
    plt.figure(figsize = (10, 6))
    plt.scatter(df[x].values, df[y].values, marker = '+', c = 'blue', alpha = 0.8)
    plt.xlabel(x, fontsize = 14)
    plt.ylabel(y, fontsize = 14)
    plt.title('Correlation Variables '+ x + ' and ' + y + '.', fontsize = 24)
    plt.show()

def fit_model_2_coefs(df, x, y):
    scale_factor = 1000000 #one milion
    coefs = np.polyfit(df[x].values , df[y].values, 1)
    b0 = coefs[1]
    b1 = coefs[0]
    pred = b1 * df[x].values + b0
    return r2_score(df[y].values, pred)


# Create Data for Exercise 2
data2 = np.array([[1, 4000],
[2, 4800],
[3, 5000],
[20, 7500],
[30, 8000],
[50, 9000],
[80, 9900],
[100, 10200]])


################################# RUNNING HERE #################################

print('Exercise 1:')
print('Plotting First Plot: 7 Days Gross Revenue..')
# First Plot
df7 = df[['7-day Gross', 'Total US Gross']].dropna()
scatter_plot(df7, '7-day Gross', 'Total US Gross')

print('Plotting Second Plot: 14 Days Gross Revenue..')
# Second Plot
df14 = df[['14-day Gross', 'Total US Gross']].dropna()
scatter_plot(df14, '14-day Gross', 'Total US Gross')

print('Plotting Terminated.')

expl1 = 'Both plots present a positive correlation. Thus it is possible to fit a regression between both combinations of variables.'
print(expl1)

print('Fitting Regression one..')
r2_7 = fit_model_2_coefs(df7, '7-day Gross', 'Total US Gross')
print(f'R2 Score for the 7-day Gross is {r2_7}.')

print('Fitting Regression two..')
r2_14 = fit_model_2_coefs(df14, '14-day Gross', 'Total US Gross')
print(f'R2 Score for the 7-day Gross is {r2_14}.')

print('Regressions ended.')

expl2 = 'By looking at the two regression it is possible to infer that there is a significant correlation between the first days sales and the overall sales revenue of a film.'
expl3 = 'Thus, it means that if a film is successfull in the early days, a high amount of revenues are expected overall.'
print(expl2 + '\n' + expl3)
print('End Exercise 1.')

print('\n')

print('Exercise 2:')


print('Fitting Linear Regression..')
# Coefficients
x = data2[:,1]
y = data2[:,0]
linear_coefs = np.polyfit(x, y, 1)
exp_coefs = np.polyfit(x, np.log(y), 1)
mult_coefs = np.polyfit(np.log(x), np.log(y), 1)


# Predictions
print('Making Predictions..')
pred_linear = linear_coefs[1] + linear_coefs[0] * data2[:,1]
pred_exp = np.exp(exp_coefs[1] + exp_coefs[0] * data2[:, 1])
pred_mult = np.exp(mult_coefs[1] + mult_coefs[0] * np.log(data2[:,1]))

print('Computing R-Squared Coefficients..')
print(f'R2 for Linear Model: {r2_score(data2[:, 0], pred_linear)}.')
print(f'R2 for Exponential Model: {r2_score(data2[:, 0], pred_exp)}.')
print(f'R2 for Multiplicative Model: {r2_score(data2[:, 0], pred_mult)}.')

print('The best model is the Multiplicative one because of its highest RSquared.')

print('Predicting Sales Units for a $60000 in Advertising: ')
sales60 = np.exp(mult_coefs[1] + mult_coefs[0] * np.log(6000))
print(f'Predicted Sales Units: {int(round(sales60))}.')

print('END.')
