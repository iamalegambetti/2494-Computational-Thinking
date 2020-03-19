# Import Dependencies
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

# Car 1 DATA
# Set Random Values in String in order to extract using Eval later

# Costs
fc1 = "np.random.choice([6e9, 8e9], p = [0.5, 0.5])" #Fixed Costs and Probs
vc1 = "np.random.choice([4600, 5400], p = [0.5, 0.5])" # Same for Variable Costs

# Number of Sales
sales1 = "np.random.choice([230000, 250000, 270000], p = [0.25, 0.5, 0.25])" # Same for # of sales
error_term1 = "np.random.normal(loc = 0, scale = 20000)" # Error Term


# Common Data
# Price
price = 10000

years = 10
r = 0.1 # Discount Rate


# Car 2 DATA

# Costs
fc2 = "np.random.choice([4e9, 5e9, 16e9], p = [0.25, 0.5, 0.25])" #Fixed Costs and Probs
vc2 = "np.random.choice([2000, 6000], p = [0.5, 0.5])" # Same for Variable Costs

# Number of Sales
sales2 = "np.random.choice([80000, 220000, 390000], p = [0.25, 0.5, 0.25])" # Same for # of sales
error_term2 = "np.random.normal(loc = 0, scale = 30000)" # Error Term


# Create Model
def model(sales, price, vc, fc, error_term, years = 10, r = 0.1):

    # Set Variables for the Previous Year Sales and Sum of Discounted Profits
    sales_previous = None
    discounted_profits = []

    # Iterate through each Year
    for i in range(1, years + 1):

        # Discriminate between first year and following ones
        if i == 1:
            ex_sales = eval(sales)
        else:
            ex_sales = sales_previous

        # Determine Actual Sales
        actual_sales = ex_sales + eval(error_term) # Expcted Sales + Shock
        sales_previous = actual_sales

        # Revenues
        revenue = actual_sales * price

        # Variables Costs
        vc_temp = actual_sales * eval(vc)

        # Profit
        profit = revenue - vc_temp

        # Discounted Profit
        discounted_profit = profit * (1+r)**(-i)
        discounted_profits.append(discounted_profit)

    # Create NPV and deduct Fixed Costs
    npv = sum(discounted_profits)
    fc_disc = eval(fc)
    npv = npv - fc_disc

    return npv

# Create Model For Simulation

def simulate(model, sales, price, vc, fc, error_term, n_trials = 1000):
    """ NPVs Simulations """
    npvs = []
    for _ in range(n_trials):
        npv = model(sales, price, vc, fc, error_term)
        npvs.append(npv)
    return npvs, round(np.mean(npvs), 2), round(np.std(npvs), 2)

def scale(n, thresold = 1000000):
    """Scale n by thresold"""
    return round(n/thresold, 2)


########################## RUNNING FROM HERE ###################################

print('Question A:')
print('Running simulations for both Models..')
# Do Simulations
trials = 1000
first_car = simulate(model, sales1, price, vc1, fc1, error_term1, trials)
second_car = simulate(model, sales2, price, vc2, fc2, error_term2, trials)
print('Simulations terminated.', '\n')

print(f"Car 1 Averaged an NPV of {scale(first_car[1])} and a Standard Deviation of {scale(first_car[2])}. In Million of €.")
print(f"Car 2 Averaged an NPV of {scale(second_car[1])} and a Standard Deviation of {scale(second_car[2])}. In Million of €.")

print('\n')

print('Question B:')
print('Plot Histogram of Both Npvs.')
# Data For Histogram
npvs1 = first_car[0]
npvs2 = second_car[0]

# Histogram
plt.figure(figsize=(10, 6))
plt.hist(npvs1, bins = 50, histtype = 'step', lw = 3, alpha = 0.8, label = 'Car 1')
plt.hist(npvs2, bins = 50, histtype = 'step', lw = 3, color = 'red', alpha = 0.8, label = 'Car 2')
plt.title('Net Present Values Distributions', fontsize = 24)
plt.xlabel('Net Present Values', fontsize = 12)
plt.ylabel('Probabilities', fontsize = 12)
plt.legend(fontsize = 16)
plt.show()

print('\n')

print('Question C:')
print('Assuming Normal Distribution for Each Car we Can Compute the following Probabilities:')
# Discuss: Normal Distribution assumptions for both of the two cars
car1_probs = norm.cdf([0, -1e9], first_car[1], first_car[2])
car2_probs = norm.cdf([0, -1e9], second_car[1], second_car[2])

print(f'Probability that Car 1 NPVs are lower than 0: {round(car1_probs[0], 2)}.')
print(f'Probability that Car 2 NPVs are lower than 0: {round(car2_probs[0], 2)}.')

print(f'Probability that Car 1 NPVs are lower than 1 Billion: {round(car1_probs[1], 2)}.')
print(f'Probability that Car 2 NPVs are lower than 1 Billion: {round(car2_probs[1], 2)}.')

print('\n')

print('Question D:')
print('Building Confidence Interval as Expected NPV (mean) + z * stdev/square root(n_trials):')
# Confidence Intervals
# Formula: Expected NPV (mean) + z * stdev/square root(n_trials)
def conf_int(mean, std, critical_value = 0.05, n = trials):
    """Build Confidence Interval"""
    z = norm.ppf(1-critical_value/2)
    lower_bound = mean - z * std/np.sqrt(n)
    upper_bound = mean + z * std/np.sqrt(n)

    return (round(lower_bound, 2), round(upper_bound, 2))

print(f'Confidence Interval for NPVs of Car 1: {conf_int(scale(first_car[1]), scale(first_car[2]))} (lower_bound, upper_bound).')
print(f'Confidence Interval for NPVs of Car 2: {conf_int(scale(second_car[1]), scale(second_car[2]))} (lower_bound, upper_bound).')
print('Both in Million of €.')
print('\n')
print('Discussion:')

text = "The model for car 2 has (it should be like) higher expected profit, but at the same time higher volatility. Probabilities of losses (and significant losses) are lower for model 1 (lower standard deviation). However, it is a choice of the decision maker which project to undertake according to his risk aversion. It could be suggested to compute the coefficient mean/standard deviation to see how much profit remunerates an unit of standard deviation. "
print(text)
