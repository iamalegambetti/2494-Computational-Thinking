# Import Dependencies
import numpy as np
from scipy.stats import norm
import pandas as pd

# Random Sample Method
def sample(df, sample_size = 50):
    N = len(df)
    # Create an array of Random Indexes
    indexes = list(np.random.randint(0, N-1, size = sample_size))
    sample_ = df.iloc[indexes] # Index the dataframe
    return sample_


####### Work Here ####

# Load Data
df = pd.read_excel('Auditing.xlsx')

# Population Mean
pop_mean = round(df['Account Balance'].mean(), 2)

# Build Method for Confidence Interval
def confint(df, sample_size = 100, alpha = 0.05):
    # Sample The Dataframe
    sampled_df = sample(df, sample_size = sample_size)

    # Sample Mean
    sample_mean = sampled_df['Account Balance'].mean()
    sample_std = sampled_df['Account Balance'].std()

    #Critical Value
    z = norm.ppf(1-alpha/2)

    # Bounds
    lower_bound = sample_mean - z * sample_std / np.sqrt(sample_size)
    upper_bound = sample_mean + z * sample_std / np.sqrt(sample_size)

    return lower_bound, upper_bound


confidence_interval = confint(df)
lower_bound = round(confidence_interval[0], 2)
upper_bound = round(confidence_interval[1], 2)

print('Running Confidence Interval..')
print(f'Lower Bound of Confidence Interval {lower_bound}, Upper Bound of Confidence Interval {upper_bound}.')

if lower_bound <= pop_mean <= upper_bound:
    print(f'The population mean of {pop_mean} is within the Confidence Interval.')
else:
    print(f'The population mean of {pop_mean} is not within the Confidence Interval.')
