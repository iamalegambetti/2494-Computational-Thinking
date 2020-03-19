
# Alessandro Gambetti 40755


# Import Dependencies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Load Data
acc_rec = pd.read_excel('Accounts Receivable.xlsx')

# Question 1 - Random Sample
def sample(df, sample_size = 50):
    N = len(df)

    # Create an array of Random Indexes
    indexes = list(np.random.randint(0, N-1, size = sample_size))
    sample_ = df.iloc[indexes] # Index the dataframe
    return sample_


# Question 1 - Stratified Sample
def stratified_sample(df, weight, sample_size = 50):
    N = len(df)
    # Compute Probabilities for the weight column
    weights_prob = df[weight].value_counts() / N

    # Get classes
    classes = list(weights_prob.index)
    # Get Probabilities
    weights_prob = list(weights_prob)

    # Initialize statement for random choice of each class weighted by its probability
    weight_list = "np.random.choice(classes, p = weights_prob)"

    # Initialize List for Indexing
    sampled_indexes = []

    # iterate through each time the sample size length
    for i in range(sample_size):
        # create rule to select the dataframe: select a class weighted by its probabilities
        rule = df[weight] == eval(weight_list)
        sampled_obs = df[rule] # implement rule
        indexes = list(sampled_obs.index) #get indexes
        index = np.random.randint(indexes[0], indexes[-1]) # get one observation from the selected dataframe
        sampled_indexes.append(index) #append index

    return df.iloc[sampled_indexes]



# Question 2
def roulette():
    # Create the game
    numbers = list(range(0, 37 + 1))
    reds = numbers[1:19]
    blacks = numbers[19:37]
    green = numbers[0] # An Int

    N = 37

    # Create Probabilities
    prob_red, prob_black, prob_green = len(reds)/N, len(blacks)/N, 1/N

    # Return Random pick weighted by Probability
    return np.random.choice(['Red!', 'Black!', 'Green!'], p = [prob_red, prob_black, prob_green])

def play(n_plays = 20):

    # Play the Game according to the benefits implied
    balance = 0
    for _ in range(n_plays):
        current_play = roulette()
        if current_play == 'Red!':
            balance += 4
        elif current_play == 'Black!':
            balance -= 6
        elif current_play == 'Green!':
            balance += 24
        else:
            print('Error!')

    return balance

def simulate_plays(n_trials = 1000):

    # Simulate Trials for number of "n_trials"
    balances = []
    for _ in range(n_trials):
        balances.append(play())

    # Get Mean and Stdev for each Balance
    x = np.mean(balances)
    std = np.std(balances)

    # Critical_value
    alpha = 0.05
    z = norm.ppf(1-alpha/2)

    # Create Confints
    lower_bound = n_trials * x - z * std / np.sqrt(n_trials) * n_trials
    upper_bound = n_trials * x + z * std / np.sqrt(n_trials) * n_trials

    return round(lower_bound, 2), round(upper_bound, 2)



# Question 3

# Load Data
drinks = pd.read_excel('softdrink.xlsx')

# Compute Probabilities
preferences_count = drinks['Preference'].value_counts() / len(drinks)


# Build Confint General Confint
def sample_confint():

    # Retreive probabilities
    p_ours, p_not_ours = tuple(preferences_count.values)

    # Compute the average standard error
    s = np.sqrt(p_ours * p_not_ours / len(drinks))

    # Critical Value
    alpha = 0.1
    z = norm.ppf(1-alpha/2)

    # Compute Upper and Lower Bound
    lower_bound = p_ours - z * s
    upper_bound = p_ours + z * s

    return round(lower_bound, 2), round(upper_bound, 2)


def confint_condition(df, condition, alpha = 0.1):

    # Extract unique values from the condition
    uniques = list(df[condition].unique())

    # Critical Value
    z = norm.ppf(1-alpha/2)

    # Output Dict
    confints = {}

    for unique in uniques:
        print(unique)
        temp_df = df[df[condition] == unique] # Select dataframe according to unique condition in dataframe
        length_df = len(temp_df)
        counts = temp_df['Preference'].value_counts() / length_df # Compute Probabilities
        prob = counts['Our brand'] # Extract probability that they like our brand

        # Standard Error
        s = np.sqrt((prob)*(1-prob)/length_df)

        # Compute Bounds
        lower_bound = prob - z * s
        upper_bound = prob + z * s

        # Store for Output
        confints[unique] = (round(lower_bound, 2), round(upper_bound, 2))

    return confints




############## RUNNING ###############

# Small Companies in Sample

# Question 1
print('Question 1: ')
sampled = sample(acc_rec)
print('The Sampled DataFrame (Head) is:')
print(sampled.head())
sampled_small = len(sampled[sampled['Size'] == 1])
print(f"Number of small companies within it: {sampled_small}.")
print('\n')
print('Stratified Sampled Observations (Head) here: ')
print(stratified_sample(acc_rec, 'Size').head())
print('END Question 1.')

print('\n\n')

# Question 2
print('Question 2:')
print('Starting Simulating Plays for 1000 Trials..')

simulated_plays = simulate_plays()
print(f'Confidence Interval Lower Bound {simulated_plays[0]} and Upper Bound {simulated_plays[1]}.')
print('Simulation Ended.')

print('\n')
print('The game is definitely unfair since both the bounds are negative in profits.')

print('END Question 2.')


print('\n\n')

# Question 3
print('Question 3:')
print('The General Sample Confindence Interval is:')
print(sample_confint(), "for lower and upper bound, respectively.")

print('\n')
print('Sexes Confidence Interval to compute are:')

confints_sex = confint_condition(drinks, 'Gender')

for key in confints_sex.keys():
    print(f"Confindence interval for {key} is: Lower Bound: {confints_sex[key][0]}, Upper Bound: {confints_sex[key][1]}.")

print('\n')
print('Ages Confidence Interval to compute are:')
confints_ages = confint_condition(drinks, 'Age')

for key in confints_ages.keys():
    print(f"Confindence interval for {key} is: Lower Bound: {confints_ages[key][0]}, Upper Bound: {confints_ages[key][1]}.")

print('END.')
