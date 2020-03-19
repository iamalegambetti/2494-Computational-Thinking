import numpy as np
import matplotlib.pyplot as plt
import random

np.random.seed(2494)
#### CRAPS GAME ####
def craps_game(pass_line):

    # Set Sides of Dices
    dice_sides = (1, 2, 3, 4, 5, 6)

    if pass_line == 'Pass Line':

        dice1 = np.random.choice(dice_sides)
        dice2 = np.random.choice(dice_sides)
        total = dice1 + dice2

        if total in (7, 11):
            return 1

        elif total in (2,3,12):
            return -1

        else:
            point = total
            while True:
                dice1 = np.random.choice(dice_sides)
                dice2 = np.random.choice(dice_sides)
                total = dice1 + dice2
                if total == point:
                    return 1
                elif total == 7:
                    return -1

    elif pass_line == "Don't Pass Line":

        dice1 = np.random.choice(dice_sides)
        dice2 = np.random.choice(dice_sides)
        total = dice1 + dice2

        if total in (7, 11):
            return -1

        elif total in (2, 3):
            return 1

        elif total == 12:
            return 0

        else:
            point = total
            while True:
                dice1 = np.random.choice(dice_sides)
                dice2 = np.random.choice(dice_sides)
                total = dice1 + dice2
                if total == point:
                    return -1
                elif total == 7:
                    return 1
    else:
        return "Please type either 'Pass Line' or 'Don't Pass Line', please."

def play():

    total = 0
    ROIs = []

    days = int(input('Please select how many days you want to go to the Casino (Type Integer please): '))
    bets = int(input('Please select the number of bets you want to do (Type Integer please): '))

    user_sel = input("Please Select between either  'Pass Line'  or  'Don't Pass Line'  :")

    for day in range(days):
        for bet in range(bets):
            total += craps_game(user_sel)
        ROI = total / bets
        ROIs.append(ROI)
        total = 0

    return f'Average Return on Investment Across Days is {round(np.mean(ROIs), 2)}.'



####### RANDOM WALKS #######

def random_walk_one(trials, steps):

    pmov = [[1,0], [0,1], [-1,0], [0,-1]] # Vector of possible movements
    ipos = [0, 0] # Initial Position

    distances = []

    for k in range(trials):
        ip = ipos
        cp = ip
        distance = 0
        for j in range(steps+1):
            movement = random.choice(pmov)
            cp[0] = cp[0] + movement[0]
            cp[1] = cp[1] + movement[1]
            if j == steps:
                distance = np.sqrt(cp[0]**2 + cp[1]**2)
        distances.append(distance)

    mean_distance = round(sum(distances) / len(distances), 2)

    return mean_distance

def random_walk_two(trials, steps):

    pmov = [[1,0], [0,1], [-1,0], [0,-2]] # Vector of possible movements
    ipos = [0, 0] # Initial Position

    distances = []

    for k in range(trials):
        ip = ipos
        cp = ip
        distance = 0
        for j in range(steps+1):
            movement = random.choice(pmov)
            cp[0] = cp[0] + movement[0]
            cp[1] = cp[1] + movement[1]
            if j == steps:
                distance = np.sqrt(cp[0]**2 + cp[1]**2)
        distances.append(distance)

    mean_distance = round(sum(distances) / len(distances), 2)

    return mean_distance

def random_walk_three(trials, steps):

    pmov = [[1,0], [-1,0]] # Vector of possible movements
    ipos = [0, 0] # Initial Position

    distances = []

    for k in range(trials):
        ip = ipos
        cp = ip
        distance = 0
        for j in range(steps+1):
            movement = random.choice(pmov)
            cp[0] = cp[0] + movement[0]
            cp[1] = cp[1] + movement[1]
            if j == steps:
                distance = np.sqrt(cp[0]**2 + cp[1]**2)
        distances.append(distance)

    mean_distance = round(sum(distances) / len(distances), 2)

    return mean_distance


########## RUN CODE HERE ##########
# Craps Game
print("LET'S PLAY THE CRAPS GAME!")
print(play())

print('\n')
# Drunk Men Distances Montecarlo
print("LET'S PRINT THE DISTANCE OF EACH DRUNK MAN!")
trials = int(input('Please type the number of trials: '))
steps = int(input('Please type the number of steps: '))

if trials < 0 or steps <0:
    raise ValueError('Trials number cannot be lower than 0.')

one = random_walk_one(trials, steps)
two = random_walk_two(trials, steps)
three = random_walk_three(trials, steps)

print(f'With a number of trials = {trials} and a number of steps = {steps} the first Drunk Man distanced {one}, the second one {two} and the third one {three}!')

print('\n')
# Plotting the Paths
print("LET'S PLOT THE RANDOM PATHS OF EACH DRUNK MAN!")
steps2 = int(input('Please enter the number of steps to complete: '))

random.seed(2494)

pmov1 = [[1,0],[0,1],[-1,0],[0,-1]]
pmov2 = [[1,0],[0,1],[-1,0],[0,-2]]
pmov3 = [[1,0],[-1,0]]

ip = [0,0]

def path_plot (initialpos, movements, nsteps, style):
    currentpos = initialpos
    for i in range(nsteps):
        currentpos = np.sum([currentpos, random.choice(movements)], axis=0)
        plt.plot(currentpos[0], currentpos[1], style)

#plt.figure(figsize = (10, 6))
path_plot(ip, pmov1, steps2, 'r+')
path_plot(ip, pmov2, steps2, 'g+')
path_plot(ip, pmov3, steps2, 'b+')
plt.title('Random Walk for the Three Drunk Men')
plt.xlabel('Red: First Drunk Man, Green: Second Drunk Man, Blue: Third Drunk Man.')

plt.show()
