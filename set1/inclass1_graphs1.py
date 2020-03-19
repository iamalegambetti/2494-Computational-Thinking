# Import dependencies
import numpy as np
from itertools import combinations

# Alessandro Gambetti
# Student ID: 40755

# IMPORTANT
# Scroll to the end of the page to tune the choose_algorithm() method according to
# the guidelines written below.


#################### ALGORITHM 1 ###############################################
# Greedy Force

def greedy_force(matrix, method, capacity = 7):

    # Transpose the matrix for easiness of sorting later
    matrix = matrix.T

    # Apply a different case for each method
    if method == 'weights':

        # Sort the matrix by weiths -> keeping the correct order
        matrix = matrix[matrix[:,1].argsort()]

        items = []
        iteration_index = 0

        # Perform selection of items
        while capacity >= 0:
            current_weight = matrix[iteration_index, 1]
            # Making sure the difference capacity and current weight is positive or null
            # If non negative perform the operation of appending an item and decrease capacity
            if capacity - current_weight >= 0:
                capacity -= current_weight
                current_item = matrix[iteration_index, 0]
                items.append(current_item)
                iteration_index += 1
                if iteration_index > 4:
                    break
            # If negative pass the element and go to the next one
            else:
                iteration_index += 1
                if iteration_index > 4:
                    break

        items = [int(i) for i in items]
        return f'Items selected with the {method} method are {items}.'

    elif method == 'values':

        # The same reasoning as above, except for the order of trasversing the vector
        matrix = matrix[matrix[:,2].argsort()]

        items = []
        iteration_index = matrix.shape[0] - 1 # Start from bottom to first

        while capacity >= 0:
            current_weight = matrix[iteration_index, 1]
            # The same reasoning as in case 'weights' but reversed
            if capacity - current_weight >= 0:
                capacity -= current_weight
                current_item = matrix[iteration_index, 0]
                items.append(current_item)
                iteration_index -= 1
                if iteration_index < 0:
                    break
            else:
                iteration_index -= 1
                if iteration_index < 0:
                    break

        items = [int(i) for i in items]
        return f'Items selected with the {method} method are {items}.'

    elif method == 'density':
        # Same reasoning as 'values'
        matrix = matrix[matrix[:,3].argsort()]

        items = []
        iteration_index = matrix.shape[0] - 1

        while capacity >= 0:
            current_weight = matrix[iteration_index, 1]
            if capacity - current_weight >= 0:
                capacity -= current_weight
                current_item = matrix[iteration_index, 0]
                items.append(current_item)
                iteration_index -= 1
                if iteration_index < 0:
                    break
            else:
                iteration_index -= 1
                if iteration_index < 0:
                    break

        items = [int(i) for i in items]
        return f'Items selected with the {method} method are {items}.'

    else:
        # Account for wrong spellings in method
        return "Wrong criterion to sort. Please input one of the following 'weights', 'values', or 'density'."


#################### ALGORITHM 2 ###############################################
# Brute Force

def brute_force(matrix, capacity = 7):

    # Matrix Definition:
    # Items in first row, Weights in second row, and Values in third row

    # Exract all the elements
    items = list(matrix[0, :])
    weights = list(matrix[1, :])
    values = list(matrix[2, :])

    # Extract all the possible combinations
    combs = []

    for i in range(1, len(items)+1):
        els = [list(x) for x in combinations(items, i)]
        combs.extend(els)

    # Create dictionary for easier mapping both for values and weights. The keys are the items
    dict_weights = dict(zip(items, weights))
    dict_values = dict(zip(items, values))

    # Selection Process
    best_value = 0
    best_comb = None

    for comb in combs:
        current_weight = 0
        for element in comb:
            current_weight += dict_weights[element]

        if current_weight <= capacity:
            current_value = 0
            for element in comb:
                current_value += dict_values[element]

        if current_value > best_value:
            best_value = current_value
            best_comb = None
            best_comb = comb

    return f'The best combination of items includes items {best_comb} with a total value of {best_value}.'



# IMPORTANT
#################### Testing ###############################################
if __name__ == "__main__":
    # Creating Data
    # order: item id, weight, value
    compass = [1, 3, 10]
    wrench = [2, 2, 6]
    chain = [3, 4, 11]
    hub_wrench = [4, 2, 4]
    tyre = [5, 3, 5]

    # Store in a Matrix
    matrix = np.array([compass, wrench, chain, hub_wrench, tyre]).T

    # Insert Density
    density = matrix[ 2, ] / matrix[1, ]

    # Stack Everything
    matrix = np.vstack((matrix, density))

    # Computation
    def choose_algorithm(matrix, algo, method = None, capacity = 7):
        if algo == 'greedy':
            if method == None:
                message = "Forgot to input the method of selection. 'weights', 'values', or 'density'."
                raise Exception(message)
            greedy = greedy_force(matrix, method, capacity)
            print('Selected the Greedy Force Algorithm.')
            print(greedy)

        elif algo == 'brute':
            brute = brute_force(matrix, capacity)
            print('Selected the Brute Force Algorithm.')
            print(brute)

        else:
            print("Wrong algorithm selected. Please choose either 'greedy' or 'brute'.")




    # IMPORTANT -> FUNCTION TO COMPLETE
    # if the second parameter == 'greedy', please type as third parameter one of the following:
    # 'weights', 'values', 'density'
    # if the second parameter == 'brute', please leave the third parameters blank
    choose_algorithm(matrix, 'brute', 'values')
