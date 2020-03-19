# Import dependencies

import numpy as np
import pandas as pd
import itertools
import networkx as nx

# Exercise 1
def sum_squares(n):

    #Rule out exceptions
    assert type(n) == int, 'Please Type a Positive Integer'
    if n < 0:
        raise ValueError('Please Type a Positive Integer')

    total = sum([i**2 for i in range(n)])

    return total


# Exercise 2

# Load Data
investments = pd.read_csv('Investments.csv').values
cash_flows = pd.read_csv('PositiveCashFlows.csv', index_col = 0, sep = ';').fillna(0).values


# Define Npv Function
def npv(inv, cash_flows, rate = 0.1):
    npv = 0
    for i in range(len(cash_flows)):
        disc = cash_flows[i] * (1+rate)**(-i-1)
        npv += disc
    npv -= inv
    return round(float(npv), 2)

# Compute Npv For Each Project
npvs = {}

for i in range(cash_flows.shape[1]):
    npvs[i+1] = npv(investments[:,i], cash_flows[:, i])

# Store in a dictionary each investment project value
investments_dict = dict(zip(list(npvs.keys()), investments.tolist()[0]))

# Best Investments combination
def best_investments(npvs, investments, target = 100000):

    # Create all the possible combinations and select the feasible ones according to a target
    combs = []

    for L in range(0, len(list(investments_dict.keys()))+1):
        for subset in itertools.combinations(list(investments_dict.keys()), L):
            combs.append(subset)

    feasible_combs = []

    for comb in combs:
        total = 0
        for element in comb:
            total += investments_dict[element]
        if total <= target:
            feasible_combs.append(comb)

    # Remove 1 option combinations since they are always dominated
    feasible_combs = [i for i in feasible_combs if len(i) > 1]

    # Select best combinations
    best_comb = None
    best_npvs = 0

    for comb in feasible_combs:

        current_npvs = 0
        for element in comb:
            current_npvs += npvs[element]

        if current_npvs > best_npvs:
            best_npvs = current_npvs
            best_comb = comb

    print(f'Best NPVs value: {best_npvs}.')
    return best_comb



# 1st Constraint combination. 1 and 3
def best_investments_1st_constraint(npvs, investments, target = 100000):

    # Create all the possible combinations and select the feasible ones according to a target
    combs = []

    for L in range(0, len(list(investments_dict.keys()))+1):
        for subset in itertools.combinations(list(investments_dict.keys()), L):
            combs.append(subset)

    feasible_combs = []

    for comb in combs:
        total = 0
        for element in comb:
            total += investments_dict[element]
        if total <= target:
            # Selecting 1 and 3 investments together condition
            if 1 in comb:
                if 3 in comb:
                    feasible_combs.append(comb)
                else:
                    continue
            feasible_combs.append(comb)

    # Remove 1 length option combinations since they are always dominated
    feasible_combs = [i for i in feasible_combs if len(i) > 1]

    # Select best combinations
    best_comb = None
    best_npvs = 0

    for comb in feasible_combs:

        current_npvs = 0
        for element in comb:
            current_npvs += npvs[element]

        if current_npvs > best_npvs:
            best_npvs = current_npvs
            best_comb = comb

    print(f'Best NPVs value: {best_npvs}.')
    return best_comb


# Best Investments Second Constraint
def best_investments_2nd_constraint(npvs, investments, target = 100000):

    # Create all the possible combinations and select the feasible ones according to a target
    combs = []

    for L in range(0, len(list(investments_dict.keys()))+1):
        for subset in itertools.combinations(list(investments_dict.keys()), L):
            combs.append(subset)

    feasible_combs = []

    for comb in combs:
        total = 0
        for element in comb:
            total += investments_dict[element]
        if total <= target:
            # Selecting either 5 or 6 condition
            if 5 in comb or 6 in comb:
                feasible_combs.append(comb)
            else:
                continue

    # Remove 1 length option combinations since they are always dominated
    feasible_combs = [i for i in feasible_combs if len(i) > 1]

    # Select best combinations
    best_comb = None
    best_npvs = 0

    for comb in feasible_combs:

        current_npvs = 0
        for element in comb:
            current_npvs += npvs[element]

        if current_npvs > best_npvs:
            best_npvs = current_npvs
            best_comb = comb

    print(f'Best NPVs value: {best_npvs}.')
    return best_comb


# Exercise 3

# Create Graph

words = 'A B C D E F G H I J K'
nodes = words.split()

G = nx.Graph()
G.add_nodes_from(nodes)
edges = [('A', 'B'), ('A', 'C'), ('A', 'D'), ('B', 'G'), ('B', 'C'), ('D', 'E'),
         ('C', 'G'), ('C', 'E'), ('G', 'H'), ('G', 'F'), ('E', 'F'), ('E', 'I'),
         ('E', 'J'), ('F', 'H'), ('F', 'I'), ('H', 'I'), ('H', 'K'), ('I', 'J'),
         ('I', 'K'), ('J', 'K')]
G.add_edges_from(edges)

# Breadth First Search Algorithm

def BFS(G, s):

    # Inputs: Graph and starting node

    # Create array of visited nodes
    nodes = list(G.nodes())
    visited = dict(zip(nodes, [0]*len(nodes)))

    # visited is = 1 if the node is visited, 0 otherwise

    # Keep track of previous visited node
    prev_node = dict(zip(nodes, [0]*len(nodes)))

    # Create the Queue
    Queue = []
    # Input in Queue (node, prev_node)

    # Beginning of the Queue
    Queue.append((s, None))

    # Implement while loop until the Queue is not empty
    while not len(Queue) == 0:
        u, pred = Queue.pop(0) # Last pair
        if visited[u] == 0:
            visited[u] = 1
            prev_node[u] = pred
            neighbors_of = list(G.neighbors(u))
            neighbors_of.sort()

            # Now explore the list of neightnors
            for w in neighbors_of:
                if visited[w] == 0:
                    Queue.append((w, u))

    return visited, prev_node

# Shortest Path Algorithm

def shortest_path(s, k, visited_nodes):

    # Input: s = start, k = last, visited_nodes = list of visited nodes

    # Initialize lenght and list of previous nodes
    length = 0
    previouses = []
    previouses.append(k) # Append the starting node

    # Looping
    while True:
        previous = visited_nodes[k]
        previouses.append(previous)
        k = previous
        length += 1
        if previous == s:
            break

    return previouses[::-1], length


def select_shortest_path(G, start_node, end_node, stop_by_node, thresold):

    # STOP BY CODE #

    # Initialize variables
    final_path_stop_by = []
    length_stop_by = 0

    # Compute the shortest path from the start node to the stop by node
    _, previous_visited = BFS(G, start_node)
    list_of_nodes_to_k, k_length = shortest_path(start_node, stop_by_node, previous_visited)
    final_path_stop_by.extend(list_of_nodes_to_k)
    length_stop_by += k_length

    # Compute the shortest path from the stop node to the end
    _nd, previous_visited2 = BFS(G, stop_by_node)
    list_of_nodes_to_end, length_end = shortest_path(stop_by_node, end_node, previous_visited2)
    final_path_stop_by.extend(list_of_nodes_to_end)
    length_stop_by += length_end

    # Delete Duplicates Stop in Stop By Nodes
    final_path_stop_by = sorted(set(final_path_stop_by), key = lambda x: final_path_stop_by.index(x))


    # DO NOT STOP BY #

    _rd, previous_visited3 = BFS(G, start_node)
    final_path_one_shot, length_one_shot = shortest_path(start_node, end_node, previous_visited3)


    # SELECTION CRITERION #
    delta_perc = (length_stop_by - length_one_shot)/length_one_shot
    print(f'Tolerance term of {round(delta_perc,2)*100}%.')

    if delta_perc <= thresold:
        print(f'Passing through {stop_by_node} is feasible.')
        return final_path_stop_by
    else:
        print(f'Passing through {stop_by_node} is not feasible.')
        return final_path_one_shot



# Complete Here:
print('Exercise 1:')
print('Sum of Squares up to n.')
print(sum_squares(10))

print('\n')
print('Exercise 2:')
print('Returning Best Combinations for each Case.')
print('Point A.')
print(best_investments(npvs, investments_dict, target = 100000))

print('\n')
print('Point B1:')
print(best_investments_1st_constraint(npvs, investments_dict, target = 100000))
print('Point B2:')
print(best_investments_2nd_constraint(npvs, investments_dict, target = 100000))

print('\n')

print('Exercise 3:')
print(select_shortest_path(G, 'A', 'K', 'F', 0.2))
