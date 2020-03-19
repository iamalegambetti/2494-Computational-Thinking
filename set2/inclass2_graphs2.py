# Import Dependencies
import networkx as nx
import numpy as np
import random


# Depth First Search (DFS) Algorithm
def DFS(G, s):

    # Inputs: Graph and starting node

    # Create array of visited nodes
    visited = np.zeros((G.number_of_nodes()))
    # visited is = 1 if the node is visited, 0 otherwise

    # Keep track of previous visited node
    prev_node = np.zeros((G.number_of_nodes()))

    # Create the stack
    Stack = []
    # Input in stack (node, prev_node)

    # Beginning of the Stack
    Stack.append((s, 0))

    # Implement while loop until the Stack is not empty
    while not len(Stack) == 0:
        u, pred = Stack.pop() # Last pair
        if visited[u-1] == 0:
            visited[u-1] = 1
            prev_node[u-1] = pred
            neighbors_of = list(G.neighbors(u))
            neighbors_of.sort(reverse = True)

            # Now explore the list of neightnors
            for w in neighbors_of:
                if visited[w-1] == 0:
                    Stack.append((w, u))

    return prev_node


# Breadth First Search (BFS) Algorithm
def BFS(G, s):

    # Inputs: Graph and starting node

    # Create array of visited nodes
    visited = np.zeros((G.number_of_nodes()))
    # visited is = 1 if the node is visited, 0 otherwise

    # Keep track of previous visited node
    prev_node = np.zeros((G.number_of_nodes()))

    # Create the Queue
    Queue = []
    # Input in Queue (node, prev_node)

    # Beginning of the Queue
    Queue.append((s, 0))

    # Implement while loop until the Queue is not empty
    while not len(Queue) == 0:
        u, pred = Queue.pop(0) # Last pair
        if visited[u-1] == 0:
            visited[u-1] = 1
            prev_node[u-1] = pred
            neighbors_of = list(G.neighbors(u))
            neighbors_of.sort()

            # Now explore the list of neightnors
            for w in neighbors_of:
                if visited[w-1] == 0:
                    Queue.append((w, u))

    return prev_node


# Find Shortest Path
def shortest_path(s, k, visited_nodes):

    # Input: s = start, k = last, visited_nodes = list of visited nodes

    # Edit visited nodes
    visited_nodes = list(visited_nodes)
    visited_nodes = [int(i) for i in visited_nodes]

    # Create the nodes
    nodes = list(range(1, len(visited_nodes) +1))

    # Create a Dictionary of Connections
    connections = dict(zip(nodes, visited_nodes))

    # Initialize lenght and list of previous nodes
    length = 0
    previouses = []
    previouses.append(k) # Append the starting node

    # Looping
    while True:
        previous = connections[k]
        previouses.append(previous)
        k = previous
        length += 1
        if previous == s:
            break

    return previouses[::-1], length


# Random Walk Algorithm
def random_walk(trials, steps):

    """
     Define Number of Trials = How many Random Walk you Want
     Steps = How many Steps to take for each Random Walk

     For k in range n trials
        inital position 0,0
        current position = ipos
         for i in range(nsteps)
            #current position = current position + random choice movement (set of possible movements)
         compute distance -> square root of current position since start at 0
    """

    # Define vector of possible movements
    pmov = [[1,0], [0,1], [-1,0], [0,-1]]

    # Define Master Initial Position
    ipos = [0, 0]

    # Initialize the distances for each trial
    distances = []

    # Iterate throught how many trials
    for k in range(trials):
        ip = ipos
        cp = ip
        distance = 0
        # Iterate for n steps
        for j in range(steps+1):
            movement = random.choice(pmov)
            cp[0] = cp[0] + movement[0]
            cp[1] = cp[1] + movement[1]
            if j == steps:
                distance = np.sqrt(cp[0]**2 + cp[1]**2)
        distances.append(distance)

    # Compute target outputs
    minimum_distance = round(min(distances), 2)
    max_distance = round(max(distances), 2)
    mean_distance = round(sum(distances) / len(distances), 2)
    median_distance = round(sorted(distances)[len(distances)//2],2)

    return minimum_distance, max_distance, mean_distance, median_distance


# Execute the code here
if __name__ == '__main__':
    random.seed(2494)

    # Create Graph with 1->9 nodes and respective edges
    G = nx.Graph()
    nodes = [1,2,3,4,5,6,7,8,9]
    G.add_nodes_from(nodes)
    G.add_edges_from([(1,2), (2,5), (1,3), (3, 4), (3,8), (4,9), (4,6), (6,7), (7,8), (8, 9)])

    def choose_task():
        task = input(" Please select a task between 'Shortest Path' and 'Random Walk': ")

        if task == 'Shortest Path':
            starting_node = int(input(f'Please type a number between 1 and {len(nodes)} as a starting node: '))
            end_node = int(input(f'Please type a number between 1 and {len(nodes)} as a end node: '))
            if starting_node not in nodes or end_node not in nodes:
                raise ValueError(f'Wrong inputs. Make sure start node and end note are in nodes.')
            visited_nodes = BFS(G, starting_node)
            shortest_path_ = shortest_path(starting_node, end_node, visited_nodes)
            print(f'Shortest path between node {starting_node} and node {end_node} is {shortest_path_[0]} with a length of {shortest_path_[1]} steps.')

        elif task == 'Random Walk':
            print('Please Choose the Number of Trials and Steps for Each Trial.')
            trials = int(input('Please type the number of trials: '))
            steps_string = input('Please type the arbitrary number of steps separated by a space: ')
            steps_list = steps_string.split()
            steps_list = [int(i) for i in steps_list]

            if trials < 0:
                raise ValueError('Trials number cannot be lower than 0.')
            print('Start Printing Random Walks for different steps.')

            for step in steps_list:
                print(f'Random Walk with step {step}:')
                min, max, mean, median = random_walk(trials, step)

                print(f'Current min is {min}')
                print(f'Current max is {max}')
                print(f'Current mean is {mean}')
                print(f'Current median is {median}')

        else:
            raise Exception("Task not Selected Correctly. Choose either 'Shortest Path' or 'Random Walk'.")

    choose_task()
