import random
import matplotlib.pyplot as plt
import numpy as np
import math
import sys
import copy


NUM_OF_CITIES = 5
CITIES = []
ROADS_A = []
ROADS_B = []
CONNECTIONS_A = {}
CONNECTIONS_B = {}


class Node:
    def __init__(self, element=None, next_element=None):
        self.element = element
        self.next = next_element


class Queue:
    def __init__(self):
        self.head = None

    def insert(self, element):
        new_node = Node(element)
        if self.head:
            current = self.head
            while current.next:
                current = current.next
            current.next = new_node
        else:
            self.head = new_node

    def delete(self):
        if self.head:
            temp = self.head
            self.head = temp.next
            temp.next = None
            return temp.element

    def empty(self):
        if self.head is None:
            return True


class Stack:
    def __init__(self):
        self.elements = []
        self.max_index = 500
        self.index = -1

    def push(self, element):
        if not Stack.is_full(self):
            self.index += 1
            self.elements += [element]
        else:
            print("pushing full stack")
            sys.exit(1)

    def pop(self):
        if not Stack.is_empty(self):
            temp = self.elements[self.index]
            del self.elements[self.index]
            self.index -= 1
            return temp
        else:
            print("popping empty stack")
            sys.exit(1)

    def is_empty(self):
        if len(self.elements) == 0:
            return True
        else:
            return False

    def is_full(self):
        if self.index == self.max_index:
            return True
        else:
            return False

    def top(self):
        if not Stack.is_empty(self):
            return self.elements[self.index]
        else:
            return None


class Tree:
    def __init__(self, index=None, value=None):
        self.index = index
        self.value = value
        self.first = []


def init_cities():
    x = np.empty(shape=NUM_OF_CITIES)
    y = np.empty(shape=NUM_OF_CITIES)

    for i in range(NUM_OF_CITIES):
        x[i] = random.randint(-100, 100)
        y[i] = random.randint(-100, 100)
        CITIES.append([x[i], y[i]])


def init_roads_a():
    for i in range(NUM_OF_CITIES):
        ROADS_A.append([])
        for j in range(NUM_OF_CITIES):
            if i == j:
                ROADS_A[i].append(0)
            else:
                ROADS_A[i].append(math.sqrt((CITIES[i][0] - CITIES[j][0]) ** 2 + (CITIES[i][1] - CITIES[j][1]) ** 2))


def init_roads_b():
    for i in range(NUM_OF_CITIES):
        ROADS_B.append([])
        for j in range(NUM_OF_CITIES):
            if i == j:
                ROADS_B[i].append(0)
            elif random.randrange(0, 100) <= 80:
                ROADS_B[i].append(math.sqrt((CITIES[i][0] - CITIES[j][0])**2 + (CITIES[i][1] - CITIES[j][1])**2))
            else:
                ROADS_B[i].append("None")

    for i in range(NUM_OF_CITIES):
        for j in range(NUM_OF_CITIES):
            if ROADS_B[i][j] == "None":
                ROADS_B[j][i] = "None"


def create_connections(roads, destination):
    temp = []
    for i in range(NUM_OF_CITIES):
        for j in range(NUM_OF_CITIES):
            if roads[i][j] != 0 and roads[i][j] != "None":
                temp.append(j)
        destination[i] = temp
        temp = []


def draw_graph(option):
    x_values = np.empty(shape=NUM_OF_CITIES)
    y_values = np.empty(shape=NUM_OF_CITIES)

    for i in range(NUM_OF_CITIES):
        x_values[i] = CITIES[i][0]
        y_values[i] = CITIES[i][1]

    plt.scatter(x_values, y_values, marker='o', c='r')

    if option == "a":
        for i in range(NUM_OF_CITIES):
            for j in range(i):
                x_to_plot = [CITIES[i][0], CITIES[j][0]]
                y_to_plot = [CITIES[i][1], CITIES[j][1]]
                plt.plot(x_to_plot, y_to_plot, 'grey')
    elif option == "b":
        for i in range(NUM_OF_CITIES):
            for j in range(NUM_OF_CITIES):
                if ROADS_B[i][j] != 'None':
                    x_to_plot = [CITIES[i][0], CITIES[j][0]]
                    y_to_plot = [CITIES[i][1], CITIES[j][1]]
                    plt.plot(x_to_plot, y_to_plot, 'k')

    for i in range(NUM_OF_CITIES):
        plt.annotate(f'{i}', CITIES[i], fontsize=12, fontweight=800, c='k')

    plt.xlim([-100, 100])
    plt.ylim([-100, 100])
    plt.show()


def bfs(roads, connection):
    queue = Queue()
    source = Tree(0, 0)
    source.first.append(source.index)
    queue.insert(source)
    result = []
    routes = []
    while not queue.empty():
        curr_element = queue.delete()
        for value in connection[curr_element.index]:
            if roads[curr_element.index][value] != 'None':
                if value not in curr_element.first:
                    new_node = Tree(value, curr_element.value + roads[curr_element.index][value])
                    new_node.first += curr_element.first
                    new_node.first.append(value)
                    queue.insert(new_node)
                    if len(new_node.first) == NUM_OF_CITIES:
                        if 0 in connection[new_node.index]:
                            result.append(new_node.value + roads[new_node.index][source.index])
                            routes.append(new_node.first + [0])
                            continue
                else:
                    continue

    show_results(routes, result)


def dfs(roads, connection):
    queue = Stack()
    source = Tree(0, 0)
    source.first.append(source.index)
    queue.push(source)
    routes = []
    results = []
    while not queue.is_empty():
        curr_element = queue.pop()
        for value in connection[curr_element.index]:
            if is_similar(curr_element, value, connection, results, routes, roads, queue, source) == "continue":
                continue

    show_results(routes, results)


def minimum_spanning_tree(roads, connection):
    new_graph = copy.deepcopy(roads)
    minimum = sys.float_info.max
    x, y = 0, 0
    cost = 0
    route = []
    for k in range(16):
        for i in range(NUM_OF_CITIES):
            for j in range(NUM_OF_CITIES):
                if new_graph[i][j] != 'None':
                    if new_graph[i][j] < minimum and new_graph[i][j] != 0:
                        minimum = new_graph[i][j]
                        x = i
                        y = j
        new_graph[x][y] = 0
        new_graph[y][x] = 0
        minimum = sys.float_info.max
        if x in route and y in route:
            continue
        elif x in route and y not in route:
            cost += roads[x][y]
            route.append(y)
        elif x not in route and y in route:
            cost += roads[x][y]
            route.append(x)
        elif x not in route and y not in route:
            cost += roads[x][y]
            route.append(x)
            route.append(y)
        if len(route) == NUM_OF_CITIES:
            break
    if route:
        if route[0] not in connection[route[NUM_OF_CITIES - 1]]:
            print("\nCould not find any route.")
        else:
            route.append(route[0])
            print(f"\nRoute: {route} of cost {round(cost, 3)}")
    else:
        print("\nCould not find any route.")


def greedy_search(roads, connection):
    queue = Stack()
    source = Tree(0, 0)
    source.first.append(source.index)
    queue.push(source)
    route = []
    cost = []
    min_val = sys.float_info.max
    new_val = 0
    while not queue.is_empty():
        curr_element = queue.pop()
        for value in connection[curr_element.index]:
            if roads[curr_element.index][value] < min_val and value not in curr_element.first:
                new_val = value
        if is_similar(curr_element, new_val, connection, cost, route, roads, queue, source) == "continue":
            continue

    if not route:
        print("\nCould not find any route.")
    else:
        print(f"\nRoute: {route[0]} of cost {round(cost[0], 3)}")


def is_similar(curr_element, value, path, results, routes, roads, queue, source):
    if roads[curr_element.index][value] != 'None':
        if value not in curr_element.first:
            new_node = Tree(value, curr_element.value + roads[curr_element.index][value])
            new_node.first += curr_element.first
            new_node.first.append(value)
            queue.push(new_node)
            if len(new_node.first) == NUM_OF_CITIES:
                if 0 in path[new_node.index]:
                    results.append(new_node.value + roads[new_node.index][source.index])
                    routes.append(new_node.first + [0])
                    return "continue"
        else:
            return "continue"


def show_results(routes, costs):
    result_id = 0
    if not routes or not costs:
        print("Could not find any routes")
    else:
        for i in range(len(routes)):
            if i > 0:
                if costs[i] < costs[result_id]:
                    result_id = i

            print(f"Route nr {i + 1} {routes[i]} of cost {round(costs[i], 3)}")

        print(f"\nShortest route {routes[result_id]} of cost {round(costs[result_id], 3)}")


def print_cities(cities):
    for i in range(len(cities)):
        print(f"City {i}: {cities[i]}")


def print_connections(connections):
    for i in range(len(connections)):
        print(f"From {i} to {connections[i]}")


def print_roads(roads):
    for i in range(len(roads)):
        print(f"\nRoads from city {i}:")
        for j in range(NUM_OF_CITIES):
            if type(roads[i][j]) == float and roads[i][j] != 0:
                print(round(roads[i][j], 3))


def execute_a():
    init_cities()
    init_roads_a()
    create_connections(ROADS_A, CONNECTIONS_A)

    print("------ Cities ------\n")
    print_cities(CITIES)
    print("\n---- Connections ----\n")
    print_connections(CONNECTIONS_A)
    print("\n------- Roads -------")
    print_roads(ROADS_A)
    print("\n-------- BFS --------\n")
    bfs(ROADS_A, CONNECTIONS_A)
    print("\n-------- DFS --------\n")
    dfs(ROADS_A, CONNECTIONS_A)
    print("\n-- Minimum Spanning Tree --")
    minimum_spanning_tree(ROADS_A, CONNECTIONS_A)
    print("\n----- Greedy Search -----")
    greedy_search(ROADS_A, CONNECTIONS_A)

    draw_graph("a")


def execute_b():
    init_cities()
    init_roads_b()
    create_connections(ROADS_B, CONNECTIONS_B)

    print("------ Cities ------\n")
    print_cities(CITIES)
    print("\n---- Connections ----\n")
    print_connections(CONNECTIONS_B)
    print("\n------- Roads -------\n")
    print_roads(ROADS_B)
    print("\n-------- BFS --------\n")
    bfs(ROADS_B, CONNECTIONS_B)
    print("\n-------- DFS --------\n")
    dfs(ROADS_B, CONNECTIONS_B)
    print("\n-- Minimum Spanning Tree --")
    minimum_spanning_tree(ROADS_B, CONNECTIONS_B)
    print("\n----- Greedy Search -----")
    greedy_search(ROADS_B, CONNECTIONS_B)

    draw_graph("b")


def main():
    execute_a()
    # execute_b()


if __name__ == "__main__":
    main()
