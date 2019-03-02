import math

'''
This class represents a whole picture about the state of the game board,
including: the board configuration, the parent node (that holds the previous configuration),
and the movement operator that lead to this state.
It also contains the 'expand_node' function and the functions that builds it.
'''
class Node:

    # Constructor
    def __init__(self, configuration, parent, operator):
        self.configuration = configuration
        self.parent = parent
        self.operator = operator
        self.size = int(math.sqrt(len(self.configuration)))
        self.g_func = 0
        self.h_func = 0
        self.f_func = 0

    # In order to ease the movement computations understanding, and to make the computations on a separate copy.
    def create_temp_board(self):
        board = [[0 for x in range(self.size)] for y in range(self.size)]
        for i in range(self.size):
            for j in range(self.size):
                board[i][j] = self.configuration[(i * self.size) + j]
        return board

    # In order to determine the optional movements.
    def find_zero_indexes(self, board):
        for i in range(self.size):
            for j in range(self.size):
                if board[i][j] == 0:
                    return i, j

    # Convert the board representation into a list representation.
    def get_configuration(self, board):
        current_configuration = [0 for i in range(len(self.configuration))]
        for i in range(self.size):
            for j in range(self.size):
                current_configuration[(i * self.size) + j] = board[i][j]
        return current_configuration

    # Check if a movement in this direction is possible, if it is- return the expanded node.
    def up(self):
        board = self.create_temp_board()
        i, j = self.find_zero_indexes(board)
        if i + 1 < self.size:
            board[i][j] = board[i + 1][j]
            board[i + 1][j] = 0
            new_configuration = self.get_configuration(board)
            expanded_node = Node(new_configuration, self, 'U')
            return expanded_node
        else:
            return None

    # Check if a movement in this direction is possible, if it is- return the expanded node.
    def down(self):
        board = self.create_temp_board()
        i, j = self.find_zero_indexes(board)
        if i - 1 >= 0:
            board[i][j] = board[i - 1][j]
            board[i - 1][j] = 0
            new_configuration = self.get_configuration(board)
            expanded_node = Node(new_configuration, self, 'D')
            return expanded_node
        else:
            return None

    # Check if a movement in this direction is possible, if it is- return the expanded node.
    def left(self):
        board = self.create_temp_board()
        i, j = self.find_zero_indexes(board)
        if j + 1 < self.size:
            board[i][j] = board[i][j + 1]
            board[i][j + 1] = 0
            new_configuration = self.get_configuration(board)
            expanded_node = Node(new_configuration, self, 'L')
            return expanded_node
        else:
            return None

    # Check if a movement in this direction is possible, if it is- return the expanded node.
    def right(self):
        board = self.create_temp_board()
        i, j = self.find_zero_indexes(board)
        if j - 1 >= 0:
            board[i][j] = board[i][j - 1]
            board[i][j - 1] = 0
            new_configuration = self.get_configuration(board)
            expanded_node = Node(new_configuration, self, 'R')
            return expanded_node
        else:
            return None

    # Find the successors nodes of this node.
    def expand_node(self):
        successors_nodes_list = []

        successor_node = self.up()
        if successor_node:
            successors_nodes_list.append(successor_node)

        successor_node = self.down()
        if successor_node:
            successors_nodes_list.append(successor_node)

        successor_node = self.left()
        if successor_node:
            successors_nodes_list.append(successor_node)

        successor_node = self.right()
        if successor_node:
            successors_nodes_list.append(successor_node)

        return successors_nodes_list

    # Define the goal configuration.
    def goal_configuration(self):
        final_configuration = [0 for i in range(len(self.configuration))]
        for i in range(len(self.configuration)-1):
            final_configuration[i] = i+1
        return final_configuration

    # Helpful print method
    def print_node(self):
        print self.configuration, self.parent, self.operator

'''
This class contains three search algorithms: IDS, BFS and A*, each algorithm in a separate function.
In some of these functions there are more functions that will help us to operate the relevant search algorithm.
'''
class Searcher:

    # Constructor
    def __init__(self):
        self.developed_nodes_counter = 0

    # Returns list of the operators path from the root configuration to the given current configuration.
    def operators_path(self, current_configuration):
        path = []
        temp_configuration = current_configuration
        while temp_configuration is not None:
            if temp_configuration.operator:
                path += [temp_configuration.operator]
            temp_configuration = temp_configuration.parent
        path.reverse()
        return path

    # IDS algorithm implementation.
    def ids(self, start_node, goal_configuration):
        import itertools

        # DFS algorithm implementation.
        def dfs(path_list, depth):
            if path_list[-1].configuration == goal_configuration:
                path = self.operators_path(path_list[-1])
                return path
            if depth == 0:
                return None
            successors_list = path_list[-1].expand_node()
            for successor in successors_list:
                self.developed_nodes_counter += 1
                new_path_list = dfs(path_list + [successor], depth - 1)
                if new_path_list:
                    return new_path_list
            return None

        for depth in itertools.count():
            self.developed_nodes_counter = 1
            path = dfs([start_node], depth)
            if path:
                return ''.join(path), str(self.developed_nodes_counter), str(depth)

    # BFS algorithm implementation.
    def bfs(self, start_node, goal_configuration):
        queue = [start_node]
        developed_nodes_counter = 0
        while len(queue) != 0:
            current_node = queue.pop(0)
            developed_nodes_counter += 1
            if current_node.configuration == goal_configuration:
                path = self.operators_path(current_node)
                return ''.join(path), str(developed_nodes_counter), str(0)
            successors_list = current_node.expand_node()
            for successor in successors_list:
                queue.append(successor)

    # A* algorithm implementation.
    def a_star(self, start_node, goal_configuration):

        # In order to ease the heuristic function computations understanding.
        def configuration_as_board(configuration):
            size = int(math.sqrt(len(configuration)))
            board = [[0 for x in range(size)] for y in range(size)]
            for i in range(size):
                for j in range(size):
                    board[i][j] = configuration[(i * size) + j]
            return board

        # Find the given number's indexes in the board.
        def find_number_indexes(board, number):
            size = len(board)
            for i in range(size):
                for j in range(size):
                    if board[i][j] == number:
                        return i, j

        # Calculate the manhattan distance of the given number.
        def manhattan_distance_calculator(current_pos_number_index, goal_pos_number_index):
            return abs(current_pos_number_index[0] - goal_pos_number_index[0]) + \
                   abs(current_pos_number_index[1] - goal_pos_number_index[1])

        # Given the current configuration, sum the manhattan distances of all the numbers in the game board.
        def heuristic_function(current_configuration):
            current_board = configuration_as_board(current_configuration)
            goal_board = configuration_as_board(goal_configuration)
            sum = 0
            for number in range(1, len(current_configuration)):
                current_pos_number_index = find_number_indexes(current_board, number)
                goal_pos_number_index = find_number_indexes(goal_board, number)
                sum += manhattan_distance_calculator(current_pos_number_index, goal_pos_number_index)
            return sum

        # Returns a list of the nodes with the minimal f_func value.
        def front_by_f_func(nodes_list):
            candidates_list = []
            min_f_value = nodes_list[0].f_func
            for index in range(len(nodes_list)):
                if nodes_list[index].f_func < min_f_value:
                    min_f_value = nodes_list[index].f_func
            for index in range(len(nodes_list)):
                if nodes_list[index].f_func == min_f_value:
                    candidates_list.append(nodes_list[index])
            return candidates_list

        # Returns a list of the nodes that inserted most early in time to the given list.
        def front_by_insertion_time(nodes_list):
            candidates_list = []
            oldest_parent_node = nodes_list[0].parent
            for index in range(len(nodes_list)):
                if nodes_list[index].parent == oldest_parent_node:
                    candidates_list.append(nodes_list[index])
            return candidates_list

        # Returns a list of the nodes with the higher prioritized operator (order of importance 'U', 'D', 'L', 'R').
        def front_by_operator(nodes_list):
            up_list = []
            down_list = []
            left_list = []
            right_list = []
            for index in range(len(nodes_list)):
                if nodes_list[index].operator == 'U':
                    up_list.append(nodes_list[index])
                elif nodes_list[index].operator == 'D':
                    down_list.append(nodes_list[index])
                elif nodes_list[index].operator == 'L':
                    left_list.append(nodes_list[index])
                elif nodes_list[index].operator == 'R':
                    right_list.append(nodes_list[index])
            if len(up_list) > 0:
                return up_list
            elif len(down_list) > 0:
                return down_list
            elif len(left_list) > 0:
                return left_list
            elif len(right_list) > 0:
                return right_list

        # Find the given node's index in the given queue.
        def find_index_in_queue(node, queue):
            for index in range(len(queue)):
                if node == queue[index]:
                    return index

        # Find and return the most prioritized node in the queue (according to the predefined order).
        def find_front_element(queue):
            first_candidates_list = front_by_f_func(queue)
            if len(first_candidates_list) == 1:
                return find_index_in_queue(first_candidates_list[0], queue)
            second_candidates_list = front_by_insertion_time(first_candidates_list)
            if len(second_candidates_list) == 1:
                return find_index_in_queue(second_candidates_list[0], queue)
            third_candidates_list = front_by_operator(second_candidates_list)
            if len(third_candidates_list) == 1:
                return find_index_in_queue(third_candidates_list[0], queue)
            return find_index_in_queue(third_candidates_list[0], queue)

        # The A* algorithm
        queue = [start_node]
        developed_nodes_counter = 0
        while len(queue) != 0:
            current_node = queue.pop(find_front_element(queue))
            developed_nodes_counter += 1
            if current_node.configuration == goal_configuration:
                path = self.operators_path(current_node)
                return ''.join(path), str(developed_nodes_counter), str(current_node.g_func)
            successors_list = current_node.expand_node()
            for successor in successors_list:
                successor.g_func = current_node.g_func + 1
                successor.h_func = heuristic_function(successor.configuration)
                successor.f_func = (successor.g_func + successor.h_func)
                queue.append(successor)


# Read the data from the 'input.txt' file.
with open('input.txt') as f:
    search_method = int(f.readline())
    board_size = int(f.readline())
    start_configuration = f.readline().split('-')
    for i in range(len(start_configuration)):
        start_configuration[i] = int(start_configuration[i])

# Create the start Node from the given data.
start_node = Node(start_configuration, None, None)
# Define the goal configuration.
goal_configuration = start_node.goal_configuration()
# Create a 'Searcher' object.
searcher = Searcher()
# Call the appropriate search method.
if search_method == 1:
    operators_path , developed_nodes, depth = searcher.ids(start_node, goal_configuration)
elif search_method == 2:
    operators_path, developed_nodes, depth = searcher.bfs(start_node, goal_configuration)
elif search_method == 3:
    operators_path, developed_nodes, depth = searcher.a_star(start_node, goal_configuration)
# write the returned data to the 'output.txt' file.
with open('output.txt', 'w') as f:
    f.write(operators_path + " " + developed_nodes + " " + depth)