import numpy as np
import random
import utility

##  This environment creates a table of 5 x 5
#   Each element of the table can have 3 values:
#   0 - empty
#   1 - snake
#   2 - fruit

##  The snake is only made of a head (1 element)

##  Possible actions: (snake can only move horizontally or vertically)
#   0 - up
#   1 - right
#   2 - down
#   3 - left

##  Position in the state can be defined as current_row * nrows + current_col 

##  The state given by the environment is made of 4 elements
#   SNAKE_HEAD    GOAL    DISTANCE(S_H - G)

##  TO DO:
##  Add a seed to the reset function

DIMENSION = 5

action_space = [0, 1, 2, 3]

action_size = len(action_space)

state_size = [3]

class SnakeEnv():

    def __init__(self):
        self.game_matrix = np.zeros((DIMENSION, DIMENSION), dtype=int)
        self.snake = (-1, -1)
        self.goal = (-1, -1)
        self.done = False
        self.state = np.empty(3)

    def step(self, action):
        if self.snake == self.goal:
            self.goal = utility.place_fruit(self.game_matrix)
            self.game_matrix[self.goal] = 2

        new_head_coordinates = utility.give_position(self.snake, action)
        if utility.is_lose(new_head_coordinates, DIMENSION):
            reward = -1
            self.done = True
            return self.state, reward, self.done

        if self.game_matrix[new_head_coordinates] == 0:
            reward = 0
            self.game_matrix, self.snake = utility.standard_movement(self.game_matrix, self.snake, new_head_coordinates)
        else:
            reward = 1
            self.game_matrix, self.snake = utility.standard_movement(self.game_matrix, self.snake, new_head_coordinates)

        self.state = utility.update_state(self.snake, self.goal, DIMENSION)

        return self.state, reward, self.done 
        

    def reset(self):
        self.done = False
        self.game_matrix.fill(0)

        # The snake is placed randomly
        self.snake = (random.randrange(DIMENSION), random.randrange(DIMENSION))

        # The fruit is placed randomly in an empty spot
        self.goal = utility.place_fruit(self.game_matrix)

        self.game_matrix = utility.create_game_matrix(self.game_matrix, self.snake, self.goal)

        ## Let's create the state
        self.state = utility.update_state(self.snake, self.goal, DIMENSION)

        return self.state

    def render(self):
        print(self.game_matrix)
        print()