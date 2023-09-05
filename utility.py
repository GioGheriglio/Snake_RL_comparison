import numpy as np
import random
import matplotlib.pyplot as plt

# UTILITY for snake game

def give_position(starting_position, direction):
    match direction:
        case 0:
            return (starting_position[0]-1, starting_position[1])
        case 1:
            return (starting_position[0], starting_position[1]+1)
        case 2:
            return (starting_position[0]+1, starting_position[1])
        case 3:
            return (starting_position[0], starting_position[1]-1)
        
def place_fruit(game_matrix):
    indexes = np.where(game_matrix == 0)
    fruit = random.randrange(len(indexes[0]))
    goal = (indexes[0][fruit], indexes[1][fruit])
    return goal

def update_state(snake, goal, game_dimension):
    head_position = snake[0]*game_dimension + snake[1]
    goal_position = goal[0]*game_dimension + goal[1]
    distance = abs(snake[0]-goal[0])+abs(snake[1]-goal[1])
    return np.array([head_position, goal_position, distance])

def create_game_matrix(game_matrix, snake, goal):
    game_matrix[snake] = 1
    game_matrix[goal] = 2

    return game_matrix

def standard_movement(game_matrix, snake, new_position):
    game_matrix[snake] = 0
    snake = new_position
    game_matrix[snake] = 1
    return game_matrix, snake

def is_getting_closer(snake, new_position, goal):
    distance_curr = abs(new_position[0]-goal[0])+abs(new_position[1]-goal[1])
    distance_prev = abs(snake[0]-goal[0])+abs(snake[1]-goal[1])
    if distance_curr < distance_prev:
        return True
    
    return False

##  Checks whether the position brings to a wrong end
#   1) The head is going outside of the matrix
def is_lose(position, game_dimension):
    if position[0] < 0 or position[0] >= game_dimension or position[1] < 0 or position[1] >= game_dimension:
        return True
    return False

def plot_results(episodes, rewards, steps, average, agent):
    fig, ax = plt.subplots(2)
    fig.suptitle('Results')
    ax[0].plot(episodes, rewards, label='Original data')
    ax[0].plot(episodes, average, 'k-', label='Running average')
    ax[0].set_yticks([0, 25, 50, 75])
    ax[0].grid(linestyle='-')
    ax[0].set(ylabel="rewards")
    ax[0].legend()
    ax[1].plot(episodes, steps)
    ax[1].set_yticks([0, 100, 200, 300])
    ax[1].grid(linestyle='-')
    ax[1].set(xlabel="episodes", ylabel="steps")

    fig.tight_layout()
    fig.savefig('Results/{}/training_snake_{}.png'.format(agent, episodes[-1]))

    plt.close()