import numpy as np
import argparse
from collections import deque
import utility

from snake_env import SnakeEnv
from DQN_snake import DQNAgent

action_space = [0, 1, 2, 3]

action_size = len(action_space)

state_size = [3]

STEP_RANGE = 300

if __name__ == '__main__':
    argParser = argparse.ArgumentParser()
    argParser.add_argument('-a', '--agent', default='DQN',  help='Specify the Agent you want to use to train [DQN]')
    argParser.add_argument('-m', '--model', help='Model to continue the training from')
    argParser.add_argument('-e', '--episode', type=int, default=2000, help='Number of episode to train')
    argParser.add_argument('-p', '--epsilon', type=float, default=1.0, help='The starting epsilon of the agent, default to 1.0.')
    args = argParser.parse_args()

    env = SnakeEnv()

    match args.agent:
        case 'DQN':
            agent = DQNAgent(action_space, action_size, state_size, args.epsilon)

    if args.model:
        agent.load(args.model)
    if args.episode:
        episode_range = args.episode

    batch_size = 32
    rewards = []
    steps = []
    fruits = []
    epsilons = []
    average = []

    last_average_100 = deque(maxlen=100)

    for e in range(1, episode_range+1):
        state = env.reset()
        
        total_reward = 0
        n_fruits = 0
        neg_consec_reward = 0

        for step in range(STEP_RANGE):
            action = agent.act(state)

            next_state, reward, done = env.step(action)
            total_reward += reward

            if reward == 1:
                n_fruits += 1

            if reward == 0:
                neg_consec_reward += 1
            else:
                neg_consec_reward = 0

            if neg_consec_reward == 30:
                reward = -0.1

            agent.remember(state, action, reward, next_state, done)

            state = next_state

            if done or total_reward < 0 or neg_consec_reward == 30:
                print("episode: {}/{}, tot_reward: {}, fruits: {}, time alive: {}, epsilon: {:.2}"
                    .format(e, episode_range, total_reward, n_fruits, step+1, agent.epsilon))
                
                break

            if len(agent.memory) > batch_size:
                agent.replay(batch_size)

        # Episode reached the max number of steps
        if step+1==STEP_RANGE:
            print("Max number of steps reached!")
            print("episode: {}/{}, tot_reward: {}, fruits: {}, time alive: {}, epsilon: {:.2}"
                    .format(e, episode_range, total_reward, n_fruits, step+1, agent.epsilon))

        rewards.append(total_reward)
        steps.append(step+1)
        fruits.append(n_fruits)
        epsilons.append(agent.epsilon)

        last_average_100.append(total_reward)
        average.append(np.average(last_average_100))

        # Update Target Weights every 25 episodes
        if e % 25 == 0:
            agent.update()

        if e % 1000 == 0:
            agent.save("./Models/DQN/snake-{}.h5".format(e))
            episodes = np.arange(1, e+1)
            utility.plot_results(episodes, rewards, steps, average, args.agent)

    if e == episode_range:
        print()
        print("End of the episode range!")
        agent.save("./Models/DQN/snake-{}.h5".format(e))
        episodes = np.arange(1, e+1)
        utility.plot_results(episodes, rewards, steps, average, args.agent)