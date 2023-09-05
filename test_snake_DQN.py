import numpy as np
import argparse

from snake_env import SnakeEnv
from DQN_snake import DQNAgent

action_space = [0, 1, 2, 3]

action_size = len(action_space)

state_size = [3]

STEP_RANGE = 300

if __name__ == '__main__':
    argParser = argparse.ArgumentParser()
    argParser.add_argument('-a', '--agent', default='DQN',  help='Specify the Agent you want to use to test [DQN]')
    argParser.add_argument('-m', '--model', required=True, help='Model to test')
    argParser.add_argument('-e', '--episode', type=int, default=100, help='Number of episode to test')
    argParser.add_argument('-p', '--epsilon', type=float, default=0.0, help='The epsilon of the agent, default to 0')
    args = argParser.parse_args()

    env = SnakeEnv()

    match args.agent:
        case 'DQN':
            agent = DQNAgent(action_space, action_size, state_size, args.epsilon)

    if args.model:
        agent.load(args.model)
    if args.episode:
        episode_range = args.episode

    rewards = []

    for e in range(1, episode_range+1):
        state = env.reset()
        #env.render()

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

            state = next_state

            #env.render()
            #print('action:', action, 'total reward:', total_reward, 'done:', done)
            #print()

            if done or neg_consec_reward == 30:
                print("episode: {}/{}, tot_reward: {}, fruits: {}, time alive: {}, epsilon: {:.2}"
                    .format(e, episode_range, total_reward, n_fruits, step+1, agent.epsilon))
                break

        # Episode reached the max number of steps
        if step+1==STEP_RANGE:
            print("Max number of steps reached!")
            print("episode: {}/{}, tot_reward: {}, fruits: {}, time alive: {}, epsilon: {:.2}"
                    .format(e, episode_range, total_reward, n_fruits, step+1, agent.epsilon))
            
        rewards.append(total_reward)

    mean = np.mean(rewards)
    max = max(rewards)
    print("Mean value: {}, Max value: {}".format(mean, max))
