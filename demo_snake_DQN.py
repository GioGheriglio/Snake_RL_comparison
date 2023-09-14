import numpy as np
import argparse
import time

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
    argParser.add_argument('-e', '--episode', type=int, default=1, help='Number of episode to test')
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

    for e in range(1, episode_range+1):
        state = env.reset()
        print(r"""
 __          __  _                            _           _____             _        _ 
 \ \        / / | |                          | |         / ____|           | |      | |
  \ \  /\  / /__| | ___ ___  _ __ ___   ___  | |_ ___   | (___  _ __   __ _| | _____| |
   \ \/  \/ / _ \ |/ __/ _ \| '_ ` _ \ / _ \ | __/ _ \   \___ \| '_ \ / _` | |/ / _ \ |
    \  /\  /  __/ | (_| (_) | | | | | |  __/ | || (_) |  ____) | | | | (_| |   <  __/_|
     \/  \/ \___|_|\___\___/|_| |_| |_|\___|  \__\___/  |_____/|_| |_|\__,_|_|\_\___(_)
                                                                                       
              """)
        print(r"""

                       /^\/^\
                     _|__|  O|
            \/     /~     \_/ \
             \____|__________/  \
                    \_______      \
                            `\     \                 \
                              |     |                  \
                             /      /                    \
                            /     /                       \\
                          /      /                         \ \
                         /     /                            \  \
                       /     /             _----_            \   \
                      /     /           _-~      ~-_         |   |
                     (      (        _-~    _--_    ~-_     _/   |
                      \      ~-____-~    _-~    ~-_    ~-_-~    /
                        ~-_           _-~          ~-_       _-~
                           ~--______-~                ~-___-~

                    """)
        time.sleep(1)
        print("Let the game begin!")
        print("Initial state:")
        env.render()
        time.sleep(0.5)

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

            env.render()
            print('step:', step+1, 'fruits:', n_fruits)
            print()
            time.sleep(0.5)

            if done or neg_consec_reward == 30:
                print("episode: {}/{}, tot_reward: {}, fruits: {}, time alive: {}, epsilon: {:.2}"
                    .format(e, episode_range, total_reward, n_fruits, step+1, agent.epsilon))
                break

        # Episode reached the max number of steps
        if step+1==STEP_RANGE:
            print("Max number of steps reached!")
            print("episode: {}/{}, tot_reward: {}, fruits: {}, time alive: {}, epsilon: {:.2}"
                    .format(e, episode_range, total_reward, n_fruits, step+1, agent.epsilon))
