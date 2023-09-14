import numpy as np
import tensorflow as tf
import argparse
import time

from snake_env import SnakeEnv
from A2C_snake import A2CAgent

action_space = [0, 1, 2, 3]

action_size = len(action_space)

state_size = [3]

STEP_RANGE = 300

if __name__ == '__main__':
    argParser = argparse.ArgumentParser()
    argParser.add_argument('-a', '--agent', default='A2C',  help='Specify the Agent you want to use to test [A2C]')
    argParser.add_argument('-m', '--modelActor', required=True, help='Model actor to test')
    argParser.add_argument('-e', '--episode', type=int, default=1, help='Number of episode to test')
    args = argParser.parse_args()

    env = SnakeEnv()

    match args.agent:
        case 'A2C':
            agent = A2CAgent(action_space, action_size, state_size)

    if args.modelActor:
        agent.load_test(args.modelActor)
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
            # Normalize values between 0 and 1
            state = state/24.0

            state = tf.convert_to_tensor(state)
            state = tf.expand_dims(state, 0)

            action_probs = agent.actor_model(state)

            action = np.random.choice(action_size, p=np.squeeze(action_probs))

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

            if done or total_reward < 0 or neg_consec_reward == 30:
                print("episode: {}/{}, tot_reward: {}, fruits: {}, time alive: {}"
                    .format(e, episode_range, total_reward, n_fruits, step+1))
                break

        # Episode reached the max number of steps
        if step+1==STEP_RANGE:
            print("Max number of steps reached!")
            print("episode: {}/{}, tot_reward: {}, fruits: {}, time alive: {}"
                    .format(e, episode_range, total_reward, n_fruits, step+1))
