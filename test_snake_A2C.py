import numpy as np
import tensorflow as tf
import argparse

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
    argParser.add_argument('-e', '--episode', type=int, default=100, help='Number of episode to test')
    args = argParser.parse_args()

    env = SnakeEnv()

    match args.agent:
        case 'A2C':
            agent = A2CAgent(action_space, action_size, state_size)

    if args.modelActor:
        agent.load_test(args.modelActor)
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

            #env.render()
            #print('action:', action, 'total reward:', total_reward, 'done:', done)
            #print()

            if done or total_reward < 0 or neg_consec_reward == 30:
                print("episode: {}/{}, tot_reward: {}, fruits: {}, time alive: {}"
                    .format(e, episode_range, total_reward, n_fruits, step+1))
                break

        # Episode reached the max number of steps
        if step+1==STEP_RANGE:
            print("Max number of steps reached!")
            print("episode: {}/{}, tot_reward: {}, fruits: {}, time alive: {}"
                    .format(e, episode_range, total_reward, n_fruits, step+1))
            
        rewards.append(total_reward)

    mean = np.mean(rewards)
    max = max(rewards)
    print("Mean value: {}, Max value: {}".format(mean, max))
