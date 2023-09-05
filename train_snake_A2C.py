import numpy as np
import tensorflow as tf
import argparse
from collections import deque
import utility

from snake_env import SnakeEnv
from A2C_snake import A2CAgent

action_space = [0, 1, 2, 3]

action_size = len(action_space)

state_size = [3]

STEP_RANGE = 300

if __name__ == '__main__':
    argParser = argparse.ArgumentParser()
    argParser.add_argument('-a', '--agent', default='A2C',  help='Specify the Agent you want to use to train [A2C]')
    argParser.add_argument('-ma', '--modelActor', help='Model actor to continue the training from')
    argParser.add_argument('-mc', '--modelCritic', help='Model critic to continue the training from')
    argParser.add_argument('-e', '--episode', type=int, default=100000, help='Number of episode to train')
    args = argParser.parse_args()

    env = SnakeEnv()

    match args.agent:
        case 'A2C':
            agent = A2CAgent(action_space, action_size, state_size)

    if args.modelActor and args.modelCritic:
        agent.load(args.modelActor, args.modelCritic)
    if args.episode:
        episode_range = args.episode

    rewards = []
    steps = []
    fruits = []

    action_probs_memory = []
    critic_value_memory = []
    reward_memory = []
    returns = []
    average = []

    last_average_100 = deque(maxlen=100)

    for e in range(1, episode_range+1):
        state = env.reset()

        action_probs_memory.clear()
        critic_value_memory.clear()
        reward_memory.clear()
        returns.clear()

        total_reward = 0
        n_fruits = 0
        neg_consec_reward = 0

        with tf.GradientTape(persistent=True) as tape:
            for step in range(STEP_RANGE):
                # Normalize values between 0 and 1
                state = state/24.0

                state = tf.convert_to_tensor(state)
                state = tf.expand_dims(state, 0)

                action_probs = agent.actor_model(state)
                critic_value = agent.critic_model(state)

                action = np.random.choice(action_size, p=np.squeeze(action_probs))
                
                action_probs_memory.append(tf.math.log(action_probs[0, action]))
                critic_value_memory.append(critic_value[0, 0])

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

                reward_memory.append(reward)

                state = next_state

                if done or total_reward < 0 or neg_consec_reward == 30:
                    print("episode: {}/{}, tot_reward: {}, fruits: {}, time alive: {}"
                        .format(e, episode_range, total_reward, n_fruits, step+1))

                    break

            # Episode reached the max number of steps
            if step+1==STEP_RANGE:
                print("Max number of steps reached!")
                print("episode: {}/{}, tot_reward: {}, fruits: {}, time alive: {}"
                        .format(e, episode_range, total_reward, n_fruits, step+1))

            discounted_sum = 0
            for rew in reward_memory[::-1]:
                discounted_sum = rew + agent.gamma * discounted_sum
                returns.append(discounted_sum)

            returns = returns[::-1]

            # Compute the advantage
            advantage = np.subtract(returns, critic_value_memory)

            # Compute actor loss
            actor_loss = -tf.math.reduce_sum(tf.multiply(action_probs_memory, advantage))

            # Compute critic loss
            critic_loss = agent.loss_fn(returns, critic_value_memory)

        actor_grads = tape.gradient(actor_loss, agent.actor_model.trainable_variables)
        agent.actor_optimizer.apply_gradients(zip(actor_grads, agent.actor_model.trainable_variables))

        critic_grads = tape.gradient(critic_loss, agent.critic_model.trainable_variables)
        agent.critic_optimizer.apply_gradients(zip(critic_grads, agent.critic_model.trainable_variables))

        rewards.append(total_reward)
        steps.append(step+1)
        fruits.append(n_fruits)

        last_average_100.append(total_reward)
        average.append(np.average(last_average_100))

        if e % 10000 == 0:
            agent.save("./Models/A2C/snake-{}-actor.h5".format(e), "./Models/A2C/snake-{}-critic.h5".format(e))
            episodes = np.arange(1, e+1)
            utility.plot_results(episodes, rewards, steps, average, args.agent)

    if e == episode_range:
        print()
        print("End of the episode range!")
        agent.save("./Models/A2C/snake-{}-actor.h5".format(e), "./Models/A2C/snake-{}-critic.h5".format(e))
        episodes = np.arange(1, e+1)
        utility.plot_results(episodes, rewards, steps, average, args.agent)