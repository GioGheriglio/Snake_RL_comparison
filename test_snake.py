import random
from snake_env import SnakeEnv

action_space = [0, 1, 2, 3]

if __name__ == '__main__':
    env = SnakeEnv()

    for i in range(10):
        state = env.reset()
        env.render()

        done = False
        total_reward = 0

        while not done:
            action = action_space[random.randrange(4)]

            next_state, reward, done = env.step(action)
            total_reward += reward

            env.render()
            print('action:', action, 'total reward:', total_reward, 'done:', done)
            print()