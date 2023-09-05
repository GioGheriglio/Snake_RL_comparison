import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from collections import deque
import random

class DQNAgent:
    def __init__(self, action_space, action_size, state_size, epsilon):
        self.action_space = action_space
        self.action_size = action_size
        self.state_size = state_size
        self.memory = deque(maxlen=5000)
        self.gamma = 0.95  # discount factor
        self.epsilon = epsilon  # initial exploration (should start from 1.0)
        self.epsilon_decay = 0.9995  # exploration decay
        self.epsilon_min = 0.01  # exploration minimum value
        self.learning_rate = 0.001  # learning rate
        self.optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.loss_fn = keras.losses.mean_squared_error
        self.model = self._build_model()
        self.target_model = self._build_model()

    def _build_model(self):
        model = Sequential([
            keras.layers.Dense(32, activation='relu', input_shape=self.state_size),
            keras.layers.Dense(32, activation="relu"),
            keras.layers.Dense(self.action_size, activation="linear")
        ])
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            action_index = self.action_space[random.randrange(self.action_size)]
        else:
            # Normalize values between 0 and 1
            state = state/24.0

            state = tf.convert_to_tensor(state)
            state = tf.expand_dims(state, 0)

            act_values = self.model(state)
            action_index = np.argmax(act_values[0])

        return self.action_space[action_index]

    def replay(self, batch_size):
        minibatch = self.sample_experiences(batch_size)
        states, actions, rewards, next_states, dones = minibatch

        # Normalize values between 0 and 1
        next_states = next_states/24.0
        states = states/24.0

        next_states = tf.convert_to_tensor(next_states)
        states = tf.convert_to_tensor(states)

        next_Q_values = self.target_model(next_states)
        max_next_Q_values = np.max(next_Q_values, axis=1)
        target_Q_values = (rewards + (1-dones)*self.gamma*max_next_Q_values)

        mask = tf.one_hot(actions, self.action_size)
        with tf.GradientTape() as tape:
            all_Q_values = self.model(states)
            Q_values = tf.reduce_sum(tf.multiply(all_Q_values, mask), axis=1)
            loss = self.loss_fn(target_Q_values,Q_values)

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def sample_experiences(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = [np.array([experience[field_index] for experience in batch])
        for field_index in range(5)]
        return states, actions, rewards, next_states, dones

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

    def update(self):
        self.target_model.set_weights(self.model.get_weights())