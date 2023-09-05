from tensorflow import keras
from keras.models import Sequential

class A2CAgent:
    def __init__(self, action_space, action_size, state_size):
        self.action_space = action_space
        self.action_size = action_size
        self.state_size = state_size
        self.gamma = 0.95  # discount factor
        self.learning_rate = 0.001  # learning rate
        self.actor_optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.critic_optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.loss_fn = keras.losses.mean_squared_error
        self.actor_model = self._build_actor()
        self.critic_model = self._build_critic()

    def _build_actor(self):
        model = Sequential([
            keras.layers.Dense(32, activation='relu', input_shape=self.state_size),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(self.action_size, activation="softmax")
        ])
        return model

    def _build_critic(self):
        model = Sequential([
            keras.layers.Dense(32, activation='relu', input_shape=self.state_size),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(1, activation="linear")
        ])
        return model

    def load(self, actor_name, critic_name):
        self.actor_model.load_weights(actor_name)
        self.critic_model.load_weights(critic_name)

    def load_test(self, actor_name):
        self.actor_model.load_weights(actor_name)

    def save(self, actor_name, critic_name):
        self.actor_model.save_weights(actor_name)
        self.critic_model.save_weights(critic_name)

