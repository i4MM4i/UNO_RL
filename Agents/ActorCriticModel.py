from tensorflow.python import keras
from tensorflow.python.keras import layers

# Prirejeno po zgledu: https://blog.tensorflow.org/2018/07/deep-reinforcement-learning-keras-eager-execution.html
class ActorCriticModel(keras.Model):
    def __init__(self, action_size):
        super(ActorCriticModel, self).__init__()
        self.action_size = action_size
        self.dense1 = layers.Dense(140, activation='relu')
        self.policy_logits = layers.Dense(action_size)
        self.dense2 = layers.Dense(140, activation='relu')
        self.values = layers.Dense(1)

    def call(self, inputs):
        x = self.dense1(inputs)
        logits = self.policy_logits(x)
        v1 = self.dense2(inputs)
        values = self.values(v1)
        return logits, values

