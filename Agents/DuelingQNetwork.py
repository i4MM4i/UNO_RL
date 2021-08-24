from tensorflow import keras
import tensorflow as tf

#TODO: A tole dela, k loadaš prek model.load???
class DuelingQNetwork(keras.Model):
    def __init__(self, action_size: int):
        super(DuelingQNetwork, self).__init__()
        self.dense1 = keras.layers.Dense(140, activation='relu')
        self.advantage = keras.layers.Dense(action_size)
        self.value = keras.layers.Dense(1)
        self.lambda_layer = keras.layers.Lambda(lambda x: x - tf.reduce_mean(x))
        self.combine = keras.layers.Add()

    def call(self, input):
        x = self.dense1(input)
        advantage = self.advantage(x)
        value = self.value(x)
        normalized_advantage = self.lambda_layer(advantage)
        combined = self.combine([value, normalized_advantage])
        return combined

