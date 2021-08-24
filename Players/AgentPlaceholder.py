import numpy as np
from tensorflow.keras.models import load_model
from Environment.player import Player


class AgentPlaceholder(Player):
    def __init__(self, model_path=None):
        super().__init__("Agent")
        if model_path is not None:
            self.model = load_model(model_path)

    def get_action(self, legal_actions, state):
        values = self.model.predict(np.array(state).reshape(-1, *state.shape))[0]
        mask = np.full(55, True)
        mask[legal_actions] = False
        values = np.ma.array(values, mask=mask)
        action = np.argmax(values)
        return action
