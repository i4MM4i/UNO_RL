from tqdm import tqdm

from Environment.environment import UnoEnvironment
from Environment.player import Player


class AgentTest:
    def __init__(self, agent: Player, opponent: Player, n_rounds=100, display=False):
        self.agent = agent
        self.agent_wins = 0
        self.opponent_wins = 0
        self.opponent = opponent
        self.n_rounds = n_rounds
        self.env = UnoEnvironment(display)
        self.tqdm_rounds = tqdm((range(self.n_rounds)))

    def start_test(self):
        for rounds in self.tqdm_rounds:
            self.play()
            self.tqdm_rounds.set_description("Agent wins: " + str(self.agent.wins) +
                                             " Opponent wins: " + str(self.opponent.wins))

    def play(self):
        state = self.env.reset(self.agent, self.opponent)
        done = False
        episode_reward = 0
        while not done:
            legal_actions = self.env.get_legal_actions()
            action = self.agent.get_action(legal_actions, state)
            next_state, reward, done = self.env.step_with_opp_step(action)
            if done:
                if reward == 1:
                    self.agent.wins += 1
                else:
                    self.opponent.wins += 1
            episode_reward += reward
            state = next_state
        return episode_reward


