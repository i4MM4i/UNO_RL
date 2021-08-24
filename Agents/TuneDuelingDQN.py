from ray import tune
import tensorflow as tf
from ray.tune.schedulers import HyperBandForBOHB
from ray.tune.suggest.bayesopt import BayesOptSearch
from ray.tune.suggest.bohb import TuneBOHB

from Agents.DeepQNetworkAgent import DeepQNetworkAgent
from Agents.DoubleDeepQNetworkAgent import DoubleQNetworkAgent
from Agents.DuelingQNetworkAgent import DuelingQNetworkAgent
from Players.randomplayer import RandomPlayer
from agent_test import AgentTest


def training_function(config):
    with tf.device('/cpu:0'):
        agent = DuelingQNetworkAgent()
        agent.learning_rate = config["learning_rate"]
        agent.alpha = config["alpha"]
        agent.gamma = config["gamma"]
        agent.delta = config["delta"]
        agent.epsilon_decay = config["epsilon_decay"]
        agent.eta = config["eta"]
        for i in range(10):
            agent.play()

        agent_test = AgentTest(agent, RandomPlayer(), 0)
        reward = agent_test.play()

    tune.report(steps=agent.steps, reward=reward)


bayesopt = BayesOptSearch(metric="reward", mode="max")
analysis = tune.run(
    training_function,
    config={
        "learning_rate": tune.loguniform(1e-3, 1e-1),
        "alpha": tune.loguniform(1e-3, 1e-1),
        "gamma": tune.loguniform(1e-3, 1e-1),
        "delta": tune.loguniform(0.1, 0.5),
        "epsilon_decay": tune.uniform(0.9999, 0.999999),
        "eta": tune.loguniform(1e-3, 1e-1)
    },
    local_dir="/DuelingDQN",
    search_alg=bayesopt,
    num_samples=100)

print("Best config: ", analysis.get_best_config(
    metric="reward", mode="max"))

# Get a dataframe for analyzing trial results.
df = analysis.results_df