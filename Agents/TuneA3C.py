import multiprocessing

from ray import tune
import tensorflow as tf

from Agents.A3CMaster import A3CMaster
from Agents.A3CWorker import A3CWorker
from Agents.DoubleDeepQNetworkAgent import DoubleQNetworkAgent
from Agents.DuelingQNetworkAgent import DuelingQNetworkAgent
from Players.A3CPlayer import A3CPlayer
from Players.randomplayer import RandomPlayer
from agent_test import AgentTest
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# NE DELUJE!!!

def training_function(config):
    master = A3CMaster()
    master.learning_rate = config["learning_rate"]
    master.beta = config["beta"]
    master.gamma = config["gamma"]
    '''worker = A3CWorker(master.master_model, master.optimizer, 0, master.folder,
                             master.beta, master.gamma, opponent_model_path=master.opponent_model_path)
     worker.run()'''
    """a3c_workers = [A3CWorker(master.master_model, master.optimizer, worker_id, master.folder,
                             master.beta, master.gamma, opponent_model_path=master.opponent_model_path)
                   for worker_id in range(2)]
    for i, worker in enumerate(a3c_workers):
        worker.start()
    [worker.join() for worker in a3c_workers]"""
    master.train()
    agent_test = AgentTest(master, RandomPlayer(), 0)
    reward = agent_test.play()
    # del a3c_workers
    del master

    tune.report(reward=reward)

analysis = tune.run(
    training_function,
    config={
        "learning_rate": tune.loguniform(1e-6, 1e-1),
        "beta": tune.loguniform(1e-4, 1),
        "gamma": tune.loguniform(1e-4, 1),
    },
    local_dir="/A3C",
    num_samples=10)

print("Best config: ", analysis.get_best_config(
    metric="reward", mode="max"))

# Get a dataframe for analyzing trial results.
df = analysis.results_df