from Agents.A3CMaster import A3CMaster
from Players.AgentPlaceholder import AgentPlaceholder
from Players.randomplayer import RandomPlayer
from agent_test import AgentTest

test = AgentTest(AgentPlaceholder("C:/Users/maipr/Documents/DiplomskaNaloga/Code/UNO_RL/Agents/models/DQN/2021-08-24_01-42/policy_model_180.h5")
                 , RandomPlayer(), 1000, False)
test.start_test()

test = AgentTest(A3CMaster(trained_model_path="C:/Users/maipr/Documents/DiplomskaNaloga/Code/UNO_RL/Agents/models/A3C/model_2021-08-24_03-55.h5")
                 , RandomPlayer(), 1000, False)
test.start_test()