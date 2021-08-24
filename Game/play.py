import sys
from Game.game import UnoGame
from Players.AgentPlaceholder import AgentPlaceholder

agent = AgentPlaceholder("C:/Users/maipr/Documents/DiplomskaNaloga/Code/UNO_RL/Agents/models/A3C/model.h5")
game = UnoGame(agent)
