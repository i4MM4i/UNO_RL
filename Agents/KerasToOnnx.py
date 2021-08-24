import os
import keras2onnx

from Agents.A3CMaster import A3CMaster
from Agents.A3CWorker import A3CWorker
from Agents.DuelingQNetworkAgent import DuelingQNetworkAgent

agent = DuelingQNetworkAgent()
agent.play()
onnx_model = keras2onnx.convert_keras(agent.policy_network, "Dueling DQN network")
model_file = 'modelDuelingDQN.onnx'
file = open(model_file, "wb")
file.write(onnx_model.SerializeToString())
print(file.name)
file.close()
'''
#keras2onnx.save_model()export_tf_frozen_graph(onnx_model)#.save_model(onnx_model, temp_model_file)
agent = A3CMaster()
onnx_model = keras2onnx.convert_keras(agent.master_model, "A3C network")
model_file = 'A3C.onnx'
file = open(model_file, "wb")
file.write(onnx_model.SerializeToString())
print(file.name)
file.close()
'''