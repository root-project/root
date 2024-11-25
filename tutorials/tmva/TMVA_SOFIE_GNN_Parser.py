## Tutorial showing how to parse a GNN from GraphNet and make a SOFIE model
## The tutorial also generate some  data which can serve as input for the tutorial TMVA_SOFIE_GNN_Application.C
import ROOT

import numpy as np
import graph_nets as gn
from graph_nets import utils_tf
import sonnet as snt
#for getting time and memory
import time
import os
import psutil

# defining graph properties. Number of edges/modes are the maximum
num_max_nodes=100
num_max_edges=300
node_size=4
edge_size=4
global_size=1
LATENT_SIZE = 100
NUM_LAYERS = 4
processing_steps = 5
numevts = 100

verbose = False

#print the used memory in MB
def printMemory(s = "") :
    #get memory of current process
    pid = os.getpid()
    python_process = psutil.Process(pid)
    memoryUse = python_process.memory_info()[0]/(1024.*1024.)    #divide by 1024 * 1024 to get memory in MB
    print(s,"memory:",memoryUse,"(MB)")


# method for returning dictionary of graph data
def get_dynamic_graph_data_dict(NODE_FEATURE_SIZE=2, EDGE_FEATURE_SIZE=2, GLOBAL_FEATURE_SIZE=1):
   num_nodes = np.random.randint(num_max_nodes-2, size=1)[0] + 2
   num_edges = np.random.randint(num_max_edges-1, size=1)[0] + 1
   return {
      "globals": 10*np.random.rand(GLOBAL_FEATURE_SIZE).astype(np.float32)-5.,
      "nodes": 10*np.random.rand(num_nodes, NODE_FEATURE_SIZE).astype(np.float32)-5.,
      "edges": 10*np.random.rand(num_edges, EDGE_FEATURE_SIZE).astype(np.float32)-5.,
      "senders": np.random.randint(num_nodes, size=num_edges, dtype=np.int32),
      "receivers": np.random.randint(num_nodes, size=num_edges, dtype=np.int32)
   }

# generate graph data with a fixed number of nodes/edges
def get_fix_graph_data_dict(num_nodes, num_edges, NODE_FEATURE_SIZE=2, EDGE_FEATURE_SIZE=2, GLOBAL_FEATURE_SIZE=1):
   return {
      "globals": np.ones((GLOBAL_FEATURE_SIZE),dtype=np.float32),
      "nodes": np.ones((num_nodes, NODE_FEATURE_SIZE), dtype = np.float32),
      "edges": np.ones((num_edges, EDGE_FEATURE_SIZE), dtype = np.float32),
      "senders":  np.random.randint(num_nodes, size=num_edges, dtype=np.int32),
      "receivers": np.random.randint(num_nodes, size=num_edges, dtype=np.int32)
    }




# method to instantiate mlp model to be added in GNN
def make_mlp_model():
  return snt.Sequential([
      snt.nets.MLP([LATENT_SIZE]*NUM_LAYERS, activate_final=True),
      snt.LayerNorm(axis=-1, create_offset=True, create_scale=True)
  ])

# defining GraphIndependent class with MLP edge, node, and global models.
class MLPGraphIndependent(snt.Module):
  def __init__(self, name="MLPGraphIndependent"):
    super(MLPGraphIndependent, self).__init__(name=name)
    self._network = gn.modules.GraphIndependent(
        edge_model_fn = lambda: snt.nets.MLP([LATENT_SIZE]*NUM_LAYERS, activate_final=True),
        node_model_fn = lambda: snt.nets.MLP([LATENT_SIZE]*NUM_LAYERS, activate_final=True),
        global_model_fn = lambda: snt.nets.MLP([LATENT_SIZE]*NUM_LAYERS, activate_final=True))

  def __call__(self, inputs):
    return self._network(inputs)

# defining Graph network class with MLP edge, node, and global models.
class MLPGraphNetwork(snt.Module):
  def __init__(self, name="MLPGraphNetwork"):
    super(MLPGraphNetwork, self).__init__(name=name)
    self._network = gn.modules.GraphNetwork(
            edge_model_fn=make_mlp_model,
            node_model_fn=make_mlp_model,
            global_model_fn=make_mlp_model)

  def __call__(self, inputs):
    return self._network(inputs)

# defining a Encode-Process-Decode module for LHCb toy model
class EncodeProcessDecode(snt.Module):

  def __init__(self,
               name="EncodeProcessDecode"):
    super(EncodeProcessDecode, self).__init__(name=name)
    self._encoder = MLPGraphIndependent()
    self._core = MLPGraphNetwork()
    self._decoder = MLPGraphIndependent()
    self._output_transform = MLPGraphIndependent()

  def __call__(self, input_op, num_processing_steps):
    latent = self._encoder(input_op)
    latent0 = latent
    output_ops = []
    for _ in range(num_processing_steps):
      core_input = utils_tf.concat([latent0, latent], axis=1)
      latent = self._core(core_input)
      decoded_op = self._decoder(latent)
      output_ops.append(self._output_transform(decoded_op))
    return output_ops


########################################################################################################

# Instantiating EncodeProcessDecode Model

printMemory("before instantiating")
ep_model = EncodeProcessDecode()
printMemory("after instantiating")

# Initializing randomized input data with maximum number of nodes/edges
GraphData = get_fix_graph_data_dict(num_max_nodes, num_max_edges, node_size, edge_size, global_size)

#input_graphs  is a tuple representing the initial data
input_graph_data = utils_tf.data_dicts_to_graphs_tuple([GraphData])

# Initializing randomized input data for core
# note that the core network has as input a double number of features
CoreGraphData = get_fix_graph_data_dict(num_max_nodes, num_max_edges, 2*LATENT_SIZE, 2*LATENT_SIZE, 2*LATENT_SIZE)
input_core_graph_data = utils_tf.data_dicts_to_graphs_tuple([CoreGraphData])

#initialize graph data for decoder (input is LATENT_SIZE)
DecodeGraphData = get_fix_graph_data_dict(num_max_nodes, num_max_edges, LATENT_SIZE, LATENT_SIZE, LATENT_SIZE)

# Make prediction of GNN. This will initialize the GNN with weights
printMemory("before first eval")
output_gn = ep_model(input_graph_data, processing_steps)
printMemory("after first eval")
#print("---> Input:\n",input_graph_data)
#print("\n\n------> Input core data:\n",input_core_graph_data)
#print("\n\n---> Output:\n",output_gn)

# Make SOFIE Model, the model will be made using a maximum number of nodes/edges which are inside GraphData

encoder = ROOT.TMVA.Experimental.SOFIE.RModel_GraphIndependent.ParseFromMemory(ep_model._encoder._network, GraphData, filename = "encoder")
encoder.Generate()
encoder.OutputGenerated()

core = ROOT.TMVA.Experimental.SOFIE.RModel_GNN.ParseFromMemory(ep_model._core._network, CoreGraphData, filename = "core")
core.Generate()
core.OutputGenerated()

decoder = ROOT.TMVA.Experimental.SOFIE.RModel_GraphIndependent.ParseFromMemory(ep_model._decoder._network, DecodeGraphData, filename = "decoder")
decoder.Generate()
decoder.OutputGenerated()

output_transform = ROOT.TMVA.Experimental.SOFIE.RModel_GraphIndependent.ParseFromMemory(ep_model._output_transform._network, DecodeGraphData, filename = "output_transform")
output_transform.Generate()
output_transform.OutputGenerated()

####################################################################################################################################

#generate data and save in a ROOT TTree
#

fileOut = ROOT.TFile.Open("graph_data.root","RECREATE")
tree = ROOT.TTree("gdata","GNN data")
#need to store each element since annot store RTensor

node_data = ROOT.std.vector['float'](num_max_nodes*node_size)
edge_data = ROOT.std.vector['float'](num_max_edges*edge_size)
global_data = ROOT.std.vector['float'](global_size)
receivers =  ROOT.std.vector['int'](num_max_edges)
senders = ROOT.std.vector['int'](num_max_edges)
outgnn = ROOT.std.vector['float'](3)

tree.Branch("node_data", "std::vector<float>" , node_data)
tree.Branch("edge_data", "std::vector<float>" ,  edge_data)
tree.Branch("global_data", "std::vector<float>" ,  global_data)
tree.Branch("receivers", "std::vector<int>" ,  receivers)
tree.Branch("senders", "std::vector<int>" ,  senders)


print("\n\nSaving data in a ROOT File:")
h1 = ROOT.TH1D("h1","GraphNet nodes output",40,1,0)
h2 = ROOT.TH1D("h2","GraphNet edges output",40,1,0)
h3 = ROOT.TH1D("h3","GraphNet global output",40,1,0)
dataset = []
for i in range(0,numevts):
    graphData = get_dynamic_graph_data_dict(node_size, edge_size, global_size)
    s_nodes = graphData['nodes'].size
    s_edges = graphData['edges'].size
    num_edges = graphData['edges'].shape[0]
    tmp = ROOT.std.vector['float'](graphData['nodes'].reshape((graphData['nodes'].size)))
    node_data.assign(tmp.begin(),tmp.end())
    tmp = ROOT.std.vector['float'](graphData['edges'].reshape((graphData['edges'].size)))
    edge_data.assign(tmp.begin(),tmp.end())
    tmp = ROOT.std.vector['float'](graphData['globals'].reshape((graphData['globals'].size)))
    global_data.assign(tmp.begin(),tmp.end())
    #make sure dtype of graphData['receivers'] and senders is int32
    tmp = ROOT.std.vector['int'](graphData['receivers'])
    receivers.assign(tmp.begin(),tmp.end())
    tmp = ROOT.std.vector['int'](graphData['senders'])
    senders.assign(tmp.begin(),tmp.end())
    if (i < 1 and verbose) :
      print("Nodes - shape:",int(node_data.size()/node_size),node_size,"data: ",node_data)
      print("Edges - shape:",num_edges, edge_size,"data: ", edge_data)
      print("Globals : ",global_data)
      print("Receivers : ",receivers)
      print("Senders   : ",senders)
#
#evaluate graph net on these events
#
    tree.Fill()
    tf_graph_data = utils_tf.data_dicts_to_graphs_tuple([graphData])
    dataset.append(tf_graph_data)

tree.Print()

#do a first evaluation
printMemory("before eval1")
output_gnn = ep_model(dataset[0], processing_steps)
printMemory("after eval1")

start = time.time()
firstEvent = True
for tf_graph_data in dataset:
    output_gnn = ep_model(tf_graph_data, processing_steps)
    output_nodes = output_gnn[-1].nodes.numpy()
    output_edges = output_gnn[-1].edges.numpy()
    output_globals = output_gnn[-1].globals.numpy()
    outgnn[0] = np.mean(output_nodes)
    outgnn[1] = np.mean(output_edges)
    outgnn[2] = np.mean(output_globals)
    h1.Fill(outgnn[0])
    h2.Fill(outgnn[1])
    h3.Fill(outgnn[2])
    if (firstEvent and verbose) :
      print("Output of first event")
      print("nodes data", output_gnn[-1].nodes.numpy())
      print("edge data", output_gnn[-1].edges.numpy())
      print("global data", output_gnn[-1].globals.numpy())
      firstEvent = False


end = time.time()

print("time to evaluate events",end-start)
printMemory("after eval Nevts")

c1 = ROOT.TCanvas()
c1.Divide(1,3)
c1.cd(1)
h1.DrawCopy()
c1.cd(2)
h2.DrawCopy()
c1.cd(3)
h3.DrawCopy()

tree.Write()
h1.Write()
h2.Write()
h3.Write()
fileOut.Close()
