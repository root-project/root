## \file
## \ingroup tutorial_ml
## \notebook -nodraw
##
## Tutorial parsing a Graph Neural Network from ONNX and generating SOFIE
## inference code.
##
## A graph network model following DeepMind's Encode-Process-Decode architecture
## (see arXiv:1806.01261) is defined in PyTorch and exported to ONNX with
## dynamic node and edge counts. The SOFIE ONNX parser then generates C++
## inference code for the four component networks. The tutorial also generates
## input data, evaluated here with PyTorch as a reference, which serves as
## input for the tutorial TMVA_SOFIE_GNN_Application.C.
##
## \macro_code
##
## \author

import time

import numpy as np
import ROOT
import torch
import torch.nn as nn

# defining graph properties. Number of nodes/edges are the maximum
num_max_nodes = 100
num_max_edges = 300
node_size = 4
edge_size = 4
global_size = 1
LATENT_SIZE = 100
NUM_LAYERS = 4
processing_steps = 5
numevts = 100

torch.manual_seed(42)
torch.set_grad_enabled(False)


# method for returning dictionary of graph data
def get_dynamic_graph_data_dict(NODE_FEATURE_SIZE=2, EDGE_FEATURE_SIZE=2, GLOBAL_FEATURE_SIZE=1):
    num_nodes = np.random.randint(num_max_nodes - 2, size=1)[0] + 2
    num_edges = np.random.randint(num_max_edges - 2, size=1)[0] + 2
    return {
        "globals": 10 * np.random.rand(1, GLOBAL_FEATURE_SIZE).astype(np.float32) - 5.0,
        "nodes": 10 * np.random.rand(num_nodes, NODE_FEATURE_SIZE).astype(np.float32) - 5.0,
        "edges": 10 * np.random.rand(num_edges, EDGE_FEATURE_SIZE).astype(np.float32) - 5.0,
        "senders": np.random.randint(num_nodes, size=num_edges, dtype=np.int64),
        "receivers": np.random.randint(num_nodes, size=num_edges, dtype=np.int64),
    }


# method to instantiate an MLP model to be added in the GNN
# (a stack of Linear+ReLU layers, with a final LayerNorm for the core network)
def make_mlp_model(num_inputs, with_layer_norm=False):
    layers = []
    for _ in range(NUM_LAYERS):
        layers += [nn.Linear(num_inputs, LATENT_SIZE), nn.ReLU()]
        num_inputs = LATENT_SIZE
    if with_layer_norm:
        layers.append(nn.LayerNorm(LATENT_SIZE))
    return nn.Sequential(*layers)


# module applying independent MLPs to the node, edge and global features
class MLPGraphIndependent(nn.Module):
    def __init__(self, num_node_inputs, num_edge_inputs, num_global_inputs):
        super().__init__()
        self.node_fn = make_mlp_model(num_node_inputs)
        self.edge_fn = make_mlp_model(num_edge_inputs)
        self.global_fn = make_mlp_model(num_global_inputs)

    def forward(self, node_data, edge_data, global_data):
        return self.node_fn(node_data), self.edge_fn(edge_data), self.global_fn(global_data)


# module implementing a full graph-network block (see arXiv:1806.01261):
#  - edge update from [edge, receiver node, sender node, global]
#  - node update from [sum of received edges, node, global]
#  - global update from [sum of edges, sum of nodes, global]
class MLPGraphNetwork(nn.Module):
    def __init__(self, num_node_inputs, num_edge_inputs, num_global_inputs):
        super().__init__()
        self.edge_fn = make_mlp_model(num_edge_inputs + 2 * num_node_inputs + num_global_inputs, True)
        self.node_fn = make_mlp_model(LATENT_SIZE + num_node_inputs + num_global_inputs, True)
        self.global_fn = make_mlp_model(2 * LATENT_SIZE + num_global_inputs, True)

    def forward(self, node_data, edge_data, global_data, receivers, senders):
        n_nodes = node_data.shape[0]
        n_edges = edge_data.shape[0]
        edge_input = torch.cat(
            [edge_data, node_data[receivers], node_data[senders], global_data.expand(n_edges, -1)], dim=1
        )
        edge_output = self.edge_fn(edge_input)
        # aggregate the updated edge data per receiving node
        received_edges = torch.zeros(n_nodes, edge_output.shape[1]).scatter_add(
            0, receivers.unsqueeze(1).expand(n_edges, edge_output.shape[1]), edge_output
        )
        node_input = torch.cat([received_edges, node_data, global_data.expand(n_nodes, -1)], dim=1)
        node_output = self.node_fn(node_input)
        global_input = torch.cat(
            [edge_output.sum(0, keepdim=True), node_output.sum(0, keepdim=True), global_data], dim=1
        )
        global_output = self.global_fn(global_input)
        return node_output, edge_output, global_output


# defining a Encode-Process-Decode module for LHCb toy model
class EncodeProcessDecode(nn.Module):
    def __init__(self):
        super().__init__()
        self._encoder = MLPGraphIndependent(node_size, edge_size, global_size)
        self._core = MLPGraphNetwork(2 * LATENT_SIZE, 2 * LATENT_SIZE, 2 * LATENT_SIZE)
        self._decoder = MLPGraphIndependent(LATENT_SIZE, LATENT_SIZE, LATENT_SIZE)
        self._output_transform = MLPGraphIndependent(LATENT_SIZE, LATENT_SIZE, LATENT_SIZE)

    def forward(self, node_data, edge_data, global_data, receivers, senders, num_processing_steps):
        latent = self._encoder(node_data, edge_data, global_data)
        latent0 = latent
        output_ops = []
        for _ in range(num_processing_steps):
            core_input = tuple(torch.cat([a, b], dim=1) for a, b in zip(latent0, latent))
            latent = self._core(*core_input, receivers, senders)
            decoded_op = self._decoder(*latent)
            output_ops.append(self._output_transform(*decoded_op))
        return output_ops


########################################################################################################

# Instantiating EncodeProcessDecode Model
ep_model = EncodeProcessDecode()
ep_model.eval()

# Export the four component models to ONNX, with dynamic node and edge counts
num_nodes_dim = torch.export.Dim("num_nodes", min=2, max=num_max_nodes)
num_edges_dim = torch.export.Dim("num_edges", min=2, max=num_max_edges)


def export_component(component, name, num_features):
    sample_input = (
        torch.zeros(num_max_nodes, num_features[0]),
        torch.zeros(num_max_edges, num_features[1]),
        torch.zeros(1, num_features[2]),
    )
    input_names = ["node_data", "edge_data", "global_data"]
    dynamic_shapes = {
        "node_data": {0: num_nodes_dim},
        "edge_data": {0: num_edges_dim},
        "global_data": None,
    }
    if isinstance(component, MLPGraphNetwork):
        sample_input += (
            torch.randint(num_max_nodes, (num_max_edges,)),
            torch.randint(num_max_nodes, (num_max_edges,)),
        )
        input_names += ["receivers", "senders"]
        dynamic_shapes.update({"receivers": {0: num_edges_dim}, "senders": {0: num_edges_dim}})
    torch.onnx.export(
        component,
        sample_input,
        name + ".onnx",
        input_names=input_names,
        output_names=["node_output", "edge_output", "global_output"],
        dynamic_shapes=dynamic_shapes,
        dynamo=True,
    )


export_component(ep_model._encoder, "encoder", (node_size, edge_size, global_size))
export_component(ep_model._core, "core", (2 * LATENT_SIZE,) * 3)
export_component(ep_model._decoder, "decoder", (LATENT_SIZE,) * 3)
export_component(ep_model._output_transform, "output_transform", (LATENT_SIZE,) * 3)

# Make the SOFIE models: parse the ONNX files and generate the inference code
parser = ROOT.TMVA.Experimental.SOFIE.RModelParser_ONNX()
for name in ["encoder", "core", "decoder", "output_transform"]:
    model = parser.Parse(name + ".onnx")
    model.Generate()
    model.OutputGenerated()
    print("generated SOFIE model", name + ".hxx")

####################################################################################################################################

# generate data and save in a ROOT TTree
fileOut = ROOT.TFile.Open("graph_data.root", "RECREATE")
tree = ROOT.TTree("gdata", "GNN data")

node_data = ROOT.std.vector["float"]()
edge_data = ROOT.std.vector["float"]()
global_data = ROOT.std.vector["float"]()
receivers = ROOT.std.vector["int"]()
senders = ROOT.std.vector["int"]()

tree.Branch("node_data", "std::vector<float>", node_data)
tree.Branch("edge_data", "std::vector<float>", edge_data)
tree.Branch("global_data", "std::vector<float>", global_data)
tree.Branch("receivers", "std::vector<int>", receivers)
tree.Branch("senders", "std::vector<int>", senders)

print("\n\nSaving data in a ROOT File:")
h1 = ROOT.TH1D("h1", "GNN nodes output", 40, 1, 0)
h2 = ROOT.TH1D("h2", "GNN edges output", 40, 1, 0)
h3 = ROOT.TH1D("h3", "GNN global output", 40, 1, 0)
dataset = []
for i in range(numevts):
    graphData = get_dynamic_graph_data_dict(node_size, edge_size, global_size)
    node_data.assign(graphData["nodes"].flatten())
    edge_data.assign(graphData["edges"].flatten())
    global_data.assign(graphData["globals"].flatten())
    receivers.assign(graphData["receivers"].astype(np.int32))
    senders.assign(graphData["senders"].astype(np.int32))
    tree.Fill()
    dataset.append(graphData)

tree.Print()

# evaluate the reference PyTorch model on these events
start = time.time()
for graphData in dataset:
    output_gnn = ep_model(
        torch.from_numpy(graphData["nodes"]),
        torch.from_numpy(graphData["edges"]),
        torch.from_numpy(graphData["globals"]),
        torch.from_numpy(graphData["receivers"]),
        torch.from_numpy(graphData["senders"]),
        processing_steps,
    )
    h1.Fill(np.mean(output_gnn[-1][0].numpy()))
    h2.Fill(np.mean(output_gnn[-1][1].numpy()))
    h3.Fill(np.mean(output_gnn[-1][2].numpy()))

end = time.time()
print("time to evaluate ", numevts, " events", end - start)

c1 = ROOT.TCanvas()
c1.Divide(1, 3)
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
