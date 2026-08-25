## \file
## \ingroup tutorial_ml
## \notebook -nodraw
##
## Tutorial showing inference of a Graph Neural Network with SOFIE.
##
## A graph network model following DeepMind's Encode-Process-Decode architecture
## (see arXiv:1806.01261) is defined in PyTorch and exported to ONNX. The ONNX
## models are then parsed with the SOFIE ONNX parser, C++ inference code is
## generated and compiled, and its output is validated against PyTorch.
##
## \macro_code
##
## \author

import time

import numpy as np
import ROOT
import torch
import torch.nn as nn

# defining graph properties
num_nodes = 5
num_edges = 20
snd = np.array([1, 2, 3, 4, 2, 3, 4, 3, 4, 4, 0, 0, 0, 0, 1, 1, 1, 2, 2, 3], dtype="int64")
rec = np.array([0, 0, 0, 0, 1, 1, 1, 2, 2, 3, 1, 2, 3, 4, 2, 3, 4, 3, 4, 4], dtype="int64")
node_size = 4
edge_size = 4
global_size = 1
LATENT_SIZE = 100
NUM_LAYERS = 4
processing_steps = 5
numevts = 40

torch.manual_seed(42)
torch.set_grad_enabled(False)


# method for returning dictionary of graph data
def get_graph_data_dict(num_nodes, num_edges, NODE_FEATURE_SIZE=2, EDGE_FEATURE_SIZE=2, GLOBAL_FEATURE_SIZE=1):
    return {
        "globals": 10 * np.random.rand(1, GLOBAL_FEATURE_SIZE).astype(np.float32) - 5.0,
        "nodes": 10 * np.random.rand(num_nodes, NODE_FEATURE_SIZE).astype(np.float32) - 5.0,
        "edges": 10 * np.random.rand(num_edges, EDGE_FEATURE_SIZE).astype(np.float32) - 5.0,
        "senders": snd,
        "receivers": rec,
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


# Instantiating EncodeProcessDecode Model
ep_model = EncodeProcessDecode()
ep_model.eval()

# Export the four component models to ONNX
sample_indices = (torch.from_numpy(rec), torch.from_numpy(snd))


def export_component(component, name, num_features):
    sample_input = (
        torch.zeros(num_nodes, num_features[0]),
        torch.zeros(num_edges, num_features[1]),
        torch.zeros(1, num_features[2]),
    )
    input_names = ["node_data", "edge_data", "global_data"]
    if isinstance(component, MLPGraphNetwork):
        sample_input += sample_indices
        input_names += ["receivers", "senders"]
    torch.onnx.export(
        component,
        sample_input,
        name + ".onnx",
        input_names=input_names,
        output_names=["node_output", "edge_output", "global_output"],
        dynamo=True,
    )


export_component(ep_model._encoder, "gnn_encoder", (node_size, edge_size, global_size))
export_component(ep_model._core, "gnn_core", (2 * LATENT_SIZE,) * 3)
export_component(ep_model._decoder, "gnn_decoder", (LATENT_SIZE,) * 3)
export_component(ep_model._output_transform, "gnn_output_transform", (LATENT_SIZE,) * 3)

# Parse the ONNX models with SOFIE and generate the C++ inference code
parser = ROOT.TMVA.Experimental.SOFIE.RModelParser_ONNX()
for name in ["gnn_encoder", "gnn_core", "gnn_decoder", "gnn_output_transform"]:
    model = parser.Parse(name + ".onnx")
    model.Generate()
    model.OutputGenerated()

# Compile now the generated C++ code from SOFIE
gen_code = """#pragma cling optimize(2)
#include "gnn_encoder.hxx"
#include "gnn_core.hxx"
#include "gnn_decoder.hxx"
#include "gnn_output_transform.hxx"
"""
ROOT.gInterpreter.Declare(gen_code)


# Build SOFIE GNN Model and run inference
class SofieGNN:
    def __init__(self):
        self.encoder_session = ROOT.TMVA_SOFIE_gnn_encoder.Session()
        self.core_session = ROOT.TMVA_SOFIE_gnn_core.Session()
        self.decoder_session = ROOT.TMVA_SOFIE_gnn_decoder.Session()
        self.output_transform_session = ROOT.TMVA_SOFIE_gnn_output_transform.Session()

    @staticmethod
    def _as_arrays(result, num_nodes, num_edges):
        # a session returns the flat node, edge and global output tensors
        return (
            np.asarray(result[0], dtype=np.float32).reshape(num_nodes, -1),
            np.asarray(result[1], dtype=np.float32).reshape(num_edges, -1),
            np.asarray(result[2], dtype=np.float32).reshape(1, -1),
        )

    def infer(self, graphData):
        n_nodes = len(graphData["nodes"])
        n_edges = len(graphData["edges"])

        def c(x):
            return np.ascontiguousarray(x, dtype=np.float32)

        receivers = np.ascontiguousarray(graphData["receivers"], dtype=np.int64)
        senders = np.ascontiguousarray(graphData["senders"], dtype=np.int64)

        latent = self._as_arrays(
            self.encoder_session.infer(c(graphData["nodes"]), c(graphData["edges"]), c(graphData["globals"])),
            n_nodes, n_edges,
        )
        latent0 = latent
        output_ops = []
        for _ in range(processing_steps):
            core_input = tuple(np.concatenate([a, b], axis=1) for a, b in zip(latent0, latent))
            latent = self._as_arrays(
                self.core_session.infer(c(core_input[0]), c(core_input[1]), c(core_input[2]), receivers, senders),
                n_nodes, n_edges,
            )
            decoded = self._as_arrays(
                self.decoder_session.infer(c(latent[0]), c(latent[1]), c(latent[2])), n_nodes, n_edges
            )
            output_ops.append(
                self._as_arrays(
                    self.output_transform_session.infer(c(decoded[0]), c(decoded[1]), c(decoded[2])),
                    n_nodes, n_edges,
                )
            )
        return output_ops


# Test both GNN on some simulated events
dataSet = [get_graph_data_dict(num_nodes, num_edges, node_size, edge_size, global_size) for i in range(numevts)]


# Function to run the PyTorch model
def RunGNet(graphData):
    return ep_model(
        torch.from_numpy(graphData["nodes"]),
        torch.from_numpy(graphData["edges"]),
        torch.from_numpy(graphData["globals"]),
        torch.from_numpy(graphData["receivers"]),
        torch.from_numpy(graphData["senders"]),
        processing_steps,
    )


start = time.time()
hG = ROOT.TH1D("hG", "Result from PyTorch", 20, 1, 0)
torchOutput = []
for i in range(numevts):
    out = RunGNet(dataSet[i])
    torchOutput.append([[t.numpy() for t in step] for step in out])
    hG.Fill(np.mean(torchOutput[-1][1][2]))

end = time.time()
print("elapsed time for ", numevts, "events = ", end - start)

# running SOFIE-GNN
hS = ROOT.TH1D("hS", "Result from SOFIE", 20, 1, 0)
start0 = time.time()
gnn = SofieGNN()
start = time.time()
print("time to create SOFIE GNN class", start - start0)
sofieOutput = []
for i in range(numevts):
    out = gnn.infer(dataSet[i])
    sofieOutput.append(out)
    hS.Fill(np.mean(out[1][2]))

end = time.time()
print("elapsed time for ", numevts, "events = ", end - start)

c0 = ROOT.TCanvas()
c0.Divide(1, 2)
c1 = c0.cd(1)
c1.Divide(2, 1)
c1.cd(1)
hG.Draw()
c1.cd(2)
hS.Draw()

hDn = ROOT.TH1D("hDn", "Difference for node data", 40, 1, 0)
hDe = ROOT.TH1D("hDe", "Difference for edge data", 40, 1, 0)
hDg = ROOT.TH1D("hDg", "Difference for global data", 40, 1, 0)
# compute differences between SOFIE and PyTorch
maxDifference = 0.0
for i in range(numevts):
    for hist, j in [(hDn, 0), (hDe, 1), (hDg, 2)]:
        difference = sofieOutput[i][1][j] - torchOutput[i][1][j]
        for value in difference.flatten():
            hist.Fill(value)
        maxDifference = max(maxDifference, np.abs(difference).max())

print("maximum difference between SOFIE and PyTorch = ", maxDifference)
if maxDifference > 1e-4:
    raise RuntimeError("SOFIE and PyTorch outputs disagree")

c2 = c0.cd(2)
c2.Divide(3, 1)
c2.cd(1)
hDn.Draw()
c2.cd(2)
hDe.Draw()
c2.cd(3)
hDg.Draw()

c0.Draw()
