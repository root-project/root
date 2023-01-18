import unittest
import ROOT

import numpy as np
from numpy.testing import assert_almost_equal
import graph_nets as gn
from graph_nets import utils_tf
import sonnet as snt


GLOBAL_FEATURE_SIZE = 2
NODE_FEATURE_SIZE = 2
EDGE_FEATURE_SIZE = 2


# for generating input data for graph nets, 
# from https://github.com/deepmind/graph_nets/blob/master/graph_nets/demos/graph_nets_basics.ipynb
def get_graph_data_dict(num_nodes, num_edges):
  return {
      "globals": np.random.rand(GLOBAL_FEATURE_SIZE).astype(np.float32),
      "nodes": np.random.rand(num_nodes, NODE_FEATURE_SIZE).astype(np.float32),
      "edges": np.random.rand(num_edges, EDGE_FEATURE_SIZE).astype(np.float32),
      "senders": np.random.randint(num_nodes, size=num_edges, dtype=np.int32),
      "receivers": np.random.randint(num_nodes, size=num_edges, dtype=np.int32),
  }


class SOFIE_GNN(unittest.TestCase):
    """
    Tests for the pythonizations of ParseFromMemory method of SOFIE GNN. 
    """

    def test_parse_gnn(self):
        '''
        Test that parsed GNN model from a graphnets model generates correct 
        inference code
        '''
        GraphModule = gn.modules.GraphNetwork(
            edge_model_fn=lambda: snt.nets.MLP([2,2], activate_final=True),
            node_model_fn=lambda: snt.nets.MLP([2,2], activate_final=True),
            global_model_fn=lambda: snt.nets.MLP([2,2], activate_final=True))

        GraphData = get_graph_data_dict(2,1)
        input_graphs = utils_tf.data_dicts_to_graphs_tuple([GraphData])
        output = GraphModule(input_graphs)
        
        # Parsing model to RModel_GNN
        model = ROOT.TMVA.Experimental.SOFIE.RModel_GNN.ParseFromMemory(GraphModule, GraphData)
        model.Generate()
        model.OutputGenerated()

        ROOT.gInterpreter.Declare('#include "gnn_network.hxx"')
        input_data = ROOT.TMVA.Experimental.SOFIE.GNN_Data()

        input_data.node_data = ROOT.TMVA.Experimental.AsRTensor(GraphData['nodes'])
        input_data.edge_data = ROOT.TMVA.Experimental.AsRTensor(GraphData['edges'])
        input_data.global_data = ROOT.TMVA.Experimental.AsRTensor(GraphData['globals'])

        ROOT.TMVA_SOFIE_gnn_network.infer(input_data)
        
        output_node_data = output.nodes.numpy()
        output_edge_data = output.edges.numpy()
        output_global_data = output.globals.numpy().flatten()

        assert_almost_equal(output_node_data, np.asarray(input_data.node_data))
        assert_almost_equal(output_edge_data, np.asarray(input_data.edge_data))
        assert_almost_equal(output_global_data, np.asarray(input_data.global_data))


    def test_parse_graph_independent(self):
        '''
        Test that parsed GraphIndependent model from a graphnets model generates correct 
        inference code
        '''
        GraphModule = gn.modules.GraphIndependent(
            edge_model_fn=lambda: snt.nets.MLP([2,2], activate_final=True),
            node_model_fn=lambda: snt.nets.MLP([2,2], activate_final=True),
            global_model_fn=lambda: snt.nets.MLP([2,2], activate_final=True))
        
        GraphData = get_graph_data_dict(2,1)
        input_graphs = utils_tf.data_dicts_to_graphs_tuple([GraphData])
        output = GraphModule(input_graphs)
        
        # Parsing model to RModel_GraphIndependent
        model = ROOT.TMVA.Experimental.SOFIE.RModel_GraphIndependent.ParseFromMemory(GraphModule, GraphData)
        model.Generate()
        model.OutputGenerated()

        ROOT.gInterpreter.Declare('#include "graph_independent_network.hxx"')
        input_data = ROOT.TMVA.Experimental.SOFIE.GNN_Data()

        input_data.node_data = ROOT.TMVA.Experimental.AsRTensor(GraphData['nodes'])
        input_data.edge_data = ROOT.TMVA.Experimental.AsRTensor(GraphData['edges'])
        input_data.global_data = ROOT.TMVA.Experimental.AsRTensor(GraphData['globals'])

        ROOT.TMVA_SOFIE_graph_independent_network.infer(input_data)
        
        output_node_data = output.nodes.numpy()
        output_edge_data = output.edges.numpy()
        output_global_data = output.globals.numpy().flatten()
        
        assert_almost_equal(output_node_data, np.asarray(input_data.node_data))
        assert_almost_equal(output_edge_data, np.asarray(input_data.edge_data))
        assert_almost_equal(output_global_data, np.asarray(input_data.global_data))


if __name__ == '__main__':
    unittest.main()
