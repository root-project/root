# Authors:
# * Sanjiban Sengupta 01/2023
# * Lorenzo Moneta    01/2023

################################################################################
# Copyright (C) 1995-2020, Rene Brun and Fons Rademakers.                      #
# All rights reserved.                                                         #
#                                                                              #
# For the licensing terms see $ROOTSYS/LICENSE.                                #
# For the list of contributors see $ROOTSYS/README/CREDITS.                    #
################################################################################

from .. import pythonization
from cppyy import gbl as gbl_namespace

def getActivationFunction(model):
    function = model._activation.__name__
    if function == 'relu':
        return gbl_namespace.TMVA.Experimental.SOFIE.Activation.RELU
    else:
        return gbl_namespace.TMVA.Experimental.SOFIE.Activation.Invalid

class RModel_GNN: 
    def ParseFromMemory(GraphModule, GraphData, filename = "gnn_network"):
        gin = gbl_namespace.TMVA.Experimental.SOFIE.GNN_Init()
        gin.num_nodes = len(GraphData['nodes'])

        # extracting the edges
        edges = []
        for i in range(len(GraphData['senders'])):
            val =  gbl_namespace.std.make_pair['int,int'](int(GraphData['receivers'][i]), int(GraphData['senders'][i]))
            gin.edges.push_back(val)

        gin.num_node_features = len(GraphData['nodes'][0])
        gin.num_edge_features = len(GraphData['edges'][0])
        gin.num_global_features = len(GraphData['globals'])

        gin.filename = filename
        
        # adding the node update function
        node_model = GraphModule._node_block._node_model
        if (node_model.name == "mlp"):
            num_layers = len(node_model._layers)
            activation = getActivationFunction(node_model)
            upd = gbl_namespace.TMVA.Experimental.SOFIE.RFunction_MLP(gbl_namespace.TMVA.Experimental.SOFIE.FunctionTarget.NODES, num_layers, activation, node_model._activate_final)
            kernel_tensor_names = gbl_namespace.std.vector['std::string']()
            bias_tensor_names   = gbl_namespace.std.vector['std::string']()

            for i in range(0, 2*num_layers, 2):
                bias_tensor_names.push_back(node_model.variables[i].name)
                kernel_tensor_names.push_back(node_model.variables[i+1].name)
            val = gbl_namespace.std.vector['std::vector<std::string>']()
            val.push_back(kernel_tensor_names)
            val.push_back(bias_tensor_names)
            upd.AddInitializedTensors(val)
            gin.createUpdateFunction(upd)

            weights = node_model.variables
            for i in weights:
                shape = gbl_namespace.std.vector['std::size_t']()
                shape_as_list = i.shape.as_list()
                for j in shape_as_list:
                    shape.push_back(j)
                gin.nodes_update_block.GetFunctionBlock().AddInitializedTensor['float'](i.name, gbl_namespace.TMVA.Experimental.SOFIE.ETensorType.FLOAT, shape, i.numpy())

        else:
            print("Invalid Model for node update.")
            return

        # adding the edge update function
        edge_model = GraphModule._edge_block._edge_model
        if (edge_model.name == "mlp"):
            num_layers = len(edge_model._layers)
            activation = getActivationFunction(edge_model)
            upd = gbl_namespace.TMVA.Experimental.SOFIE.RFunction_MLP(gbl_namespace.TMVA.Experimental.SOFIE.FunctionTarget.EDGES, num_layers, activation, edge_model._activate_final)
            kernel_tensor_names = gbl_namespace.std.vector['std::string']()
            bias_tensor_names   = gbl_namespace.std.vector['std::string']()

            for i in range(0, 2*num_layers, 2):
                bias_tensor_names.push_back(edge_model.variables[i].name)
                kernel_tensor_names.push_back(edge_model.variables[i+1].name)
            val = gbl_namespace.std.vector['std::vector<std::string>']()
            val.push_back(kernel_tensor_names)
            val.push_back(bias_tensor_names)
            upd.AddInitializedTensors(val)
            gin.createUpdateFunction(upd)

            weights = edge_model.variables
            for i in weights:
                shape = gbl_namespace.std.vector['std::size_t']()
                shape_as_list = i.shape.as_list()
                for j in shape_as_list:
                    shape.push_back(j)
                gin.edges_update_block.GetFunctionBlock().AddInitializedTensor['float'](i.name, gbl_namespace.TMVA.Experimental.SOFIE.ETensorType.FLOAT, shape, i.numpy())

        else:
            print("Invalid Model for edge update.")
            return

        # adding the global update function
        global_model = GraphModule._global_block._global_model
        if (global_model.name == "mlp"):
            num_layers = len(global_model._layers)
            activation = getActivationFunction(global_model)
            upd = gbl_namespace.TMVA.Experimental.SOFIE.RFunction_MLP(gbl_namespace.TMVA.Experimental.SOFIE.FunctionTarget.GLOBALS, num_layers, activation, global_model._activate_final)
            kernel_tensor_names = gbl_namespace.std.vector['std::string']()
            bias_tensor_names   = gbl_namespace.std.vector['std::string']()

            for i in range(0, 2*num_layers, 2):
                bias_tensor_names.push_back(global_model.variables[i].name)
                kernel_tensor_names.push_back(global_model.variables[i+1].name)
            val = gbl_namespace.std.vector['std::vector<std::string>']()
            val.push_back(kernel_tensor_names)
            val.push_back(bias_tensor_names)
            upd.AddInitializedTensors(val)
            gin.createUpdateFunction(upd)

            weights = global_model.variables
            for i in weights:
                shape = gbl_namespace.std.vector['std::size_t']()
                shape_as_list = i.shape.as_list()
                for j in shape_as_list:
                    shape.push_back(j)
                gin.globals_update_block.GetFunctionBlock().AddInitializedTensor['float'](i.name, gbl_namespace.TMVA.Experimental.SOFIE.ETensorType.FLOAT, shape, i.numpy())
        else:
            print("Invalid Model for global update.")
            return

        # adding edge-node aggregate function
        edge_node_reducer = GraphModule._node_block._received_edges_aggregator._reducer.__qualname__
        if(edge_node_reducer == "unsorted_segment_sum"):
            agg = gbl_namespace.TMVA.Experimental.SOFIE.RFunction_Sum()
            gin.createAggregateFunction[gbl_namespace.TMVA.Experimental.SOFIE.RFunction_Sum](agg, gbl_namespace.TMVA.Experimental.SOFIE.FunctionRelation.NODES_EDGES)
        elif(edge_node_reducer == "unsorted_segment_mean"):
            agg = gbl_namespace.TMVA.Experimental.SOFIE.RFunction_Mean()
            gin.createAggregateFunction[gbl_namespace.TMVA.Experimental.SOFIE.RFunction_Mean](agg, gbl_namespace.TMVA.Experimental.SOFIE.FunctionRelation.NODES_EDGES)
        else:
            print("Invalid aggregate function for edge-node reduction")
            return


        # adding node-global aggregate function
        node_global_reducer = GraphModule._global_block._nodes_aggregator._reducer.__qualname__
        if(node_global_reducer == "unsorted_segment_sum"):
            agg = gbl_namespace.TMVA.Experimental.SOFIE.RFunction_Sum()
            gin.createAggregateFunction[gbl_namespace.TMVA.Experimental.SOFIE.RFunction_Sum](agg, gbl_namespace.TMVA.Experimental.SOFIE.FunctionRelation.NODES_GLOBALS)
        elif(node_global_reducer == "unsorted_segment_mean"):
            agg = gbl_namespace.TMVA.Experimental.SOFIE.RFunction_Mean()
            gin.createAggregateFunction[gbl_namespace.TMVA.Experimental.SOFIE.RFunction_Mean](agg, gbl_namespace.TMVA.Experimental.SOFIE.FunctionRelation.NODES_GLOBALS)
        else:
            print("Invalid aggregate function for node-global reduction")
            return

        # adding edge-global aggregate function
        node_global_reducer = GraphModule._global_block._edges_aggregator._reducer.__qualname__
        if(node_global_reducer == "unsorted_segment_sum"):
            agg = gbl_namespace.TMVA.Experimental.SOFIE.RFunction_Sum()
            gin.createAggregateFunction[gbl_namespace.TMVA.Experimental.SOFIE.RFunction_Sum](agg, gbl_namespace.TMVA.Experimental.SOFIE.FunctionRelation.EDGES_GLOBALS)
        elif(node_global_reducer == "unsorted_segment_mean"):
            agg = gbl_namespace.TMVA.Experimental.SOFIE.RFunction_Mean()
            gin.createAggregateFunction[gbl_namespace.TMVA.Experimental.SOFIE.RFunction_Mean](agg, gbl_namespace.TMVA.Experimental.SOFIE.FunctionRelation.EDGES_GLOBALS)
        else:
            print("Invalid aggregate function for node-global reduction")
            return


        gnn_model = gbl_namespace.TMVA.Experimental.SOFIE.RModel_GNN(gin)
        blas_routines = gbl_namespace.std.vector['std::string']()
        blas_routines.push_back("Gemm")
        gnn_model.AddBlasRoutines(blas_routines)
        return gnn_model

class RModel_GraphIndependent:
    def ParseFromMemory(GraphModule, GraphData, filename = "graph_independent_network"):
        gin = gbl_namespace.TMVA.Experimental.SOFIE.GraphIndependent_Init()
        gin.num_nodes = len(GraphData['nodes'])

        # extracting the edges
        edges = []
        for i in range(len(GraphData['senders'])):
            val =  gbl_namespace.std.make_pair['int,int'](int(GraphData['receivers'][i]), int(GraphData['senders'][i]))
            gin.edges.push_back(val)

        gin.num_node_features = len(GraphData['nodes'][0])
        gin.num_edge_features = len(GraphData['edges'][0])
        gin.num_global_features = len(GraphData['globals'])

        gin.filename = filename

        # adding the node update function
        node_model = GraphModule._node_model._model
        if (node_model.name == "mlp"):
            num_layers = len(node_model._layers)
            activation = getActivationFunction(node_model)
            upd = gbl_namespace.TMVA.Experimental.SOFIE.RFunction_MLP(gbl_namespace.TMVA.Experimental.SOFIE.FunctionTarget.NODES, num_layers, activation, node_model._activate_final, gbl_namespace.TMVA.Experimental.SOFIE.GraphType.GraphIndependent)
            kernel_tensor_names = gbl_namespace.std.vector['std::string']()
            bias_tensor_names   = gbl_namespace.std.vector['std::string']()

            for i in range(0, 2*num_layers, 2):
                bias_tensor_names.push_back(node_model.variables[i].name)
                kernel_tensor_names.push_back(node_model.variables[i+1].name)
            val = gbl_namespace.std.vector['std::vector<std::string>']()
            val.push_back(kernel_tensor_names)
            val.push_back(bias_tensor_names)
            upd.AddInitializedTensors(val)
            gin.createUpdateFunction(upd)

            weights = node_model.variables
            for i in weights:
                shape = gbl_namespace.std.vector['std::size_t']()
                shape_as_list = i.shape.as_list()
                for j in shape_as_list:
                    shape.push_back(j)
                gin.nodes_update_block.GetFunctionBlock().AddInitializedTensor['float'](i.name, gbl_namespace.TMVA.Experimental.SOFIE.ETensorType.FLOAT, shape, i.numpy())

        else:
            print("Invalid Model for node update.")
            return

        # adding the edge update function
        edge_model = GraphModule._edge_model._model
        if (edge_model.name == "mlp"):
            num_layers = len(edge_model._layers)
            activation = getActivationFunction(edge_model)
            upd = gbl_namespace.TMVA.Experimental.SOFIE.RFunction_MLP(gbl_namespace.TMVA.Experimental.SOFIE.FunctionTarget.EDGES, num_layers, activation, edge_model._activate_final, gbl_namespace.TMVA.Experimental.SOFIE.GraphType.GraphIndependent)
            kernel_tensor_names = gbl_namespace.std.vector['std::string']()
            bias_tensor_names   = gbl_namespace.std.vector['std::string']()

            for i in range(0, 2*num_layers, 2):
                bias_tensor_names.push_back(edge_model.variables[i].name)
                kernel_tensor_names.push_back(edge_model.variables[i+1].name)
            val = gbl_namespace.std.vector['std::vector<std::string>']()
            val.push_back(kernel_tensor_names)
            val.push_back(bias_tensor_names)
            upd.AddInitializedTensors(val)
            gin.createUpdateFunction(upd)

            weights = edge_model.variables
            for i in weights:
                shape = gbl_namespace.std.vector['std::size_t']()
                shape_as_list = i.shape.as_list()
                for j in shape_as_list:
                    shape.push_back(j)
                gin.edges_update_block.GetFunctionBlock().AddInitializedTensor['float'](i.name, gbl_namespace.TMVA.Experimental.SOFIE.ETensorType.FLOAT, shape, i.numpy())

        else:
            print("Invalid Model for edge update.")
            return

        # adding the global update function
        global_model = GraphModule._global_model._model
        if (global_model.name == "mlp"):
            num_layers = len(global_model._layers)
            activation = getActivationFunction(global_model)
            upd = gbl_namespace.TMVA.Experimental.SOFIE.RFunction_MLP(gbl_namespace.TMVA.Experimental.SOFIE.FunctionTarget.GLOBALS, num_layers, activation, global_model._activate_final, gbl_namespace.TMVA.Experimental.SOFIE.GraphType.GraphIndependent)
            kernel_tensor_names = gbl_namespace.std.vector['std::string']()
            bias_tensor_names   = gbl_namespace.std.vector['std::string']()

            for i in range(0, 2*num_layers, 2):
                bias_tensor_names.push_back(global_model.variables[i].name)
                kernel_tensor_names.push_back(global_model.variables[i+1].name)
            val = gbl_namespace.std.vector['std::vector<std::string>']()
            val.push_back(kernel_tensor_names)
            val.push_back(bias_tensor_names)
            upd.AddInitializedTensors(val)
            gin.createUpdateFunction(upd)

            weights = global_model.variables
            for i in weights:
                shape = gbl_namespace.std.vector['std::size_t']()
                shape_as_list = i.shape.as_list()
                for j in shape_as_list:
                    shape.push_back(j)
                gin.globals_update_block.GetFunctionBlock().AddInitializedTensor['float'](i.name, gbl_namespace.TMVA.Experimental.SOFIE.ETensorType.FLOAT, shape, i.numpy())
        else:
            print("Invalid Model for global update.")
            return

        graph_independent_model = gbl_namespace.TMVA.Experimental.SOFIE.RModel_GraphIndependent(gin)
        blas_routines = gbl_namespace.std.vector['std::string']()
        blas_routines.push_back("Gemm")
        graph_independent_model.AddBlasRoutines(blas_routines)
        return graph_independent_model

@pythonization("RModel_GNN", ns="TMVA::Experimental::SOFIE")
def pythonize_gnn_parse(klass):
    setattr(klass, "ParseFromMemory", RModel_GNN.ParseFromMemory)

@pythonization("RModel_GraphIndependent", ns="TMVA::Experimental::SOFIE")
def pythonize_graph_independent_parse(klass):
    setattr(klass, "ParseFromMemory", RModel_GraphIndependent.ParseFromMemory)

