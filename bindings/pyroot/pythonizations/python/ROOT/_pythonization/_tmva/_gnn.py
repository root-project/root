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
import sys
from cppyy import gbl as gbl_namespace


def getActivationFunction(model):
    """
    Get the activation function for the model.

    Parameters:
        model: The graph_nets' component model to extract the activation function from.
               The component model can be either of the update functions for
               nodes, edges or globals.

    Returns:
        The activation function enum value.
    """
    function = model._activation.__name__
    if function == 'relu':
        return gbl_namespace.TMVA.Experimental.SOFIE.Activation.RELU
    else:
        return gbl_namespace.TMVA.Experimental.SOFIE.Activation.Invalid

def make_mlp_model(gin, model, function_target, type):
    """
    Create an MLP model and add it to the GNN Initializer.

    Parameters:
        gin: The GNN Initializer to which the model will be added.
        model: The model extracted from graph_nets's GNN component
        function_target: Target for the function to update either of nodes, edges or globals
        graph_type: The type of the graph, i.e. GNN or GraphIndependent

    """
    num_layers = len(model._layers)
    activation = getActivationFunction(model)
    upd = gbl_namespace.TMVA.Experimental.SOFIE.RFunction_MLP(function_target, num_layers, activation, model._activate_final, type)
    kernel_tensor_names = gbl_namespace.std.vector['std::string']()
    bias_tensor_names   = gbl_namespace.std.vector['std::string']()

    for i in range(0, 2*num_layers, 2):
        bias_tensor_names.push_back(model.variables[i].name)
        kernel_tensor_names.push_back(model.variables[i+1].name)
    val = gbl_namespace.std.vector['std::vector<std::string>']()
    val.push_back(kernel_tensor_names)
    val.push_back(bias_tensor_names)
    upd.AddInitializedTensors(val)
    gin.createUpdateFunction(upd)

def make_linear_model(gin, model, function_target, type):
    """
    Create an Linear model and add it to the GNN Initializer.

    Parameters:
        gin: The GNN Initializer to which the model will be added.
        model: The model extracted from graph_nets's GNN component
        function_target: Target for the function to update either of nodes, edges or globals
        graph_type: The type of the graph, i.e. GNN or GraphIndependent

    """
    activation = gbl_namespace.TMVA.Experimental.SOFIE.Activation.Invalid
    upd = gbl_namespace.TMVA.Experimental.SOFIE.RFunction_MLP(function_target, 1, activation, False, type)
    kernel_tensor_names = gbl_namespace.std.vector['std::string'](1)
    bias_tensor_names   = gbl_namespace.std.vector['std::string'](1)

    if (len(model.variables) == 1) :
       kernel_tensor_names[0] = model.variables[0].name
    else :
       bias_tensor_names[0] = model.variables[0].name
       kernel_tensor_names[0] = model.variables[1].name

    val = gbl_namespace.std.vector['std::vector<std::string>']()
    val.push_back(kernel_tensor_names)
    val.push_back(bias_tensor_names)
    upd.AddInitializedTensors(val)
    gin.createUpdateFunction(upd)

def add_layer_norm(gin, module_layer, function_target):
    """
    Add a LayerNormalization operator to the particular function target
    in the Graph Initializer

    Parameters:
        gin: The GNN Initializer to which the LayerNorm operator will be added
        module_layer: Extracted LayerNorm from graph_nets' model
        function_target: Target for the function to update either of nodes, edges or globals

    """
    if function_target == gbl_namespace.TMVA.Experimental.SOFIE.FunctionTarget.NODES:
        model_block = gin.nodes_update_block
    elif function_target == gbl_namespace.TMVA.Experimental.SOFIE.FunctionTarget.EDGES:
        model_block = gin.edges_update_block
    else:
        model_block = gin.globals_update_block
    axis = module_layer._axis
    eps  = module_layer._eps
    stash_type = 1
    name_x = model_block.GetFunctionBlock().GetOutputTensorNames()[0]
    name_bias = module_layer.offset.name
    name_scale = module_layer.scale.name
    name_Y = name_x+"output"
    model_block.AddLayerNormalization(axis[0], eps, stash_type, name_x, name_scale, name_bias, name_Y)
    current_output_tensors = model_block.GetFunctionBlock().GetOutputTensorNames()
    new_output_tensors = gbl_namespace.std.vector['std::string']()
    new_output_tensors.push_back(name_Y)
    model_block.GetFunctionBlock().AddOutputTensorNameList(new_output_tensors)

def add_weights(gin, weights, function_target):
    """
    Add weights to respective function targets, either of nodes, edges or globals

    Parameters:
        gin: The GNN Initializer to which the weights will be added
        weights: Weight information, containing the names, shapes and values of initialized tensors
        function_target: Target for the function to update either of nodes, edges or globals

    """
    if function_target == gbl_namespace.TMVA.Experimental.SOFIE.FunctionTarget.NODES:
        model_block = gin.nodes_update_block
    elif function_target == gbl_namespace.TMVA.Experimental.SOFIE.FunctionTarget.EDGES:
        model_block = gin.edges_update_block
    else:
        model_block = gin.globals_update_block

    for i in weights:
        shape = gbl_namespace.std.vector['std::size_t']()
        shape_as_list = i.shape.as_list()
        for j in shape_as_list:
            shape.push_back(j)
        model_block.GetFunctionBlock().AddInitializedTensor['float'](i.name, shape, i.numpy())

def add_aggregate_function(gin, reducer, relation):
    """
    Add aggregate function to the Graph Initializer

    Parameters:
        gin: The GNN Initializer to which the Aggregate function will be added
        reducer: Specifies the means of aggregate, i.e. sum or mean of supplied values
        relation: Specifies the relation of aggregate, i.e. Node-Edge, Global-Edge or Global-Node

    """
    if(reducer == "unsorted_segment_sum"):
        agg = gbl_namespace.TMVA.Experimental.SOFIE.RFunction_Sum()
        gin.createAggregateFunction[gbl_namespace.TMVA.Experimental.SOFIE.RFunction_Sum](agg, relation)
    elif(node_global_reducer == "unsorted_segment_mean"):
        agg = gbl_namespace.TMVA.Experimental.SOFIE.RFunction_Mean()
        gin.createAggregateFunction[gbl_namespace.TMVA.Experimental.SOFIE.RFunction_Mean](agg, relation)
    else:
        raise RuntimeError("Invalid aggregate function for reduction")


def add_update_function(gin, component_model, graph_type, function_target):
    """
    Add update function for respective function target, either of nodes, edges or globals
    based on the supplied component_model

    Parameters:
        gin: The GNN Initializer to which the update function will be added
        component_model: The update function to add, either of MLP or Sequential
        graph_type: The type of the graph, i.e. GNN or GraphIndependent
        function_target: Target for the function to update either of nodes, edges or globals

    """
    if (type(component_model).__name__ == 'MLP'):
        make_mlp_model(gin, component_model, function_target, graph_type)
    elif (type(component_model).__name__ == 'Sequential'):
        for i in component_model._layers:
            if(type(i).__name__ == 'MLP'):
                make_mlp_model(gin, i, function_target, graph_type)
            elif(type(i).__name__ == 'LayerNorm'):
                add_layer_norm(gin, i, function_target)
            else:
                raise RuntimeError("Invalid Model " + type(i).__name__ + " for layer update")
    elif (type(component_model).__name__ == 'Linear'):
        make_linear_model(gin, component_model, function_target, graph_type)
    else:
        raise RuntimeError("Invalid Model " + type(component_model).__name__ + " for update function")
    add_weights(gin, component_model.variables, function_target)



class RModel_GNN:
    """
    Wrapper class for graph_nets' GNN model;s parsing and inference generation

    graph_nets' GNN model comprises of three components, the nodes, edges and globals.
    The entire model and its inference is based on the respective update functions,
    and aggregate function with other components.
    """

    def ParseFromMemory(graph_module, graph_data, filename = "gnn_network"):
        """
        Parse graph_nets' GraphNetwork model and create RModel_GNN.

        Parameters:
            graph_module: The graph module built from graph_nets
            graph_data: Sample graph input data required for parsing of graph_nets' model
                        containing dict with keys: {"globals", "nodes", "edges", "senders", "receivers"}
            filename: The filename to be used for output of inference code.

        Returns:
            An instance of RModel_GNN.
        """
        gin = gbl_namespace.TMVA.Experimental.SOFIE.GNN_Init()
        gin.num_nodes = len(graph_data['nodes'])

        # extracting the edges
        edges = []
        for sender, receiver in zip(graph_data['senders'], graph_data['receivers']):
            gin.edges.push_back(gbl_namespace.std.make_pair['int,int'](int(receiver), int(sender)))

        gin.num_node_features = len(graph_data['nodes'][0])
        gin.num_edge_features = len(graph_data['edges'][0])
        gin.num_global_features = len(graph_data['globals'])

        gin.filename = filename

        # adding the node update function
        node_model = graph_module._node_block._node_model
        add_update_function(gin, node_model, gbl_namespace.TMVA.Experimental.SOFIE.GraphType.GNN,
                                         gbl_namespace.TMVA.Experimental.SOFIE.FunctionTarget.NODES)

        # adding the edge update function
        edge_model = graph_module._edge_block._edge_model
        add_update_function(gin, edge_model, gbl_namespace.TMVA.Experimental.SOFIE.GraphType.GNN,
                                             gbl_namespace.TMVA.Experimental.SOFIE.FunctionTarget.EDGES)

        # adding the global update function
        global_model = graph_module._global_block._global_model
        add_update_function(gin, global_model, gbl_namespace.TMVA.Experimental.SOFIE.GraphType.GNN,
                                             gbl_namespace.TMVA.Experimental.SOFIE.FunctionTarget.GLOBALS)

        # adding edge-node aggregate function
        add_aggregate_function(gin, graph_module._node_block._received_edges_aggregator._reducer.__qualname__, gbl_namespace.TMVA.Experimental.SOFIE.FunctionRelation.NODES_EDGES)

        # adding node-global aggregate function
        add_aggregate_function(gin, graph_module._global_block._nodes_aggregator._reducer.__qualname__, gbl_namespace.TMVA.Experimental.SOFIE.FunctionRelation.NODES_GLOBALS)

        # adding edge-global aggregate function
        add_aggregate_function(gin, graph_module._global_block._edges_aggregator._reducer.__qualname__, gbl_namespace.TMVA.Experimental.SOFIE.FunctionRelation.EDGES_GLOBALS)

        gnn_model = gbl_namespace.TMVA.Experimental.SOFIE.RModel_GNN(gin)
        blas_routines = gbl_namespace.std.vector['std::string']()
        blas_routines.push_back("Gemm")
        blas_routines.push_back("Axpy")
        blas_routines.push_back("Gemv")
        gnn_model.AddBlasRoutines(blas_routines)
        gnn_model.AddNeededStdLib("algorithm")
        gnn_model.AddNeededStdLib("cmath")
        return gnn_model



class RModel_GraphIndependent:
    """
    Wrapper class for graph_nets' GraphIndependent model's parsing and inference generation

    graph_nets' GraphIndependent model is similar to the GNN implementation, with the
    difference being that it has no aggregate function. GraphIndependent is useful
    for independent transformation on the graph data.

    """

    def ParseFromMemory(graph_module, graph_data, filename = "graph_independent_network"):
        """
        Parse graph_nets' GraphIndependent model and create RModel_GraphIndependent.

        Parameters:
            graph_module: The graph module built from graph_nets
            graph_data: Sample graph input data required for parsing of graph_nets' model
                        containing dict with keys: {"globals", "nodes", "edges", "senders", "receivers"}
            filename: The filename to be used for output of inference code.

        Returns:
            An instance of RModel_GraphIndependent.
        """
        gin = gbl_namespace.TMVA.Experimental.SOFIE.GraphIndependent_Init()
        gin.num_nodes = len(graph_data['nodes'])

        # extracting the edges
        edges = []
        for sender, receiver in zip(graph_data['senders'], graph_data['receivers']):
            gin.edges.push_back(gbl_namespace.std.make_pair['int,int'](int(receiver), int(sender)))

        gin.num_node_features = len(graph_data['nodes'][0])
        gin.num_edge_features = len(graph_data['edges'][0])
        gin.num_global_features = len(graph_data['globals'])

        gin.filename = filename


        # adding the node update function
        # when an update is present in graph_nets, the update function has the _model attribute
        # otherwise it is just a simple function defining output = input.
        node_model = graph_module._node_model
        if (hasattr(node_model,"_model")) :
            add_update_function(gin, node_model._model, gbl_namespace.TMVA.Experimental.SOFIE.GraphType.GraphIndependent,
                                         gbl_namespace.TMVA.Experimental.SOFIE.FunctionTarget.NODES)

        # adding the edge update function
        edge_model = graph_module._edge_model
        if (hasattr(edge_model,"_model")) :
            add_update_function(gin, edge_model._model, gbl_namespace.TMVA.Experimental.SOFIE.GraphType.GraphIndependent,
                                             gbl_namespace.TMVA.Experimental.SOFIE.FunctionTarget.EDGES)

        # adding the global update function
        global_model = graph_module._global_model
        if (hasattr(global_model,"_model")) :
            add_update_function(gin, global_model._model, gbl_namespace.TMVA.Experimental.SOFIE.GraphType.GraphIndependent,
                                             gbl_namespace.TMVA.Experimental.SOFIE.FunctionTarget.GLOBALS)

        graph_independent_model = gbl_namespace.TMVA.Experimental.SOFIE.RModel_GraphIndependent(gin)
        blas_routines = gbl_namespace.std.vector['std::string']()
        blas_routines.push_back("Gemm")
        graph_independent_model.AddBlasRoutines(blas_routines)
        graph_independent_model.AddNeededStdLib("algorithm")
        graph_independent_model.AddNeededStdLib("cmath")
        return graph_independent_model


@pythonization("RModel_GNN", ns="TMVA::Experimental::SOFIE")
def pythonize_gnn_parse(klass):
    setattr(klass, "ParseFromMemory", RModel_GNN.ParseFromMemory)

@pythonization("RModel_GraphIndependent", ns="TMVA::Experimental::SOFIE")
def pythonize_graph_independent_parse(klass):
    setattr(klass, "ParseFromMemory", RModel_GraphIndependent.ParseFromMemory)

