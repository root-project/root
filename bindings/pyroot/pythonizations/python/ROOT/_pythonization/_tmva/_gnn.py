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

def make_mlp_model(gin, model, target, type):
    num_layers = len(model._layers)
    activation = getActivationFunction(model)
    upd = gbl_namespace.TMVA.Experimental.SOFIE.RFunction_MLP(target, num_layers, activation, model._activate_final, type)
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

def add_layer_norm(module_layer, model_block):
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

def add_weights(weights, model_block):
    for i in weights:
        shape = gbl_namespace.std.vector['std::size_t']()
        shape_as_list = i.shape.as_list()
        for j in shape_as_list:
            shape.push_back(j)
        model_block.GetFunctionBlock().AddInitializedTensor['float'](i.name, gbl_namespace.TMVA.Experimental.SOFIE.ETensorType.FLOAT, shape, i.numpy())

def add_aggregate_function(gin, reducer, relation):
    if(reducer == "unsorted_segment_sum"):
        agg = gbl_namespace.TMVA.Experimental.SOFIE.RFunction_Sum()
        gin.createAggregateFunction[gbl_namespace.TMVA.Experimental.SOFIE.RFunction_Sum](agg, relation)
    elif(node_global_reducer == "unsorted_segment_mean"):
        agg = gbl_namespace.TMVA.Experimental.SOFIE.RFunction_Mean()
        gin.createAggregateFunction[gbl_namespace.TMVA.Experimental.SOFIE.RFunction_Mean](agg, relation)
    else:
        raise RuntimeError("Invalid aggregate function for reduction")    



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
        if (node_model.name == 'mlp'):
            make_mlp_model(gin, node_model, gbl_namespace.TMVA.Experimental.SOFIE.FunctionTarget.NODES, gbl_namespace.TMVA.Experimental.SOFIE.GraphType.GNN)
        elif (node_model.name == 'sequential'):
            for i in node_model._layers:
                if(i.name == 'mlp'):
                    make_mlp_model(gin, node_model._layers[0], gbl_namespace.TMVA.Experimental.SOFIE.FunctionTarget.NODES, gbl_namespace.TMVA.Experimental.SOFIE.GraphType.GNN)
                elif(i.name == 'layer_norm'):
                    add_layer_norm(i, gin.nodes_update_block)
                else:
                    print("Invalid Model for node update.")
                    return
        
        else:
            print("Invalid Model for node update.")
            return
        
        add_weights(node_model.variables, gin.nodes_update_block)

        # adding the edge update function
        edge_model = GraphModule._edge_block._edge_model
        if (edge_model.name == 'mlp'):
            make_mlp_model(gin, edge_model, gbl_namespace.TMVA.Experimental.SOFIE.FunctionTarget.EDGES, gbl_namespace.TMVA.Experimental.SOFIE.GraphType.GNN)
        elif (edge_model.name == 'sequential'):
            for i in edge_model._layers:
                if(i.name == 'mlp'):
                    make_mlp_model(gin, edge_model._layers[0], gbl_namespace.TMVA.Experimental.SOFIE.FunctionTarget.EDGES, gbl_namespace.TMVA.Experimental.SOFIE.GraphType.GNN)
                elif(i.name == 'layer_norm'):
                    add_layer_norm(i, gin.edges_update_block)
                else:
                    print("Invalid Model for edge update.")
                    return
        
        else:
            print("Invalid Model for edge update.")
            return

        add_weights(edge_model.variables, gin.edges_update_block)

        # adding the global update function
        global_model = GraphModule._global_block._global_model
        if (global_model.name == 'mlp'):
            make_mlp_model(gin, global_model, gbl_namespace.TMVA.Experimental.SOFIE.FunctionTarget.GLOBALS, gbl_namespace.TMVA.Experimental.SOFIE.GraphType.GNN)
        elif (global_model.name == 'sequential'):
            for i in global_model._layers:
                if(i.name == 'mlp'):
                    make_mlp_model(gin, global_model._layers[0], gbl_namespace.TMVA.Experimental.SOFIE.FunctionTarget.GLOBALS, gbl_namespace.TMVA.Experimental.SOFIE.GraphType.GNN)
                elif(i.name == 'layer_norm'):
                    add_layer_norm(i, gin.globals_update_block)
                else:
                    print("Invalid Model for global update.")
                    return
        
        else:
            print("Invalid Model for global update.")
            return

        add_weights(global_model.variables, gin.globals_update_block)

        # adding edge-node aggregate function
        add_aggregate_function(gin, GraphModule._node_block._received_edges_aggregator._reducer.__qualname__, gbl_namespace.TMVA.Experimental.SOFIE.FunctionRelation.NODES_EDGES)

        # adding node-global aggregate function
        add_aggregate_function(gin, GraphModule._global_block._nodes_aggregator._reducer.__qualname__, gbl_namespace.TMVA.Experimental.SOFIE.FunctionRelation.NODES_GLOBALS)

        # adding edge-global aggregate function
        add_aggregate_function(gin, GraphModule._global_block._edges_aggregator._reducer.__qualname__, gbl_namespace.TMVA.Experimental.SOFIE.FunctionRelation.EDGES_GLOBALS)

        gnn_model = gbl_namespace.TMVA.Experimental.SOFIE.RModel_GNN(gin)
        blas_routines = gbl_namespace.std.vector['std::string']()
        blas_routines.push_back("Gemm")
        blas_routines.push_back("Axpy")
        blas_routines.push_back("Gemv")
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
        if (node_model.name == 'mlp'):
            make_mlp_model(gin, node_model, gbl_namespace.TMVA.Experimental.SOFIE.FunctionTarget.NODES, gbl_namespace.TMVA.Experimental.SOFIE.GraphType.GraphIndependent)
        elif (node_model.name == 'sequential'):
            for i in node_model._layers:
                if(i.name == 'mlp'):
                    make_mlp_model(gin, node_model._layers[0], gbl_namespace.TMVA.Experimental.SOFIE.FunctionTarget.NODES, gbl_namespace.TMVA.Experimental.SOFIE.GraphType.GraphIndependent)
                elif(i.name == 'layer_norm'):
                    add_layer_norm(i, gin.nodes_update_block)
                else:
                    print("Invalid Model for node update.")
                    return
        
        else:
            print("Invalid Model for node update.")
            return
        
        add_weights(node_model.variables, gin.nodes_update_block)


        # adding the edge update function
        edge_model = GraphModule._edge_model._model
        if (edge_model.name == 'mlp'):
            make_mlp_model(gin, edge_model, gbl_namespace.TMVA.Experimental.SOFIE.FunctionTarget.EDGES, gbl_namespace.TMVA.Experimental.SOFIE.GraphType.GraphIndependent)
        elif (edge_model.name == 'sequential'):
            for i in edge_model._layers:
                if(i.name == 'mlp'):
                    make_mlp_model(gin, edge_model._layers[0], gbl_namespace.TMVA.Experimental.SOFIE.FunctionTarget.EDGES, gbl_namespace.TMVA.Experimental.SOFIE.GraphType.GraphIndependent)
                elif(i.name == 'layer_norm'):
                    add_layer_norm(i, gin.edges_update_block)
                else:
                    print("Invalid Model for edge update.")
                    return
        
        else:
            print("Invalid Model for edge update.")
            return
        
        add_weights(edge_model.variables, gin.edges_update_block)


        # adding the global update function
        global_model = GraphModule._global_model._model
        if (global_model.name == 'mlp'):
            make_mlp_model(gin, global_model, gbl_namespace.TMVA.Experimental.SOFIE.FunctionTarget.GLOBALS, gbl_namespace.TMVA.Experimental.SOFIE.GraphType.GraphIndependent)
        elif (global_model.name == 'sequential'):
            for i in global_model._layers:
                if(i.name == 'mlp'):
                    make_mlp_model(gin, global_model._layers[0], gbl_namespace.TMVA.Experimental.SOFIE.FunctionTarget.GLOBALS, gbl_namespace.TMVA.Experimental.SOFIE.GraphType.GraphIndependent)
                elif(i.name == 'layer_norm'):
                    add_layer_norm(i, gin.globals_update_block)
                else:
                    print("Invalid Model for global update.")
                    return
        
        else:
            print("Invalid Model for global update.")
            return

        add_weights(global_model.variables, gin.globals_update_block)

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

