import ROOT 


def ParseFromMemory(GraphModule, GraphData):
    gin = ROOT.TMVA.Experimental.SOFIE.GNN_Init()
    gin.num_nodes = len(GraphData['nodes'])

    # extracting the edges
    edges = []
    for i in range(len(GraphData['senders'])):
        val =  ROOT.std.make_pair['int,int'](int(GraphData['senders'][i]), int(GraphData['receivers'][i]))
        gin.edges.push_back(val)

    gin.num_node_features = len(GraphData['nodes'][0])
    gin.num_edge_features = len(GraphData['edges'][0])
    gin.num_global_features = len(GraphData['globals'])

    gin.filename = GraphModule.name 

    # adding the node update function
    node_model = GraphModule._node_block._node_model
    if (node_model.name == "mlp"):
        num_layers = len(node_model._layers)
        upd = ROOT.TMVA.Experimental.SOFIE.RFunction_MLP(ROOT.TMVA.Experimental.SOFIE.FunctionTarget.NODES, num_layers, 0)
        kernel_tensor_names = ROOT.std.vector['std::string']()
        bias_tensor_names   = ROOT.std.vector['std::string']()

        for i in range(0, 2*num_layers, 2):
            print("adding, ", node_model.variables[i].name, node_model.variables[i+1].name)
            bias_tensor_names.push_back(node_model.variables[i].name)
            kernel_tensor_names.push_back(node_model.variables[i+1].name)
        val = ROOT.std.vector['std::vector<std::string>']()
        val.push_back(kernel_tensor_names)
        val.push_back(bias_tensor_names)
        upd.AddInitializedTensors(val)
        gin.createUpdateFunction(upd)
        
        weights = node_model.variables
        for i in weights:
            shape = ROOT.std.vector['std::size_t']()
            shape_as_list = i.shape.as_list()
            for j in shape_as_list:
                shape.push_back(j)
            gin.nodes_update_block.GetFunctionBlock().AddInitializedTensor['float'](i.name, ROOT.TMVA.Experimental.SOFIE.ETensorType.FLOAT, shape, i.numpy())
            
    else:
        print("Invalid Model for node update.")
        return
    
    # adding the edge update function
    edge_model = GraphModule._edge_block._edge_model
    if (edge_model.name == "mlp"):
        num_layers = len(edge_model._layers)
        upd = ROOT.TMVA.Experimental.SOFIE.RFunction_MLP(ROOT.TMVA.Experimental.SOFIE.FunctionTarget.EDGES, num_layers, 0)
        kernel_tensor_names = ROOT.std.vector['std::string']()
        bias_tensor_names   = ROOT.std.vector['std::string']()

        for i in range(0, 2*num_layers, 2):
            print("adding, ", edge_model.variables[i].name, edge_model.variables[i+1].name)
            bias_tensor_names.push_back(edge_model.variables[i].name)
            kernel_tensor_names.push_back(edge_model.variables[i+1].name)
        val = ROOT.std.vector['std::vector<std::string>']()
        val.push_back(kernel_tensor_names)
        val.push_back(bias_tensor_names)
        upd.AddInitializedTensors(val)
        gin.createUpdateFunction(upd)
        
        weights = edge_model.variables
        for i in weights:
            shape = ROOT.std.vector['std::size_t']()
            shape_as_list = i.shape.as_list()
            for j in shape_as_list:
                shape.push_back(j)
            gin.edges_update_block.GetFunctionBlock().AddInitializedTensor['float'](i.name, ROOT.TMVA.Experimental.SOFIE.ETensorType.FLOAT, shape, i.numpy())
            
    else:
        print("Invalid Model for edge update.")
        return

    # adding the global update function
    global_model = GraphModule._global_block._global_model
    if (global_model.name == "mlp"):
        num_layers = len(global_model._layers)
        upd = ROOT.TMVA.Experimental.SOFIE.RFunction_MLP(ROOT.TMVA.Experimental.SOFIE.FunctionTarget.GLOBALS, num_layers, 0)
        kernel_tensor_names = ROOT.std.vector['std::string']()
        bias_tensor_names   = ROOT.std.vector['std::string']()

        for i in range(0, 2*num_layers, 2):
            print("adding, ", global_model.variables[i].name, global_model.variables[i+1].name)
            bias_tensor_names.push_back(global_model.variables[i].name)
            kernel_tensor_names.push_back(global_model.variables[i+1].name)
        val = ROOT.std.vector['std::vector<std::string>']()
        val.push_back(kernel_tensor_names)
        val.push_back(bias_tensor_names)
        upd.AddInitializedTensors(val)
        gin.createUpdateFunction(upd)
        
        weights = global_model.variables
        for i in weights:
            shape = ROOT.std.vector['std::size_t']()
            shape_as_list = i.shape.as_list()
            for j in shape_as_list:
                shape.push_back(j)
            gin.globals_update_block.GetFunctionBlock().AddInitializedTensor['float'](i.name, ROOT.TMVA.Experimental.SOFIE.ETensorType.FLOAT, shape, i.numpy())
    else:
        print("Invalid Model for global update.")
        return
    
    # adding edge-node aggregate function
    edge_node_reducer = GraphModule._node_block._received_edges_aggregator._reducer.__qualname__
    if(edge_node_reducer == "unsorted_segment_sum"):
        agg = ROOT.TMVA.Experimental.SOFIE.RFunction_Sum()
        gin.createAggregateFunction[ROOT.TMVA.Experimental.SOFIE.RFunction_Sum](agg, ROOT.TMVA.Experimental.SOFIE.FunctionRelation.NODES_EDGES)
    elif(edge_node_reducer == "unsorted_segment_mean"):
        agg = ROOT.TMVA.Experimental.SOFIE.RFunction_Mean()
        gin.createAggregateFunction[ROOT.TMVA.Experimental.SOFIE.RFunction_Mean](agg, ROOT.TMVA.Experimental.SOFIE.FunctionRelation.NODES_EDGES)
    else:
        print("Invalid aggregate function for edge-node reduction")
        return

    
    # adding node-global aggregate function
    node_global_reducer = GraphModule._global_block._nodes_aggregator._reducer.__qualname__
    if(node_global_reducer == "unsorted_segment_sum"):
        agg = ROOT.TMVA.Experimental.SOFIE.RFunction_Sum()
        gin.createAggregateFunction[ROOT.TMVA.Experimental.SOFIE.RFunction_Sum](agg, ROOT.TMVA.Experimental.SOFIE.FunctionRelation.NODES_GLOBALS)
    elif(node_global_reducer == "unsorted_segment_mean"):
        agg = ROOT.TMVA.Experimental.SOFIE.RFunction_Mean()
        gin.createAggregateFunction[ROOT.TMVA.Experimental.SOFIE.RFunction_Mean](agg, ROOT.TMVA.Experimental.SOFIE.FunctionRelation.NODES_GLOBALS)
    else:
        print("Invalid aggregate function for node-global reduction")
        return

    # adding edge-global aggregate function
    node_global_reducer = GraphModule._global_block._edges_aggregator._reducer.__qualname__
    if(node_global_reducer == "unsorted_segment_sum"):
        agg = ROOT.TMVA.Experimental.SOFIE.RFunction_Sum()
        gin.createAggregateFunction[ROOT.TMVA.Experimental.SOFIE.RFunction_Sum](agg, ROOT.TMVA.Experimental.SOFIE.FunctionRelation.EDGES_GLOBALS)
    elif(node_global_reducer == "unsorted_segment_mean"):
        agg = ROOT.TMVA.Experimental.SOFIE.RFunction_Mean()
        gin.createAggregateFunction[ROOT.TMVA.Experimental.SOFIE.RFunction_Mean](agg, ROOT.TMVA.Experimental.SOFIE.FunctionRelation.EDGES_GLOBALS)
    else:
        print("Invalid aggregate function for node-global reduction")
        return


    gnn_model = ROOT.TMVA.Experimental.SOFIE.RModel_GNN(gin)
    blas_routines = ROOT.std.vector['std::string']()
    blas_routines.push_back("Gemm")
    gnn_model.AddBlasRoutines(blas_routines)
    return gnn_model
