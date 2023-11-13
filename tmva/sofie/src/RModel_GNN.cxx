#include <algorithm>
#include <cctype>
#include <fstream>
#include <limits>

#include "TMVA/RModel_GNN.hxx"
#include "TMVA/RFunction.hxx"

namespace TMVA {
namespace Experimental {
namespace SOFIE {

RModel_GNN::RModel_GNN(RModel_GNN&& other) {
    edges_update_block = std::move(other.edges_update_block);
    nodes_update_block = std::move(other.nodes_update_block);
    globals_update_block = std::move(other.globals_update_block);

    edge_node_agg_block = std::move(other.edge_node_agg_block);
    edge_global_agg_block = std::move(other.edge_global_agg_block);
    node_global_agg_block = std::move(other.node_global_agg_block);

    num_nodes = std::move(other.num_nodes);
    num_edges = std::move(other.num_edges);
    senders = std::move(other.senders);
    receivers = std::move(other.receivers);

    fName = std::move(other.fName);
    fFileName = std::move(other.fFileName);
    fParseTime = std::move(other.fParseTime);
}

RModel_GNN& RModel_GNN::operator=(RModel_GNN&& other) {
    edges_update_block = std::move(other.edges_update_block);
    nodes_update_block = std::move(other.nodes_update_block);
    globals_update_block = std::move(other.globals_update_block);

    edge_node_agg_block = std::move(other.edge_node_agg_block);
    edge_global_agg_block = std::move(other.edge_global_agg_block);
    node_global_agg_block = std::move(other.node_global_agg_block);

    num_nodes = std::move(other.num_nodes);
    num_edges = std::move(other.num_edges);
    senders = std::move(other.senders);
    receivers = std::move(other.receivers);

    fName = std::move(other.fName);
    fFileName = std::move(other.fFileName);
    fParseTime = std::move(other.fParseTime);

    return *this;
}

RModel_GNN::RModel_GNN(GNN_Init& graph_input_struct) {
    edges_update_block = std::move(graph_input_struct.edges_update_block);
    nodes_update_block = std::move(graph_input_struct.nodes_update_block);
    globals_update_block = std::move(graph_input_struct.globals_update_block);

    edge_node_agg_block = std::move(graph_input_struct.edge_node_agg_block);
    edge_global_agg_block = std::move(graph_input_struct.edge_global_agg_block);
    node_global_agg_block = std::move(graph_input_struct.node_global_agg_block);

    num_nodes = graph_input_struct.num_nodes;
    num_edges = graph_input_struct.edges.size();
    num_node_features = graph_input_struct.num_node_features;
    num_edge_features = graph_input_struct.num_edge_features;
    num_global_features = graph_input_struct.num_global_features;
    for(auto& it:graph_input_struct.edges) {
        receivers.emplace_back(it.first);
        senders.emplace_back(it.second);
    }
    fFileName = graph_input_struct.filename;
    fName = fFileName.substr(0, fFileName.rfind("."));

    std::time_t ttime = std::time(0);
    std::tm* gmt_time = std::gmtime(&ttime);
    fParseTime  = std::asctime(gmt_time);
}

void RModel_GNN::Generate() {
    std::string hgname;
    GenerateHeaderInfo(hgname);

    std::ofstream f;
    f.open(fName+".dat");
    f.close();

    // Generating Infer function definition for Edge Update function
    long next_pos;
    size_t block_size = num_edges;
    fGC+="\n\nnamespace Edge_Update{\nstruct Session {\n";
    std::vector<std::vector<std::size_t>> Update_Input_edges = {{block_size, num_edge_features},{block_size, num_node_features},{block_size, num_node_features},{block_size, num_global_features}};
    edges_update_block->Initialize();
    edges_update_block->AddInputTensors(Update_Input_edges);
    fGC+=edges_update_block->GenerateModel(fName);
    next_pos = edges_update_block->GetFunctionBlock()->WriteInitializedTensorsToFile(fName+".dat");
    fGC+="};\n}\n";

    // the number of output edges features can be smaller, so we need to correct here
    auto num_edge_features_input = num_edge_features;
    if(edges_update_block->GetFunctionBlock()->GetTensorShape(edges_update_block->GetFunctionBlock()->GetOutputTensorNames()[0])[1] != num_edge_features) {
        num_edge_features = edges_update_block->GetFunctionBlock()->GetTensorShape(edges_update_block->GetFunctionBlock()->GetOutputTensorNames()[0])[1];
    }

    fGC+="\n\nnamespace Node_Update{\nstruct Session {\n";
    // Generating Infer function definition for Node Update function
    // num_node_features is  the output one

    block_size = num_nodes;
    std::vector<std::vector<std::size_t>>  Update_Input_nodes = {{block_size, num_edge_features},{block_size, num_node_features},{block_size, num_global_features}};
    nodes_update_block->Initialize();
    nodes_update_block->AddInputTensors(Update_Input_nodes);
    fGC+=nodes_update_block->GenerateModel(fName,next_pos);
    next_pos = nodes_update_block->GetFunctionBlock()->WriteInitializedTensorsToFile(fName+".dat");
    fGC+="};\n}\n";

    // we need to correct the output number of node features
    auto num_node_features_input = num_node_features;
    if(nodes_update_block->GetFunctionBlock()->GetTensorShape(nodes_update_block->GetFunctionBlock()->GetOutputTensorNames()[0])[1] != num_node_features) {
        num_node_features = nodes_update_block->GetFunctionBlock()->GetTensorShape(nodes_update_block->GetFunctionBlock()->GetOutputTensorNames()[0])[1];
    }

    fGC+="\n\nnamespace Global_Update{\nstruct Session {\n";
    // Generating Infer function definition for Global Update function
    std::vector<std::vector<std::size_t>> Update_Input_globals = {{1, num_edge_features},{1, num_node_features},{1, num_global_features}};
    globals_update_block->Initialize();
    globals_update_block->AddInputTensors(Update_Input_globals);
    fGC+=globals_update_block->GenerateModel(fName,next_pos);
    next_pos = globals_update_block->GetFunctionBlock()->WriteInitializedTensorsToFile(fName+".dat");
    fGC+="};\n}\n";

    // correct for difference in global size  (check shape[1] of output og globals update)
    auto num_global_features_input = num_global_features;
    if(globals_update_block->GetFunctionBlock()->GetTensorShape(globals_update_block->GetFunctionBlock()->GetOutputTensorNames()[0])[1] != num_global_features) {
        num_global_features = globals_update_block->GetFunctionBlock()->GetTensorShape(globals_update_block->GetFunctionBlock()->GetOutputTensorNames()[0])[1];
    }

    fGC+=edge_node_agg_block->GenerateModel();

    if(edge_node_agg_block->GetFunctionType() != edge_global_agg_block->GetFunctionType()) {
        fGC+=edge_global_agg_block->GenerateModel();
    }
    if((edge_node_agg_block->GetFunctionType() != node_global_agg_block->GetFunctionType()) && (edge_global_agg_block->GetFunctionType() != node_global_agg_block->GetFunctionType())) {
        fGC+=node_global_agg_block->GenerateModel();
    }
    fGC+="\n\n";

    // computing inplace on input graph
    fGC += "struct Session {\n";
    fGC += "\n// Instantiating session objects for graph components\n";
    fGC += "Edge_Update::Session edge_update;\n";
    fGC += "Node_Update::Session node_update;\n";
    fGC += "Global_Update::Session global_update;\n\n";

    std::string e_num = std::to_string(num_edges);
    std::string n_num = std::to_string(num_nodes);
    std::string e_size_input =  std::to_string(num_edge_features_input);
    std::string n_size_input =  std::to_string(num_node_features_input);
    std::string g_size_input =  std::to_string(num_global_features_input);
    std::string e_size =  std::to_string(num_edge_features);
    std::string n_size =  std::to_string(num_node_features);
    std::string g_size =  std::to_string(num_global_features);

    // create temp vector for edge and node updates
    fGC += "std::vector<float> fEdgeUpdates = std::vector<float>(" + e_num + "*" + e_size + ");\n";
    fGC += "\n\nstd::vector<float> fNodeUpdates = std::vector<float>(" + n_num + "*" + n_size + ");\n";

    fGC += "\n// input vectors for edge update\n";
    fGC += "std::vector<float> fEdgeInputs = std::vector<float>(" + e_num + "*" + e_size_input + ");\n";
    fGC += "std::vector<float> fRecNodeInputs = std::vector<float>(" + e_num + "*" + n_size_input + ");\n";
    fGC += "std::vector<float> fSndNodeInputs = std::vector<float>(" + e_num + "*" + n_size_input + ");\n";
    fGC += "std::vector<float> fGlobInputs = std::vector<float>(" + e_num + "*" + g_size_input + ");\n\n";

    fGC += "\n// input vectors for node update\n";
    fGC += "std::vector<float> fNodeInputs = std::vector<float>(" + n_num + "*" + n_size_input + ");\n";
    fGC += "std::vector<float> fNodeEdgeAggregate = std::vector<float>(" + n_num + "*" + n_size_input + ", 0);\n";
    fGC += "std::vector<float> fNodeAggregateTemp;\n";

    fGC += "\nvoid infer(TMVA::Experimental::SOFIE::GNN_Data& input_graph){\n";

    // computing updated edge attributes
    fGC += "\n// --- Edge Update ---\n";
    fGC +=  "size_t n_edges = input_graph.edge_data.GetShape()[0];\n";

    fGC += "for (size_t k = 0; k < n_edges; k++) { \n";
    fGC += "   std::copy(input_graph.edge_data.GetData() + k * " + e_size_input +
           ", input_graph.edge_data.GetData() + (k + 1) * " + e_size_input +
           ", fEdgeInputs.begin() + k * " + e_size_input + ");\n";
    fGC += "   std::copy(input_graph.node_data.GetData() + input_graph.receivers[k] * " + n_size_input +
           ", input_graph.node_data.GetData() + (input_graph.receivers[k] + 1) * " + n_size_input +
           ", fRecNodeInputs.begin() + k * " + n_size_input + ");\n";
    fGC += "   std::copy(input_graph.node_data.GetData() + input_graph.senders[k] * " + n_size_input +
           ", input_graph.node_data.GetData() + (input_graph.senders[k] + 1) * " + n_size_input +
           ", fSndNodeInputs.begin() + k * " + n_size_input + ");\n";
    fGC += "   std::copy(input_graph.global_data.GetData()";
    fGC += ", input_graph.global_data.GetData() + " + g_size_input +
           ", fGlobInputs.begin() + k * " + g_size_input + ");\n";
    fGC += "}\n";

    fGC += "fEdgeUpdates = " + edges_update_block->Generate({"fEdgeInputs.data(), fRecNodeInputs.data(), fSndNodeInputs.data(), fGlobInputs.data()"}) + "\n";

    if(num_edge_features != num_edge_features_input) {
        fGC += "\n//  resize edge graph data since output feature size is not equal to input size\n";
        fGC+="input_graph.edge_data = input_graph.edge_data.Resize({n_edges, "+e_size+"});\n";
    }
    // copy output
    fGC += "\nfor (size_t k = 0; k < n_edges; k++) { \n";
    fGC += "   std::copy(fEdgeUpdates.begin()+ k * " + e_size + ", fEdgeUpdates.begin()+ (k+1) * " + e_size +
           ",input_graph.edge_data.GetData() + k * " + e_size + ");\n";
    fGC += "}\n";
    fGC += "\n";

    fGC += "\n\n// --- Node Update ---\n";
    fGC += "size_t n_nodes = input_graph.node_data.GetShape()[0];\n";
    // computing updated node attributes
    fGC += "for (size_t k = 0; k < n_nodes; k++) { \n";
    fGC += "   std::copy(input_graph.node_data.GetData() + k * " + n_size_input +
           ", input_graph.node_data.GetData() + (k + 1) * " + n_size_input +
           ", fNodeInputs.begin() + k * " + n_size_input + ");\n";
    fGC += "}\n";
    // reset initial aggregate edge vector to zero
    fGC += "\nstd::fill(fNodeEdgeAggregate.begin(), fNodeEdgeAggregate.end(), 0.);\n";
    // fGlobInputs is size { nedges, ngloblas}. It needs to be here { nnodes, nglobals}
    // if number of nodes is larger than edges we need to resize it and copy values

    fGC += "\n// resize global vector feature to number of nodes if needed\n";
    fGC += "if (n_nodes > n_edges) {\n";
    fGC += "   fGlobInputs.resize( n_nodes * " + std::to_string(num_global_features_input) + ");\n";
    fGC += "   for (size_t k = n_edges; k < n_nodes; k++)\n";
    fGC += "      std::copy(fGlobInputs.begin(), fGlobInputs.begin() + " + g_size_input +
                   " , fGlobInputs.begin() + k * " + g_size_input + ");\n";
    fGC += "}\n";

    // loop on nodes and aggregate incoming edges
    fGC += "\n// aggregate edges going to a node\n";
    fGC += "for (size_t j = 0; j < n_nodes; j++) {\n";
    // approximate number of receivers/node to allocate vector
    fGC += "   std::vector<float *> edgesData; edgesData.reserve( int(n_edges/n_nodes) +1);\n";
    // loop on edges
    fGC += "   for (size_t k = 0; k < n_edges; k++) {\n";
    fGC += "      if (input_graph.receivers[k] == j) \n";
    fGC += "         edgesData.emplace_back(input_graph.edge_data.GetData() + k * " + e_size + ");\n";
    fGC += "   }\n";
    fGC += "   fNodeAggregateTemp = " + edge_node_agg_block->Generate(num_edge_features, "edgesData") + ";\n";
    fGC += "   std::copy(fNodeAggregateTemp.begin(), fNodeAggregateTemp.end(), fNodeEdgeAggregate.begin() + " +
                   e_size + " * j);\n";
    fGC += "}\n";   // end node loop


    fGC+="\n";
    fGC+="fNodeUpdates = ";
    fGC+=nodes_update_block->Generate({"fNodeEdgeAggregate.data()","fNodeInputs.data()","fGlobInputs.data()"});    // computing updated node attributes
    fGC+="\n";

    if(num_node_features != num_node_features_input) {
        fGC += "\n//  resize node graph data since output feature size is not equal to input size\n";
        fGC+="input_graph.node_data = input_graph.node_data.Resize({n_nodes, " + n_size + "});\n";
    }
    // copy output
    fGC += "\nfor (size_t k = 0; k < n_nodes; k++) { \n";
    fGC += "   std::copy(fNodeUpdates.begin()+ k * " + n_size + ", fNodeUpdates.begin() + (k+1) * " + n_size +
           ",input_graph.node_data.GetData() + k * " + n_size+ ");\n";
    fGC += "}\n";
    fGC += "\n";

    // aggregating edges & nodes for global update
    fGC += "std::vector<float *> allEdgesData; allEdgesData.reserve(n_edges);\n";
    fGC += "for (size_t k = 0; k < n_edges; k++) {\n";
    fGC += "   allEdgesData.emplace_back(input_graph.edge_data.GetData() + k * " + e_size + ");\n";
    fGC += "}\n";
    fGC += "std::vector<float *> allNodesData; allNodesData.reserve(n_nodes);\n";
    fGC += "for (size_t k = 0; k < n_nodes; k++) {\n";
    fGC += "   allNodesData.emplace_back(input_graph.node_data.GetData() + k * " + n_size + ");\n";
    fGC += "}\n";


    fGC += "\n// --- Global Update ---\n";
    fGC+="std::vector<float> Edge_Global_Aggregate = ";
    fGC+=edge_global_agg_block->Generate(num_edge_features, "allEdgesData");     // aggregating edge attributes globally
    fGC+=";\n";

    fGC+="std::vector<float> Node_Global_Aggregate = ";
    fGC+=node_global_agg_block->Generate(num_node_features, "allNodesData");     // aggregating node attributes globally
    fGC+=";\n";

    // computing updated global attributes
    fGC += "std::vector<float> Global_Data = ";
    fGC += globals_update_block->Generate({"Edge_Global_Aggregate.data()","Node_Global_Aggregate.data()", "input_graph.global_data.GetData()"});
    if(num_global_features != num_global_features_input) {
        fGC += "\n//  resize global graph data since output feature size is not equal to input size\n";
        fGC+="input_graph.global_data = input_graph.global_data.Resize({"+g_size+"});\n";
    }
    fGC += "\nstd::copy(Global_Data.begin(), Global_Data.end(), input_graph.global_data.GetData());";
    fGC+="\n}\n";
    fGC+="};\n";

    fGC += ("} //TMVA_SOFIE_" + fName + "\n");
    fGC += "\n#endif  // TMVA_SOFIE_" + hgname + "\n";
}

}//SOFIE
}//Experimental
}//TMVA
