#include <limits>
#include <algorithm>
#include <cctype>

#include "TMVA/RModel_GraphIndependent.hxx"

namespace TMVA {
namespace Experimental {
namespace SOFIE {

RModel_GraphIndependent::RModel_GraphIndependent(RModel_GraphIndependent&& other) {
    edges_update_block = std::move(other.edges_update_block);
    nodes_update_block = std::move(other.nodes_update_block);
    globals_update_block = std::move(other.globals_update_block);

    num_nodes = std::move(other.num_nodes);
    num_edges = std::move(other.num_edges);

    fName = std::move(other.fName);
    fFileName = std::move(other.fFileName);
    fParseTime = std::move(other.fParseTime);
}

RModel_GraphIndependent& RModel_GraphIndependent::operator=(RModel_GraphIndependent&& other) {
    edges_update_block = std::move(other.edges_update_block);
    nodes_update_block = std::move(other.nodes_update_block);
    globals_update_block = std::move(other.globals_update_block);

    num_nodes = std::move(other.num_nodes);
    num_edges = std::move(other.num_edges);

    fName = std::move(other.fName);
    fFileName = std::move(other.fFileName);
    fParseTime = std::move(other.fParseTime);

    return *this;
}

RModel_GraphIndependent::RModel_GraphIndependent(GraphIndependent_Init& graph_input_struct) {
    edges_update_block = std::move(graph_input_struct.edges_update_block);
    nodes_update_block = std::move(graph_input_struct.nodes_update_block);
    globals_update_block = std::move(graph_input_struct.globals_update_block);

    num_nodes = graph_input_struct.num_nodes;
    num_edges = graph_input_struct.edges.size();
    num_node_features = graph_input_struct.num_node_features;
    num_edge_features = graph_input_struct.num_edge_features;
    num_global_features = graph_input_struct.num_global_features;

    fFileName = graph_input_struct.filename;
    fName = fFileName.substr(0, fFileName.rfind("."));

    std::time_t ttime = std::time(0);
    std::tm* gmt_time = std::gmtime(&ttime);
    fParseTime  = std::asctime(gmt_time);
}

void RModel_GraphIndependent::Generate() {
    std::string hgname;
    GenerateHeaderInfo(hgname);

    std::ofstream f;
    f.open(fName+".dat");
    f.close();


    long next_pos = 0;
    auto num_edge_features_input = num_edge_features;
    auto num_node_features_input = num_node_features;
    auto num_global_features_input = num_global_features;

    //Generating Infer function definition for Edge update function
    if (edges_update_block) {
       size_t block_size = num_edges;
       fGC += "\n\nnamespace Edge_Update{\nstruct Session {\n";
       std::vector<std::vector<Dim>> update_Input = { { Dim{"num_edges",block_size}, Dim{num_edge_features}} };
       edges_update_block->Initialize();
       edges_update_block->AddInputTensors(update_Input);
       fGC += edges_update_block->GenerateModel(fName);
       next_pos = edges_update_block->GetFunctionBlock()->WriteInitializedTensorsToFile(fName + ".dat");
       fGC += "};\n}\n";

       // the number of output edges features can be smaller, so we need to correct here
       // assume num_edge_features is not a parametric shape
       auto edges_update_output_shape =  edges_update_block->GetFunctionBlock()->GetDynamicTensorShape(edges_update_block->GetFunctionBlock()->GetOutputTensorNames()[0]);
       if(!edges_update_output_shape[1].isParam && edges_update_output_shape[1].dim != num_edge_features_input) {
          num_edge_features = edges_update_output_shape[1].dim;
       }
    }

    if (nodes_update_block) {
      fGC+="\n\nnamespace Node_Update{\nstruct Session {\n";
      // Generating Infer function definition for Node Update function
      // num_node_features is  the output one
      size_t block_size = num_nodes;
      std::vector<std::vector<Dim>> update_Input = { {Dim{"num_nodes", block_size}, Dim{num_node_features}} };
      nodes_update_block->Initialize();
      nodes_update_block->AddInputTensors(update_Input);
      fGC+=nodes_update_block->GenerateModel(fName,next_pos);
      next_pos = nodes_update_block->GetFunctionBlock()->WriteInitializedTensorsToFile(fName+".dat");
      fGC+="};\n}\n";

      // we need to correct the output number of node features
      auto nodes_update_output_shape =  nodes_update_block->GetFunctionBlock()->GetDynamicTensorShape(nodes_update_block->GetFunctionBlock()->GetOutputTensorNames()[0]);
       if(!nodes_update_output_shape[1].isParam && nodes_update_output_shape[1].dim != num_node_features_input) {
          num_node_features = nodes_update_output_shape[1].dim;
      }
    }

    // Generating Infer function definition for Global Update function
    if (globals_update_block) {
      fGC+="\n\nnamespace Global_Update{\nstruct Session {\n";
      std::vector<std::vector<std::size_t>> update_Input = {{1, num_global_features}};
      globals_update_block->Initialize();
      globals_update_block->AddInputTensors(update_Input);
      fGC+=globals_update_block->GenerateModel(fName,next_pos);
      next_pos = globals_update_block->GetFunctionBlock()->WriteInitializedTensorsToFile(fName+".dat");
      fGC+="};\n}\n";

      // we need to correct the output number of global features
      // global features are in shape[1]
#if 0
      auto globals_update_output_shape =  globals_update_block->GetFunctionBlock()->GetDynamicTensorShape(globals_update_block->GetFunctionBlock()->GetOutputTensorNames()[0]);
       if(!globals_update_output_shape[1].isParam && globals_update_output_shape[1].dim != num_global_features_input) {
          num_global_features = globals_update_output_shape[1].dim;
       }
#endif
      if(globals_update_block->GetFunctionBlock()->GetTensorShape(globals_update_block->GetFunctionBlock()->GetOutputTensorNames()[0])[1] != num_global_features) {
         num_global_features = globals_update_block->GetFunctionBlock()->GetTensorShape(globals_update_block->GetFunctionBlock()->GetOutputTensorNames()[0])[1];
      }
    }


    // computing inplace on input graph
    fGC += "struct Session {\n";
    fGC += "\n// Instantiating session objects for graph components\n";
    // create session classes and corresponding temporary vectors
    if (edges_update_block) {
       fGC += "Edge_Update::Session edge_update;\n";
       // this we can remove when we support full dynamic edges and nodes
       fGC += "std::vector<float> fEdgeInputs = std::vector<float>(" + std::to_string(num_edges) + "*" + std::to_string(num_edge_features_input) + ");\n";
       //fGC += "std::vector<float> fEdgeUpdates {" + std::to_string(num_edges) + "*" + std::to_string(num_edge_features) + "};";
    }
    if (nodes_update_block) {
       fGC += "Node_Update::Session node_update;\n";
       fGC += "std::vector<float> fNodeInputs = std::vector<float>(" + std::to_string(num_nodes) + "*" + std::to_string(num_node_features_input) + ");\n";
       //fGC += "std::vector<float> fNodeUpdates {" + std::to_string(num_nodes) + "*" + std::to_string(num_node_features) + "};";
    }
    if (globals_update_block) {
      fGC += "Global_Update::Session global_update;\n\n";
      //fGC += "std::vector<float> fGlobalUpdates {" + std::to_string(num_global_features) + "};";
    }

    fGC += "\nvoid infer(TMVA::Experimental::SOFIE::GNN_Data& input_graph){\n";

    // computing updated edge attributes
    // could use std::span
    if (edges_update_block) {
       fGC += "\n// --- Edge Update ---\n";

       std::string e_size_input = std::to_string(num_edge_features_input);
       fGC += "size_t n_edges = input_graph.edge_data.GetShape()[0];\n";
       fGC += "for (size_t k = 0; k < n_edges; k++) { \n";
       fGC += "   std::copy(input_graph.edge_data.GetData() + k * " + e_size_input +
              ", input_graph.edge_data.GetData() + (k + 1) * " + e_size_input + ", fEdgeInputs.begin() + k * " +
              e_size_input + ");\n";
       fGC += "}\n";

       fGC += "auto edgeUpdates = " + edges_update_block->Generate({"n_edges","fEdgeInputs.data()"}) + "\n";

       if (num_edge_features != num_edge_features_input) {
          fGC += "\n//  resize edge graph data since output feature size is not equal to input size\n";
          fGC += "input_graph.edge_data = input_graph.edge_data.Resize({ n_edges, " +
                 std::to_string(num_edge_features) + "});\n";
       }
       // copy output
       fGC += "\nfor (size_t k = 0; k < n_edges; k++) { \n";
       fGC += "   std::copy(edgeUpdates.begin()+ k * " + std::to_string(num_edge_features) +
              ", edgeUpdates.begin()+ (k+1) * " + std::to_string(num_edge_features) +
              ",input_graph.edge_data.GetData() + k * " + std::to_string(num_edge_features) + ");\n";
       fGC += "}\n";
       fGC += "\n";
    }

    // computing updated node attributes
    if (nodes_update_block) {
       std::string n_size_input = std::to_string(num_node_features_input);
       fGC += "\n// --- Node Update ---\n";
       fGC += "size_t n_nodes = input_graph.node_data.GetShape()[0];\n";
       fGC += "for (size_t k = 0; k < n_nodes; k++) { \n";
       fGC += "   std::copy(input_graph.node_data.GetData() + k * " + n_size_input +
              ", input_graph.node_data.GetData() + (k + 1) * " + n_size_input + ", fNodeInputs.begin() + k * " +
              n_size_input + ");\n";
       fGC += "}\n";

       fGC += "auto nodeUpdates = ";
       fGC += nodes_update_block->Generate({"n_nodes","fNodeInputs.data()"}); // computing updated node attributes
       fGC += "\n";

       if (num_node_features != num_node_features_input) {
          fGC += "\n//  resize node graph data since output feature size is not equal to input size\n";
          fGC += "input_graph.node_data = input_graph.node_data.Resize({ n_nodes, " +
                 std::to_string(num_node_features) + "});\n";
       }
       // copy output
       fGC += "\nfor (size_t k = 0; k < n_nodes; k++) { \n";
       fGC += "   std::copy(nodeUpdates.begin()+ k * " + std::to_string(num_node_features) +
              ", nodeUpdates.begin() + (k+1) * " + std::to_string(num_node_features) +
              ",input_graph.node_data.GetData() + k * " + std::to_string(num_node_features) + ");\n";
       fGC += "}\n";
       fGC += "\n";
    }

    // computing updated global attributes
    if (globals_update_block) {
       fGC += "\n// --- Global Update ---\n";
       fGC += "std::vector<float> Global_Data = ";
       fGC += globals_update_block->Generate({"input_graph.global_data.GetData()"});
       fGC += "\n";

       if (num_global_features != num_global_features_input) {
          fGC += "\n//  resize global graph data since output feature size is not equal to input size\n";
          fGC += "input_graph.global_data = input_graph.global_data.Resize({" + std::to_string(num_global_features) +
                 "});\n";
       }

       fGC += "\nstd::copy(Global_Data.begin(), Global_Data.end(), input_graph.global_data.GetData());";
       fGC += "\n";
    }

    fGC += ("}\n};\n} //TMVA_SOFIE_" + fName + "\n");
    fGC += "\n#endif  // TMVA_SOFIE_" + hgname + "\n";

}

}//SOFIE
}//Experimental
}//TMVA
