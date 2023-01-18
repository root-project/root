#include <algorithm>
#include <cctype>
#include <fstream>
#include <limits>

#include "TMVA/RModel_GNN.hxx"


namespace TMVA{
namespace Experimental{
namespace SOFIE{


    RModel_GNN::RModel_GNN(RModel_GNN&& other){
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

   RModel_GNN& RModel_GNN::operator=(RModel_GNN&& other){
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

    RModel_GNN::RModel_GNN(GNN_Init& graph_input_struct){
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
        for(auto& it:graph_input_struct.edges){
            receivers.emplace_back(it.first); 
            senders.emplace_back(it.second);
        }
        fFileName = graph_input_struct.filename;
        fName = fFileName.substr(0, fFileName.rfind("."));

        std::time_t ttime = std::time(0);
        std::tm* gmt_time = std::gmtime(&ttime);
        fParseTime  = std::asctime(gmt_time);
    }

    void RModel_GNN::Generate(){
        std::string hgname;
        GenerateHeaderInfo(hgname);

        std::ofstream f;
        f.open(fName+".dat");
        f.close();

        // Generating Infer function definition for Edge Update function
        long next_pos;
        fGC+="\n\nnamespace Edge_Update{\nstruct Session {\n";
        std::vector<std::vector<std::size_t>> Update_Input = {{1, num_edge_features},{1, num_node_features},{1, num_node_features},{1, num_global_features}};
        edges_update_block->Initialize();
        edges_update_block->AddInputTensors(Update_Input);
        fGC+=edges_update_block->GenerateModel(fName);
        next_pos = edges_update_block->GetFunctionBlock()->WriteInitializedTensorsToFile(fName+".dat");
        fGC+="};\n}\n";

        fGC+="\n\nnamespace Node_Update{\nstruct Session {\n";
        // Generating Infer function definition for Node Update function
        Update_Input = {{1, num_edge_features},{1, num_node_features},{1, num_global_features}};
        nodes_update_block->Initialize();
        nodes_update_block->AddInputTensors(Update_Input);
        fGC+=nodes_update_block->GenerateModel(fName,next_pos);
        next_pos = nodes_update_block->GetFunctionBlock()->WriteInitializedTensorsToFile(fName+".dat");
        fGC+="};\n}\n";

        fGC+="\n\nnamespace Global_Update{\nstruct Session {\n";
        // Generating Infer function definition for Global Update function
        globals_update_block->Initialize();
        globals_update_block->AddInputTensors(Update_Input);
        fGC+=globals_update_block->GenerateModel(fName,next_pos);
        next_pos = globals_update_block->GetFunctionBlock()->WriteInitializedTensorsToFile(fName+".dat");
        fGC+="};\n}\n";

        fGC+=edge_node_agg_block->GenerateModel();

        if(edge_node_agg_block->GetFunctionType() != edge_global_agg_block->GetFunctionType()){
            fGC+=edge_global_agg_block->GenerateModel();
        }
        if((edge_node_agg_block->GetFunctionType() != node_global_agg_block->GetFunctionType()) && (edge_global_agg_block->GetFunctionType() != node_global_agg_block->GetFunctionType())){
            fGC+=node_global_agg_block->GenerateModel();
        }
        fGC+="\n\n";

        // computing inplace on input graph
        fGC += "void infer(TMVA::Experimental::SOFIE::GNN_Data& input_graph){\n";
        
        fGC += "\n// Instantiating session objects for graph components\n";
        fGC += "Edge_Update::Session edge_update;\n";
        fGC += "Node_Update::Session node_update;\n";
        fGC += "Global_Update::Session global_update;\n";


        // computing updated edge attributes
        fGC += "\n// --- Edge Update ---\n";
        for(int k=0; k<num_edges; ++k){
            fGC+="std::vector<float> Edge_"+std::to_string(k)+"_Update = ";
            fGC+=edges_update_block->Generate({"input_graph.edge_data.GetData()+"+std::to_string(k*num_edge_features),"input_graph.node_data.GetData()+"+std::to_string(receivers[k] * num_node_features),"input_graph.node_data.GetData()+"+std::to_string(senders[k] * num_node_features),"input_graph.global_data.GetData()"});
            fGC+="\nstd::copy(Edge_"+std::to_string(k)+"_Update.begin(),Edge_"+std::to_string(k)+"_Update.end(),input_graph.edge_data.begin()+"+std::to_string(k*num_edge_features)+");";
        }
        fGC+="\n";

        // aggregating edge if it's a receiver node and then updating corresponding node
        fGC += "\n// --- Node Update ---\n";
        std::vector<std::string> Node_Edge_Aggregate_String;
        for(int i=0; i<num_nodes; ++i){
            for(int k=0; k<num_edges; ++k){
                if(receivers[k] == i){
                    Node_Edge_Aggregate_String.emplace_back("input_graph.edge_data.GetData()+"+std::to_string(k*num_edge_features));
                }
            }
            
            fGC+="std::vector<float> Node_"+std::to_string(i)+"_Edge_Aggregate = ";
            
            // when node is not a receiver, fill the aggregated vector with 0 values
            if(Node_Edge_Aggregate_String.size()==0){
                fGC.resize(fGC.size()-2);
                fGC+="("+std::to_string(num_edge_features)+", 0);";
            } else {
                fGC+=edge_node_agg_block->Generate(num_edge_features,{Node_Edge_Aggregate_String});                      // aggregating edge attributes per node
            }

            fGC+="\n";
            fGC+="std::vector<float> Node_"+std::to_string(i)+"_Update = ";
            fGC+=nodes_update_block->Generate({"Node_"+std::to_string(i)+"_Edge_Aggregate.data()","input_graph.node_data.GetData()+"+std::to_string(i*num_node_features),"input_graph.global_data.GetData()"});    // computing updated node attributes 
            fGC+="\n";
            fGC+="std::copy(Node_"+std::to_string(i)+"_Update.begin(), Node_"+std::to_string(i)+"_Update.end(), input_graph.node_data.begin()+"+std::to_string(i*num_node_features)+");";
            fGC+="\n";
            Node_Edge_Aggregate_String.clear();
        }

        // aggregating edges & nodes for global update
        std::vector<std::string> Node_Global_Aggregate_String;
        for(int k=0; k<num_nodes; ++k){
            Node_Global_Aggregate_String.emplace_back("input_graph.node_data.GetData()+"+std::to_string(k*num_node_features));
        }

        std::vector<std::string> Edge_Global_Aggregate_String;
        for(int k=0; k<num_edges; ++k){
            Edge_Global_Aggregate_String.emplace_back("input_graph.edge_data.GetData()+"+std::to_string(k*num_edge_features));
        }

        fGC += "\n// --- Global Update ---\n";        
        fGC+="std::vector<float> Edge_Global_Aggregate = ";
        fGC+=edge_global_agg_block->Generate(num_edge_features, Edge_Global_Aggregate_String);     // aggregating edge attributes globally
        fGC+="\n";

        fGC+="std::vector<float> Node_Global_Aggregate = ";
        fGC+=node_global_agg_block->Generate(num_node_features, Node_Global_Aggregate_String);     // aggregating node attributes globally
        fGC+="\n";

        // computing updated global attributes
        fGC += "std::vector<float> Global_Data = ";
        fGC += globals_update_block->Generate({"Edge_Global_Aggregate.data()","Node_Global_Aggregate.data()", "input_graph.global_data.GetData()"}); 
        fGC += "\nstd::copy(Global_Data.begin(), Global_Data.end(), input_graph.global_data.GetData());"; 
        fGC+="\n}\n";

        fGC += ("} //TMVA_SOFIE_" + fName + "\n");
        fGC += "\n#endif  // TMVA_SOFIE_" + hgname + "\n";
    }

}//SOFIE
}//Experimental
}//TMVA
