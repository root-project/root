#include <limits>
#include <algorithm>
#include <cctype>

#include "TMVA/RModel_GraphIndependent.hxx"


namespace TMVA{
namespace Experimental{
namespace SOFIE{


    RModel_GraphIndependent::RModel_GraphIndependent(RModel_GraphIndependent&& other){
      edges_update_block = std::move(other.edges_update_block);
      nodes_update_block = std::move(other.nodes_update_block);
      globals_update_block = std::move(other.globals_update_block);

      num_nodes = std::move(other.num_nodes);
      num_edges = std::move(other.num_edges);

      fName = std::move(other.fName);
      fFileName = std::move(other.fFileName);
      fParseTime = std::move(other.fParseTime);
    }

   RModel_GraphIndependent& RModel_GraphIndependent::operator=(RModel_GraphIndependent&& other){
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

    RModel_GraphIndependent::RModel_GraphIndependent(GraphIndependent_Init& graph_input_struct){
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

    void RModel_GraphIndependent::Generate(){
        std::string hgname;
        GenerateHeaderInfo(hgname);

        std::ofstream f;
        f.open(fName+".dat");
        f.close();

        //Generating Infer function definition for Edge update function
        long next_pos;
        fGC+="\n\nnamespace Edge_Update{\nstruct Session {\n";
        std::vector<std::vector<std::size_t>> Update_Input = {{1, num_edge_features}};
        edges_update_block->Initialize();
        edges_update_block->AddInputTensors(Update_Input);
        fGC+=edges_update_block->GenerateModel(fName);
        next_pos = edges_update_block->GetFunctionBlock()->WriteInitializedTensorsToFile(fName+".dat");
        fGC+="};\n}\n";

        // the number of output edges features can be smaller, so we need to correct here
        auto num_edge_features_input = num_edge_features;
        if(edges_update_block->GetFunctionBlock()->GetTensorShape(edges_update_block->GetFunctionBlock()->GetOutputTensorNames()[0])[1] != num_edge_features){
            num_edge_features = edges_update_block->GetFunctionBlock()->GetTensorShape(edges_update_block->GetFunctionBlock()->GetOutputTensorNames()[0])[1];
        }

        fGC+="\n\nnamespace Node_Update{\nstruct Session {\n";
        // Generating Infer function definition for Node Update function
        Update_Input = {{1, num_node_features}};
        nodes_update_block->Initialize();
        nodes_update_block->AddInputTensors(Update_Input);
        fGC+=nodes_update_block->GenerateModel(fName,next_pos);
        next_pos = nodes_update_block->GetFunctionBlock()->WriteInitializedTensorsToFile(fName+".dat");
        fGC+="};\n}\n";

        // we need to correct the output number of node features
        auto num_node_features_input = num_node_features;
        if(nodes_update_block->GetFunctionBlock()->GetTensorShape(nodes_update_block->GetFunctionBlock()->GetOutputTensorNames()[0])[1] != num_node_features){
            num_node_features = nodes_update_block->GetFunctionBlock()->GetTensorShape(nodes_update_block->GetFunctionBlock()->GetOutputTensorNames()[0])[1];
        }

        fGC+="\n\nnamespace Global_Update{\nstruct Session {\n";
        // Generating Infer function definition for Global Update function
        Update_Input = {{1, num_global_features}};
        globals_update_block->Initialize();
        globals_update_block->AddInputTensors(Update_Input);
        fGC+=globals_update_block->GenerateModel(fName,next_pos);
        next_pos = globals_update_block->GetFunctionBlock()->WriteInitializedTensorsToFile(fName+".dat");
        fGC+="};\n}\n";
        
        // computing inplace on input graph
        fGC += "struct Session {\n";
        fGC += "\n// Instantiating session objects for graph components\n";
        fGC += "Edge_Update::Session edge_update;\n";
        fGC += "Node_Update::Session node_update;\n";
        fGC += "Global_Update::Session global_update;\n\n";

        fGC += "void infer(TMVA::Experimental::SOFIE::GNN_Data& input_graph){\n";

        // computing updated edge attributes
        fGC += "\n// --- Edge Update ---\n";
        for(int k=0; k<num_edges; ++k){
            fGC+="std::vector<float> Edge_"+std::to_string(k)+"_Update = ";
            fGC+=edges_update_block->Generate({"input_graph.edge_data.GetData()+"+std::to_string(k*num_edge_features)});
        }
        if(num_edge_features != num_edge_features_input) {
            fGC+="\ninput_graph.edge_data = input_graph.edge_data.Resize({"+std::to_string(num_edges)+", "+std::to_string(num_edge_features)+"});\n";
        }
        for(int k=0; k<num_edges; ++k){
            fGC+="\nstd::copy(Edge_"+std::to_string(k)+"_Update.begin(),Edge_"+std::to_string(k)+"_Update.end(),input_graph.edge_data.GetData()+"+std::to_string(k*num_edge_features)+");\n";
        }

        // computing updated node attributes
        fGC += "\n// --- Node Update ---\n";
        for(int k=0; k<num_nodes; ++k){
            fGC+="std::vector<float> Node_"+std::to_string(k)+"_Update = ";
            fGC+=nodes_update_block->Generate({"input_graph.node_data.GetData()+"+std::to_string(k*num_node_features)});
        }
        // copy the output of the nodes
        if (num_node_features != num_node_features_input) {
            fGC+="input_graph.node_data = input_graph.node_data.Resize({"+std::to_string(num_nodes)+", "+std::to_string(num_node_features)+"});\n";
        }
        for(int i=0; i<num_nodes; ++i){
            fGC+="std::copy(Node_"+std::to_string(i)+"_Update.begin(), Node_"+std::to_string(i)+"_Update.end(), input_graph.node_data.GetData()+"+std::to_string(i*num_node_features)+");\n";
        }

        // computing updated global attributes
        fGC += "\n// --- Global Update ---\n";
        fGC += "std::vector<float> Global_Data = ";
        fGC += globals_update_block->Generate({"input_graph.global_data.GetData()"});         
        fGC += "\nstd::copy(Global_Data.begin(), Global_Data.end(), input_graph.global_data.GetData());"; 
        fGC += "\n";
        
        fGC += ("}\n};\n} //TMVA_SOFIE_" + fName + "\n");
        fGC += "\n#endif  // TMVA_SOFIE_" + hgname + "\n";

    }

}//SOFIE
}//Experimental
}//TMVA
