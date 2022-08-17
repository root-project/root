#include <limits>
#include <algorithm>
#include <cctype>

#include "TMVA/RModel_GNN.hxx"


namespace TMVA{
namespace Experimental{
namespace SOFIE{


    RModel_GNN::RModel_GNN(RModel_GNN&& other){
      edges_updation_block = std::move(other.edges_block);
      nodes_updation_block = std::move(other.nodes_block);
      globals_updation_block = std::move(other.globals_block);

      edge_node_agg_block = std::move(other.edge_node_agg_block);
      edge_global_agg_block = std::move(other.edge_global_agg_block);
      node_global_agg_block = std::move(other.node_global_agg_block);

      edges = std::move(other.edges);
      nodes = std::move(other.nodes);
      globals = std::move(other.globals);
      senders = std::move(other.senders);
      receivers = std::move(other.receivers);
    }

   RModel_GNN& RModel_GNN::operator=(RModel_GNN&& other){
      edges_updation_block = std::move(other.edges_block);
      nodes_updation_block = std::move(other.nodes_block);
      globals_updation_block = std::move(other.globals_block);

      edge_node_agg_block = std::move(other.edge_node_agg_block);
      edge_global_agg_block = std::move(other.edge_global_agg_block);
      node_global_agg_block = std::move(other.node_global_agg_block);

      edges = std::move(other.edges);
      nodes = std::move(other.nodes);
      globals = std::move(other.globals);
      senders = std::move(other.senders);
      receivers = std::move(other.receivers);
    }

    RModel_GNN::RModel_GNN(const GNN_Init& graph_input_struct){
        edges_updation_block = std::make_unique<RFunction>(graph_input_struct.edges_block);
        nodes_updation_block = std::make_unique<RFunction>(graph_input_struct.nodes_block);
        globals_updation_block = std::make_unique<RFunction>(graph_input_struct.globals_block);

        edge_node_agg_block = std::make_unique<RFunction>(graph_input_struct.edge_node_agg_block);
        edge_global_agg_block = std::make_unique<RFunction>(graph_input_struct.edge_global_agg_block);
        node_global_agg_block = std::make_unique<RFunction>(graph_input_struct.node_global_agg_block);

        nodes = std::move(graph_input_struct.nodes);
        globals = std::move(graph_input_struct.globals);
        nodes = std::move(other.nodes);
        for(auto& it:graph_input_struct.edges){
            senders.emplace_back(it.first);
            receivers.emplace_back(it.second);
        }
    }

    void RModel_GNN::InitializeGNN(int batch_size){
        edges_updation_block->function_block->Initialize(batch_size);
        nodes_updation_block->function_block->Initialize(batch_size);

        edge_node_agg_block->function_block->Initialize(batch_size);
        edge_global_agg_block->function_block->Initialize(batch_size);
        node_global_agg_block->function_block->Initialize(batch_size);        
    }

    void RModel_GNN::GenerateGNN(int batchSize){
        InitializeGNN(batchSize);
        Generate(Options::kGNN, batchSize);
        
        // computing inplace on input graph
        fGC += "GNN::GNN_Data infer(GNN::GNN_Data input_graph){\n";
        
        // computing updated edge attributes
        for(int k=0; k<edges.size(); ++k){
            fGC+=edges_block->Generate(edges[k],nodes[edges[k].first], nodes[edges[k].second], globals);
        }

        for(int i=0; i<nodes.size(); ++i){
            std::vector<GNN::GNN_Agg> agg_data_per_node;
            for(int k=0; k<edges.size(); ++k){
                agg_data_per_node.push_back({edges[k],nodes[i],nodes[edges[k].second]});
            }
            fGC+=edge_node_agg_block->Generate(agg_data_per_node);             // aggregating edge attributes per node
            fGC+=nodes_updation_block->Generate(edges[i],nodes[i],globals);    // computing updated node attributes 
        }

        std::vector<GNN::GNN_Agg> agg_data;
        for(int k=0; k<edges.size(); ++k){
                agg_data.push_back({edges[k],nodes[i],nodes[edges[k].second]});
        }
        fGC+=edge_global_agg_block->Generate(agg_data);     // aggregating edge attributes globally
        fGC+=node_global_agg_block->Generate(agg_data);     // aggregating node attributes globally
        fGC+=globals_updation_block->Generate(edges,nodes,globals); // computing updated global attributes

        fGC+="\nreturn input_graph;\n}";
        if (fUseSession) {
            fGC += "};\n";
         }
         fGC += ("} //TMVA_SOFIE_" + fName + "\n");
         fGC += "\n#endif  // TMVA_SOFIE_" + fName + "\n";
    }

    void RModel_GNN::AddFunction(std::unique_ptr<RFunction> func){
        switch(func->fTarget){
            case(FunctionTarget::NODES){
                nodes_block.emplace_back(func);
                break;
            }
            case(FunctionTarget::EDGES){
                edges_block.emplace_back(func);
                break;
            }
            case(FunctionTarget::GLOBALS){
                globals_block.emplace_back(func);
                break;
            }
        }
    }




}//SOFIE
}//Experimental
}//TMVA
