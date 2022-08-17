#ifndef TMVA_SOFIE_RMODEL_GNN
#define TMVA_SOFIE_RMODEL_GNN

#include "TMVA/RModel.hxx"
#include "TMVA/RFunction.hxx"

namespace TMVA{
namespace Experimental{
namespace SOFIE{

struct GNN_Init {
    // updation blocks
    RFunction edges_updation_block;
    RFunction nodes_updation_block;
    RFunction globals_updation_block;
    
    // aggregation blocks
    RFunction edge_node_agg_block;
    RFunction edge_global_agg_block;
    RFunction node_global_agg_block;
   
    std::vector<std::string> nodes;
    std::vector<std::pair<int,int>> edges;
    std::vector<std::string> globals;
};

class RModel_GNN: public RModel{

private:
    
    // updation function for edges, nodes & global attributes
    std::unique_ptr<RFunction> edges_updation_block;
    std::unique_ptr<RFunction> nodes_updation_block;
    std::unique_ptr<RFunction> globals_updation_block;

    // aggregation function for edges, nodes & global attributes
    std::unique_ptr<RFunction> edge_node_agg_block;
    std::unique_ptr<RFunction> edge_global_agg_block;
    std::unique_ptr<RFunction> node_global_agg_block;

    std::vector<std::pair<int,int>> edges; // contains node indices
    std::vector<std::string> nodes;
    std::vector<std::string> globals;
    std::vector<int> senders;              // contains node indices
    std::vector<int> receivers;            // contains node indices

public:

   //explicit move ctor/assn
   RModel_GNN(RModel_GNN&& other);

   RModel_GNN& operator=(RModel_GNN&& other);

   //disallow copy
   RModel_GNN(const RModel_GNN& other) = delete;
   RModel_GNN& operator=(const RModel_GNN& other) = delete;

   RModel_GNN(const GNN_Init& graph_input_struct);
   RModel_GNN(){}
   RModel_GNN(std::string name, std::string parsedtime);

   void AddFunction(std::unique_ptr<RFunction> func);
   
   void InitializeGNN(int batch_size=1);
   void GenerateGNN(int batchSize = 1);
   
   ~RModel_GNN(){}}
   ClassDef(RModel_GNN,1);
};

}//SOFIE
}//Experimental
}//TMVA


#endif //TMVA_SOFIE_RMODEL_GNN