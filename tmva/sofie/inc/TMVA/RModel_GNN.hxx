#ifndef TMVA_SOFIE_RMODEL_GNN
#define TMVA_SOFIE_RMODEL_GNN

#include <ctime>

#include "TMVA/RModel_Base.hxx"
#include "TMVA/RModel.hxx"
#include "TMVA/RFunction.hxx"

namespace TMVA{
namespace Experimental{
namespace SOFIE{

class RFunction_Update;
class RFunction_Aggregate;

struct GNN_Init {
    // updation blocks
    std::shared_ptr<RFunction_Update> edges_update_block;
    std::shared_ptr<RFunction_Update> nodes_update_block;
    std::shared_ptr<RFunction_Update> globals_update_block;
    
    // aggregation blocks
    std::shared_ptr<RFunction_Aggregate> edge_node_agg_block;
    std::shared_ptr<RFunction_Aggregate> edge_global_agg_block;
    std::shared_ptr<RFunction_Aggregate> node_global_agg_block;
   
    int num_nodes;
    std::vector<std::pair<int,int>> edges;
   
    int num_node_features;
    int num_edge_features;
    int num_global_features;

    std::string filename;
};

class RModel_GNN: public RModel_Base{

private:
    
    // updation function for edges, nodes & global attributes
    std::unique_ptr<RFunction_Update> edges_update_block;
    std::unique_ptr<RFunction_Update> nodes_update_block;
    std::unique_ptr<RFunction_Update> globals_update_block;

    // aggregation function for edges, nodes & global attributes
    std::unique_ptr<RFunction_Aggregate> edge_node_agg_block;
    std::unique_ptr<RFunction_Aggregate> edge_global_agg_block;
    std::unique_ptr<RFunction_Aggregate> node_global_agg_block;

    int num_nodes;
    int num_edges;
    std::vector<int> senders;              // contains node indices
    std::vector<int> receivers;            // contains node indices

    int num_node_features;
    int num_edge_features;
    int num_global_features;

    std::string fFilename;

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

//    void AddFunction(std::unique_ptr<RFunction> func);
   
   void InitializeGNN(int batch_size=1);
   void GenerateGNN(int batchSize = 1);

   ~RModel_GNN(){}
//    ClassDef(RModel_GNN,1);
};

}//SOFIE
}//Experimental
}//TMVA


#endif //TMVA_SOFIE_RMODEL_GNN