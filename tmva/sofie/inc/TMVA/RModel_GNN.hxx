#ifndef TMVA_SOFIE_RMODEL_GNN
#define TMVA_SOFIE_RMODEL_GNN

#include <ctime>

#include "TMVA/RModel_Base.hxx"
#include "TMVA/RModel.hxx"
#include "TMVA/RFunction.hxx"

namespace TMVA {
namespace Experimental {
namespace SOFIE {

class RFunction_Update;
class RFunction_Aggregate;

struct GNN_Init {

   // Explicitly define default constructor so cppyy doesn't attempt
   // aggregate initialization.
   GNN_Init() {}

   // update blocks
   std::unique_ptr<RFunction_Update> edges_update_block;
   std::unique_ptr<RFunction_Update> nodes_update_block;
   std::unique_ptr<RFunction_Update> globals_update_block;

   // aggregation blocks
   std::unique_ptr<RFunction_Aggregate> edge_node_agg_block;
   std::unique_ptr<RFunction_Aggregate> edge_global_agg_block;
   std::unique_ptr<RFunction_Aggregate> node_global_agg_block;

   std::size_t num_nodes;
   std::vector<std::pair<int, int>> edges;

   std::size_t num_node_features;
   std::size_t num_edge_features;
   std::size_t num_global_features;

   std::string filename;

   template <typename T>
   void createUpdateFunction(T &updateFunction)
   {
      switch (updateFunction.GetFunctionTarget()) {
      case FunctionTarget::EDGES: {
         edges_update_block.reset(new T(updateFunction));
         break;
      }
      case FunctionTarget::NODES: {
         nodes_update_block.reset(new T(updateFunction));
         break;
      }
      case FunctionTarget::GLOBALS: {
         globals_update_block.reset(new T(updateFunction));
         break;
      }
      default: {
         throw std::runtime_error("TMVA SOFIE: Invalid Update function supplied for creating GNN function block.");
      }
      }
   }

   template <typename T>
   void createAggregateFunction(T &aggFunction, FunctionRelation relation)
   {
      switch (relation) {
      case FunctionRelation::NODES_EDGES: {
         edge_node_agg_block.reset(new T(aggFunction));
         break;
      }
      case FunctionRelation::NODES_GLOBALS: {
         node_global_agg_block.reset(new T(aggFunction));
         break;
      }
      case FunctionRelation::EDGES_GLOBALS: {
         edge_global_agg_block.reset(new T(aggFunction));
         break;
      }
      default: {
         throw std::runtime_error("TMVA SOFIE: Invalid Aggregate function supplied for creating GNN function block.");
      }
      }
   }
};

class RModel_GNN final : public RModel_GNNBase {

private:
   // update function for edges, nodes & global attributes
   std::unique_ptr<RFunction_Update> edges_update_block;
   std::unique_ptr<RFunction_Update> nodes_update_block;
   std::unique_ptr<RFunction_Update> globals_update_block;

   // aggregation function for edges, nodes & global attributes
   std::unique_ptr<RFunction_Aggregate> edge_node_agg_block;
   std::unique_ptr<RFunction_Aggregate> edge_global_agg_block;
   std::unique_ptr<RFunction_Aggregate> node_global_agg_block;

   std::size_t num_nodes; // maximum number of nodes
   std::size_t num_edges; // maximum number of edges

   std::size_t num_node_features;
   std::size_t num_edge_features;
   std::size_t num_global_features;

public:
   RModel_GNN(GNN_Init &graph_input_struct);

   void Generate() final;
};

} // namespace SOFIE
} // namespace Experimental
} // namespace TMVA

#endif // TMVA_SOFIE_RMODEL_GNN
