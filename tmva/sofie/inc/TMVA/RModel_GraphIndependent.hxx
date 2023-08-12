#ifndef TMVA_SOFIE_RMODEL_GraphIndependent
#define TMVA_SOFIE_RMODEL_GraphIndependent

#include <ctime>

#include "TMVA/RModel_Base.hxx"
#include "TMVA/RModel.hxx"
#include "TMVA/RFunction.hxx"

namespace TMVA {
namespace Experimental {
namespace SOFIE {

class RFunction_Update;

struct GraphIndependent_Init {
    // updation blocks
    std::unique_ptr<RFunction_Update> edges_update_block;
    std::unique_ptr<RFunction_Update> nodes_update_block;
    std::unique_ptr<RFunction_Update> globals_update_block;

    int num_nodes;
    std::vector<std::pair<int,int>> edges;

    int num_node_features;
    int num_edge_features;
    int num_global_features;

    std::string filename;

    template <typename T>
    void createUpdateFunction(T& updateFunction) {
        switch(updateFunction.GetFunctionTarget()) {
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
            throw std::runtime_error("TMVA SOFIE: Invalid Update function supplied for creating GraphIndependent function block.");
        }
        }
    }

    ~GraphIndependent_Init() {
        edges_update_block.reset();
        nodes_update_block.reset();
        globals_update_block.reset();
    }
};

class RModel_GraphIndependent: public RModel_GNNBase {

private:

    // updation function for edges, nodes & global attributes
    std::unique_ptr<RFunction_Update> edges_update_block;
    std::unique_ptr<RFunction_Update> nodes_update_block;
    std::unique_ptr<RFunction_Update> globals_update_block;

    int num_nodes;
    int num_edges;

    std::size_t num_node_features;
    std::size_t num_edge_features;
    std::size_t num_global_features;

public:

    //explicit move ctor/assn
    RModel_GraphIndependent(RModel_GraphIndependent&& other);

    RModel_GraphIndependent& operator=(RModel_GraphIndependent&& other);

    //disallow copy
    RModel_GraphIndependent(const RModel_GraphIndependent& other) = delete;
    RModel_GraphIndependent& operator=(const RModel_GraphIndependent& other) = delete;

    RModel_GraphIndependent(GraphIndependent_Init& graph_input_struct);
    RModel_GraphIndependent() {}

    void Generate();

    ~RModel_GraphIndependent() {}
//    ClassDef(RModel_GNN,1);
};

}//SOFIE
}//Experimental
}//TMVA

#endif //TMVA_SOFIE_RMODEL_GNN
