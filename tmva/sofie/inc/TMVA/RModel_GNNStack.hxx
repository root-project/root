#ifndef TMVA_SOFIE_RMODEL_GNNStack
#define TMVA_SOFIE_RMODEL_GNNStack

#include <ctime>

#include "TMVA/RModel_Base.hxx"
#include "TMVA/RModel.hxx"
#include "TMVA/RFunction.hxx"

namespace TMVA{
namespace Experimental{
namespace SOFIE{

class RModel_GNNStack: public RModel_GNNBase{

    private:
        std::vector<std::unique_ptr<RModel_GNNBase>> fGraphs;
    
    public:
        RModel_GNNStack(){}
        void Initialize();
        void Generate();
        void AddGraph(std::unique_ptr<RModel_GNNBase> graph){
            fGraphs.emplace_back(std::move(graph));
        }
        ~RModel_GNNStack(){}
};

} // namespace SOFIE
} // namespace Experimental
} // namespace TMVA

#endif //TMVA_SOFIE_RMODEL_GNNStack
