#ifndef TMVA_SOFIE_RFUNCTION_MEAN
#define TMVA_SOFIE_RFUNCTION_MEAN

#include "TMVA/RFunction.hxx"

namespace TMVA {
namespace Experimental {
namespace SOFIE {

class RFunction_Mean: public RFunction_Aggregate {

public:
    RFunction_Mean():RFunction_Aggregate(FunctionReducer::MEAN) {
        fFuncName = "Aggregate_by_Mean";
    }

    std::string GenerateModel() override;
};

} //SOFIE
} //Experimental
} //TMVA

#endif //TMVA_SOFIE_RFUNCTION_MEAN
