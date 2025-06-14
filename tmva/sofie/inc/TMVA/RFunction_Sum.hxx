#ifndef TMVA_SOFIE_RFUNCTION_SUM
#define TMVA_SOFIE_RFUNCTION_SUM


#include "TMVA/RFunction.hxx"

namespace TMVA {
namespace Experimental {
namespace SOFIE {

class RFunction_Sum: public RFunction_Aggregate {

public:
    RFunction_Sum():RFunction_Aggregate(FunctionReducer::SUM) {
        fFuncName = "Aggregate_by_Sum";
    }

    std::string GenerateModel() override;
};

} //SOFIE
} //Experimental
} //TMVA

#endif //TMVA_SOFIE_RFUNCTION_SUM
