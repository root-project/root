#ifndef TMVA_SOFIE_RFUNCTION_SUM
#define TMVA_SOFIE_RFUNCTION_SUM


#include "TMVA/RFunction.hxx"
#include "TMVA/RModel_GNN.hxx"

#include <sstream>
#include <algorithm>
#include <iterator>
#include <iomanip>
#include <limits>
#include <cassert>

namespace TMVA{
namespace Experimental{
namespace SOFIE{

class RFunction_Sum: public RFunction_Aggregate{
    
    public:
        RFunction_Sum():RFunction_Aggregate(FunctionReducer::SUM){
            fFuncName = "Aggregate_by_Sum";
        }

        std::string GenerateModel(){
            std::string modelGenerationString;
            modelGenerationString = "\n//--------- GNN_Aggregate_Function---"+fFuncName+"\n";
            modelGenerationString += "std::vector<float> "+fFuncName+"(const int& num_features, const std::vector<std::vector<float>::iterator>& inputs){\n";
            modelGenerationString += "\tstd::vector<float> result(num_features,0);\n";
            modelGenerationString += "\tfor(auto &it:inputs){\n";
            modelGenerationString += "\t\tstd::transform(result.begin(), result.end(), it, result.begin(), std::plus<float>());\n\t}\n";
            modelGenerationString += "\treturn result;\n}";
            return modelGenerationString;
        }

};

} //SOFIE
} //Experimental
} //TMVA

#endif //TMVA_SOFIE_RFUNCTION_SUM
