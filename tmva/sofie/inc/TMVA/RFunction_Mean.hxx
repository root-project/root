#ifndef TMVA_SOFIE_RFUNCTION_MEAN
#define TMVA_SOFIE_RFUNCTION_MEAN

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

class RFunction_Mean: public RFunction_Aggregate{
    
    public:
        RFunction_Mean():RFunction_Aggregate(FunctionReducer::MEAN){
            fFuncName = "Aggregate_by_Mean";
        }

        std::string GenerateModel(){
            std::string modelGenerationString;
            modelGenerationString = "\n//--------- GNN_Aggregate_Function---"+fFuncName+"\n";
            modelGenerationString += "std::vector<float> "+fFuncName+"(const int& num_features, const std::vector<std::vector<float>::iterator>& inputs){\n";
            modelGenerationString += "\tstd::vector<float> result(num_features,0);\n";
            modelGenerationString += "\tfor(auto &it:inputs){\n";
            modelGenerationString += "\t\tstd::transform(result.begin(), result.end(), it, result.begin(), std::plus<float>());\n\t}\n";
            modelGenerationString += "\tfor_each(result.begin(), result.end(), [&result](float &x){ x /= result.size();\n";
            modelGenerationString += "\treturn result;\n}";
            return modelGenerationString;
        }
                
};

} //SOFIE
} //Experimental
} //TMVA

#endif //TMVA_SOFIE_RFUNCTION_MEAN
