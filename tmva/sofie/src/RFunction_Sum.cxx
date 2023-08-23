#include "TMVA/RFunction_Sum.hxx"


namespace TMVA {
namespace Experimental {
namespace SOFIE {

std::string RFunction_Sum::GenerateModel() {
    std::string modelGenerationString;
    modelGenerationString = "\n//--------- GNN_Aggregate_Function---"+fFuncName+"\n";
    modelGenerationString += "std::vector<float> "+fFuncName+"(const int& num_features, const std::vector<float*>& inputs){\n";
    modelGenerationString += "\tstd::vector<float> result(num_features,0);\n";
    modelGenerationString += "\tfor(auto &it:inputs){\n";
    modelGenerationString += "\t\tstd::transform(result.begin(), result.end(), it, result.begin(), std::plus<float>());\n\t}\n";
    modelGenerationString += "\treturn result;\n}";
    return modelGenerationString;
}

}
}
}
