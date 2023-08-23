#include "TMVA/RFunction_Mean.hxx"


namespace TMVA {
namespace Experimental {
namespace SOFIE {

std::string RFunction_Mean::GenerateModel() {
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

}
}
}
