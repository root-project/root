#ifndef TMVA_SOFIE_RMODELPARSER_ONNX
#define TMVA_SOFIE_RMODELPARSER_ONNX

#include "TMVA/SOFIE_common.hxx"
#include "TMVA/RModel.hxx"

#include <string>

namespace TMVA{
namespace Experimental{
namespace SOFIE{

class RModelParser_ONNX{
public:
   RModel Parse(std::string filename);
};

}//SOFIE
}//Experimental
}//TMVA

#endif //TMVA_SOFIE_RMODELPARSER_ONNX
