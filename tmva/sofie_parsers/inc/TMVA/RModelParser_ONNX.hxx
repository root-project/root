#ifndef TMVA_SOFIE_RMODELPARSER_ONNX
#define TMVA_SOFIE_RMODELPARSER_ONNX



#include "TMVA/SOFIE_common.hxx"
#include "TMVA/RModel.hxx"
#include "TMVA/OperatorList.hxx"

#include <string>
#include <fstream>
#include <memory>
#include <ctime>
#include <unordered_map>

//forward declaration
namespace onnx{
   class NodeProto;
   class GraphProto;
}

namespace TMVA{
namespace Experimental{
namespace SOFIE{

class RModelParser_ONNX{
public:
   RModel Parse(std::string filename, bool verbose = false);
};


}//SOFIE
}//Experimental
}//TMVA

#endif //TMVA_SOFIE_RMODELPARSER_ONNX
