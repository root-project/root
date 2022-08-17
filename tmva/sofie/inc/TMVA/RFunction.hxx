#ifndef TMVA_SOFIE_RFUNCTION
#define TMVA_SOFIE_RFUNCTION


#include "TMVA/ROperator.hxx"

namespace TMVA{
namespace Experimental{
namespace SOFIE{

class RModel;

enum class FunctionType{
        UPDATION=0, AGGREGATE=1
};
enum class FunctionTarget{
        NODES=0, EDGES=1, GLOBALS=2
};
class RFunction: public ROperator{
    FunctionType fType;
    FunctionTarget fTarget;
    std::unique_ptr<RModel> function_block;
};


}//SOFIE
}//Experimental
}//TMVA


#endif //TMVA_SOFIE_RFUNCTION