#ifndef TMVA_SOFIE_ROPERATOR
#define TMVA_SOFIE_ROPERATOR

#include <vector>
#include <memory>

#include "TMVA/SOFIE_common.hxx"
//#include "RModel.hxx"



namespace TMVA{
namespace Experimental{
namespace SOFIE{

class RModel;

class ROperator{


public:
   virtual std::vector<std::vector<size_t>> ShapeInference(std::vector<std::vector<size_t>>) = 0;
   virtual std::vector<ETensorType> TypeInference(std::vector<ETensorType>) = 0;
   virtual void Initialize(RModel&) = 0;
   virtual std::string Generate(std::string OpName) = 0;  //expect unique opname for each operator within the same RModel
   virtual std::string Header() { return "";}


   //virtual void Forward_reference() = 0;
   //irtual void Forward_blas() = 0;
   virtual ~ROperator(){}

   // Operator's name generally inherited by the original parsed model.
   std::string name = "UnnamedOperator";

};



}//SOFIE
}//Experimental
}//TMVA


#endif //TMVA_SOFIE_OPERATOR
