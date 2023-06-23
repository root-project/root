#ifndef TMVA_SOFIE_ROPERATOR_Where
#define TMVA_SOFIE_ROPERATOR_Where

#include "TMVA/SOFIE_common.hxx"
#include "TMVA/ROperator.hxx"
#include "TMVA/RModel.hxx"

#include <sstream>

namespace TMVA{
namespace Experimental{
namespace SOFIE{

template <typename T>
class ROperator_Where final : public ROperator
{

private:

   std::string fNX;
   std::string fNY;
   std::string fNCondition;
   std::string fNOutput;
   std::vector<size_t> fShape;

public:
   ROperator_Where(){}
   ROperator_Where(std::string nameX, std::string nameY, std::string nameCondition, std::string nameOutput):
      fNX(UTILITY::Clean_name(nameX)), fNY(UTILITY::Clean_name(nameY)), fNCondition(UTILITY::Clean_name(nameCondition)), fNOutput(UTILITY::Clean_name(nameOutput)){}

   std::vector<ETensorType> TypeInference(std::vector<ETensorType> input){
      return input;
   }

   std::vector<std::vector<size_t>> ShapeInference(std::vector<std::vector<size_t>> input){
      auto ret = input; //suggest copy to compiler
      return ret;
   }

   void Initialize(RModel& model){
      if (model.CheckIfTensorAlreadyExist(fNX) == false){   //input must be a graph input, or already initialized intermediate tensor
         throw std::runtime_error("TMVA SOFIE Where Op Input Tensor is not found in model");
      }
      fShape = model.GetTensorShape(fNX);
      model.AddIntermediateTensor(fNY, model.GetTensorType(fNX), fShape);
   }


   std::string Generate(std::string OpName){
      OpName = "op_" + OpName;
      if (fShape.empty()){
         throw std::runtime_error("TMVA SOFIE Operator Where called to Generate without being initialized first");
      }
      std::stringstream out;
      int length = 1;
      length = ConvertShapeToLength(fShape);
      out << "\t" << "for (int id = 0; id < " << length << " ; id++){\n";
      out << "\t\t" << "if(tensor_" << fNCondition << "[id] != 0) { tensor_" << fNOutput << "[id] = tensor_"  << fNX << "[id])};\n";
      out << "\t\t" << "else {tensor_" << fNOutput << "[id] = tensor_"  << fNY << "[id]};\n";
      out << "\t}\n";
      return out.str();
   }

   std::vector<std::string> GetStdLibs() { return { std::string("cmath") };}
};

}//SOFIE
}//Experimental
}//TMVA


#endif //TMVA_SOFIE_ROPERATOR_Where