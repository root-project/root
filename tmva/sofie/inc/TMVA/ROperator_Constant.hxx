#ifndef TMVA_SOFIE_ROPERATOR_Constant
#define TMVA_SOFIE_ROPERATOR_Constant

#include "TMVA/SOFIE_common.hxx"
#include "TMVA/ROperator.hxx"
#include "TMVA/RModel.hxx"

#include <sstream>

namespace TMVA{
namespace Experimental{
namespace SOFIE{

template<typename T>
class ROperator_Constant final : public ROperator
{

private:

   std::string fNX;
   std::string fNY;
   std::vector<size_t> fShape;
   std::vector<T> fValues;
   std::string fAttrType;

public:
   ROperator_Constant(){}

   ROperator_Constant(const std::string & type, const std::vector<T> & values, const std::vector<size_t> & shape, std::string nameX, std::string nameY):
      fNX(UTILITY::Clean_name(nameX)),
      fNY(UTILITY::Clean_name(nameY)),
      fShape(shape),
      fValues(values),
      fAttrType(type) {}

   std::vector<ETensorType> TypeInference(std::vector<ETensorType> input){
      return input;
   }

   std::vector<std::vector<size_t>> ShapeInference(std::vector<std::vector<size_t>> input){
      auto ret = input; //suggest copy to compiler
      return ret;
   }

   void Initialize(RModel& model){
       //input must be a graph input, or already initialized intermediate tensor
      if (!fNX.empty()) {
         // case of ConstantOfShape
         if (model.CheckIfTensorAlreadyExist(fNX) == false){
           throw std::runtime_error("TMVA SOFIE Constant Op Input Tensor is not found in model");
         }
         fShape = model.GetTensorShape(fNX);
      }
       // in case of standard constant the shape is provided as input
       if (ConvertShapeToLength(fShape) != fValues.size())
         throw std::runtime_error("TMVA SOFIE Constant Op has invalid shape : " + ConvertShapeToString(fShape));

      model.AddIntermediateTensor(fNY, ConvertStringToType(fAttrType), fShape);
   }

   std::string Generate(std::string OpName){
      OpName = "op_" + OpName;
      if (fShape.empty()) {
         throw std::runtime_error("TMVA SOFIE Constant called to Generate without being initialized first");
      }
      std::stringstream out;
      out << "\n//------ Constant\n";

      out << SP << "fTensor_" << fNY << "[id] = {";
      for (size_t i = 0; i < fValues.size(); i++) {
         out << fValues[i];
         if (i < fValues.size()-1) out << ", ";
         if (i > 0 && i %10 == 0) out << "\n";
      }

      out << SP << "}\n";
      return out.str();
   }

};

}//SOFIE
}//Experimental
}//TMVA


#endif //TMVA_SOFIE_ROPERATOR_Constant
