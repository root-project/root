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
   bool fIsConstantOfShape = false;

public:
   ROperator_Constant(){}

   ROperator_Constant(const std::string & type, const std::vector<T> & values, const std::vector<size_t> & shape, std::string nameX, std::string nameY):
      fNX(UTILITY::Clean_name(nameX)),
      fNY(UTILITY::Clean_name(nameY)),
      fShape(shape),
      fValues(values),
      fAttrType(type)
      { }

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
         fIsConstantOfShape  = true;
         if (model.CheckIfTensorAlreadyExist(fNX) == false){
           throw std::runtime_error("TMVA SOFIE ConstantOfShape Op Input Tensor is not found in model");
         }
         // shape is given by values of input in this case. Use empty one
         fShape = std::vector<size_t> ();
         if (fValues.size() != 1)
            throw std::runtime_error("TMVA SOFIE ConstantOfShape Op value Tensor has invalid size " + std::to_string(fValues.size()));
      }
       // in case of standard constant the shape is provided as input
       if (ConvertShapeToLength(fShape) != fValues.size())
         throw std::runtime_error("TMVA SOFIE Constant Op has invalid shape : " + ConvertShapeToString(fShape) +
                                 " with " + std::to_string(fValues.size()) + " values");

      model.AddIntermediateTensor(fNY, ConvertStringToType(fAttrType), fShape);
   }

   std::string Generate(std::string OpName){
      OpName = "op_" + OpName;
      if (!fIsConstantOfShape && fShape.empty()) {
         throw std::runtime_error("TMVA SOFIE Constant called to Generate without being initialized first");
      }
      std::stringstream out;
      if (fIsConstantOfShape)
         out << "\n//------ ConstantOfShape\n";
      else
         out << "\n//------ Constant\n";

      if (!fIsConstantOfShape) {
         out << SP << "fTensor_" << fNY << " = {";
         for (size_t i = 0; i < fValues.size(); i++) {
            out << fValues[i];
            if (i < fValues.size()-1) {
               out << ", ";
               if  (i > 0 && i %10 == 0) out << "\n";
            }
         }
         out << "};\n";
      }
      // in case of Constant of Shape shape is given by input. fValues could be empty and all
      // vector is initialiazed with zero values
      else {
          // in case of ConstantOfShape
          // compute length of output tensor from input tensor
         out << SP << "size_t outputLength = 1;\n";
         out << SP << "for (auto& dim: fTensor_" << fNX << ") outputLength *= dim;\n";
         out << SP << "fTensor_" << fNY << ".assign(outputLength, " << fValues[0] << ");\n";
      }

      return out.str();
   }

};

}//SOFIE
}//Experimental
}//TMVA


#endif //TMVA_SOFIE_ROPERATOR_Constant
