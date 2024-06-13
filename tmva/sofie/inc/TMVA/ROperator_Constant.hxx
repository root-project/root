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
         // case of ConstantOfShape (since no inputs in case of Constant operator)
         fIsConstantOfShape  = true;
         if (model.CheckIfTensorAlreadyExist(fNX) == false){
           throw std::runtime_error("TMVA SOFIE ConstantOfShape Op Input Tensor is not found in model");
         }
         // shape is given by values of input in this case. Use empty one
         fShape = std::vector<size_t> ();
         if (fValues.size() != 1)
            throw std::runtime_error("TMVA SOFIE ConstantOfShape Op value Tensor has invalid size " + std::to_string(fValues.size()));

         // in case of constant of shape output is an intermediate tensor
         // the values are set in the Generate function, since the output tensor shape is an input
         // and can be known only at run time
         model.AddIntermediateTensor(fNY, ConvertStringToType(fAttrType), fShape);
         return;
      }
      // case of constant operator
      // in case of standard constant the shape is provided as input
      if (ConvertShapeToLength(fShape) != fValues.size())
         throw std::runtime_error("TMVA SOFIE Constant Op has invalid shape : " + ConvertShapeToString(fShape) +
                                 " with " + std::to_string(fValues.size()) + " values");

      // we need to create an initialized tensor of type constant to flag to not save it in a weight file
      // but keep its initialization in the generated code
      size_t length = ConvertShapeToLength(fShape);
      std::shared_ptr<void> data(malloc(length * sizeof(T)), free);
      std::memcpy(data.get(), (void*) fValues.data(), length * sizeof(T));
      model.AddInitializedTensor(fNY, ConvertStringToType(fAttrType), fShape, data);
   }

   std::string Generate(std::string OpName){
      OpName = "op_" + OpName;
      // if (!fIsConstantOfShape && fShape.empty()) {
      //    throw std::runtime_error("TMVA SOFIE Constant called to Generate without being initialized first");
      // }
      std::stringstream out;
      if (fIsConstantOfShape)
         out << "\n//------ ConstantOfShape\n";
      // else
      //    out << "\n//------ Constant\n";

      if (!fIsConstantOfShape) {
         // code is generated in RModel initialization
         // out << SP << "fTensor_" << fNY << " = {";
         // for (size_t i = 0; i < fValues.size(); i++) {
         //    out << fValues[i];
         //    if (i < fValues.size()-1) out << ", ";
         //    if  (i > 0 && i %10 == 0) out << "\n";
         // }
         // out << SP << "};\n";
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