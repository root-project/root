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
      size_t length = 1;
      if (!fNX.empty()) {
         // case of ConstantOfShape (since no inputs in case of Constant operator)
         fIsConstantOfShape  = true;
         if (model.CheckIfTensorAlreadyExist(fNX) == false){
           throw std::runtime_error("TMVA SOFIE ConstantOfShape Op Input Tensor is not found in model");
         }
         // get output shape from input values:
         // can work only if input is a constant or initialized tensor (or dynamic one)
         auto dptr = model.GetInitializedTensorData(fNX);
         auto input_tensor = static_cast<int64_t *>(dptr.get());
         auto input_shape = model.GetTensorShape(fNX);
         if (input_shape.size() > 1 )
            throw std::runtime_error("TMVA SOFIE ConstantOfShape Op Input Tensor has invalid shape");
         if (input_tensor != nullptr && !input_shape.empty()) {
            fShape = std::vector<size_t> (input_shape[0]);
            for (size_t i = 0; i < fShape.size(); i++)
               fShape[i] = input_tensor[i];
         } else
            fShape = {1};  // scalar case

         length = ConvertShapeToLength(fShape);
         if (fValues.size() != 1)
            throw std::runtime_error("TMVA SOFIE ConstantOfShape Op value Tensor has invalid size " + std::to_string(fValues.size()));

         T value = fValues[0];
         fValues = std::vector<T>(length, value);

      } else {
         // case of constant operator
         // in case of standard constant the shape is provided as input
         length = ConvertShapeToLength(fShape);
         if (length != fValues.size())
            throw std::runtime_error("TMVA SOFIE Constant Op has invalid shape : " + ConvertShapeToString(fShape) +
                                 " with " + std::to_string(fValues.size()) + " values");
      }

      // we need to create an initialized tensor of type constant to flag to not save it in a weight file
      // but keep its initialization in the generated code. The values might also be needed in initializing the
      // following operators using as input Constant or ConstantOfShape
       // resize fValues to shape length

      std::shared_ptr<void> data(malloc(length * sizeof(T)), free);
      std::memcpy(data.get(), (void*) fValues.data(), length * sizeof(T));
      model.AddConstantTensor(fNY, ConvertStringToType(fAttrType), fShape, data);
   }

   std::string Generate(std::string /* OpName */){
      // no code to generate here. Tensor are defined in Session constructor
      return "//---------------------------------------\n";
   }
};

}//SOFIE
}//Experimental
}//TMVA


#endif //TMVA_SOFIE_ROPERATOR_Constant