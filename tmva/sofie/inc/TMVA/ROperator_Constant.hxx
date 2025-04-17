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
   std::vector<Dim> fDimShape;
   std::vector<Dim> fDimOutputShape;
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
      {
         fInputTensorNames = { };
         fOutputTensorNames = { };
      }

   std::vector<ETensorType> TypeInference(std::vector<ETensorType> input) override {
      return input;
   }

   std::vector<std::vector<size_t>> ShapeInference(std::vector<std::vector<size_t>> input) override {
      auto ret = input; //suggest copy to compiler
      return ret;
   }

   void Initialize(RModel& model) override {
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
         if (model.IsInitializedTensor(fNX) || model.IsConstantTensor(fNX)) {
            fIsOutputConstant = true;
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
         }
         else {
            // case of non constant tensors- we need to do at run time
            fDimShape = model.GetDimTensorShape(fNX);
            if (fDimShape.size() > 1 )
               throw std::runtime_error("TMVA SOFIE ConstantOfShape Op Input Tensor has invalid shape");
            if (!fDimShape[0].isParam) {
               fDimOutputShape.resize(fDimShape[0].dim);
               for (size_t i = 0; i < fDimShape[0].dim; i++) {
                  fDimOutputShape[i] = Dim{ std::string("s_") + fNY + "_" + std::to_string(i)};
               }
            }
            else {
               throw std::runtime_error("TMVA SOFIE ConstantOfShape Op Input Tensor has not defied shape");
            }
         }

      } else {
         // case of constant operator
         // in case of standard constant the shape is provided as input
         fIsOutputConstant = true;
         length = ConvertShapeToLength(fShape);
         if (length != fValues.size())
            throw std::runtime_error("TMVA SOFIE Constant Op has invalid shape : " + ConvertShapeToString(fShape) +
                                 " with " + std::to_string(fValues.size()) + " values");
      }

      // we need to create an initialized tensor of type constant to flag to not save it in a weight file
      // but keep its initialization in the generated code. The values might also be needed in initializing the
      // following operators using as input Constant or ConstantOfShape
       // resize fValues to shape length
      if (fIsOutputConstant) {
         model.AddConstantTensor(fNY, fShape, fValues);
         if (model.Verbose()) {
            std::cout << "adding constant tensor " << fNY << " with shape " << ConvertShapeToString(fShape)
            << " and values [";
            for (auto v : fValues) std::cout << " " << v;
            std::cout << "]" << std::endl;
         }
      } else {
         model.AddIntermediateTensor(fNY, ConvertStringToType(TensorType<T>::Name()), fDimOutputShape);
      }
   }

   std::string Generate(std::string opName) override {
      // no code to generate here. Tensor are defined in Session constructor
      if (fIsOutputConstant) {
         if (fNX.empty())
            return "// ---- Constant (no-op) \n";
         else
            return "// ---- ConstantOfShape (no-op) \n";
      }
      // Only ConstantOfShape might require generation code
      // generate constant tensor according to input
      std::stringstream out;
      out << "\n//--------- ConstantOfShape " << opName << "\n";
       // set shape values
      for (size_t i = 0; i < fDimOutputShape.size(); i++) {
         out << SP << "size_t " << fDimOutputShape[i].param << " = " << "tensor_" << fNX << "[" << i << "];\n";
      }
      auto length = ConvertDimShapeToLength(fDimOutputShape);
      // vector is already allocated- fill with values
      out << SP << "if (" << length << " > fTensor_" << fNY << ".size())\n";
      out << SP << SP << "fTensor_" << fNY << ".resize(" << length  << ");\n";
      out << SP << "std::fill(fTensor_" << fNY << ".begin(), fTensor_" << fNY << ".end(), " << fValues[0] << ");\n";
      return out.str();
   }
};

}//SOFIE
}//Experimental
}//TMVA


#endif //TMVA_SOFIE_ROPERATOR_Constant
