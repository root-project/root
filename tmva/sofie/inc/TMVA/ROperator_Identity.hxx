#ifndef TMVA_SOFIE_ROPERATOR_IDENTITY
#define TMVA_SOFIE_ROPERATOR_IDENTITY

#include "TMVA/SOFIE_common.hxx"
#include "TMVA/ROperator.hxx"
#include "TMVA/RModel.hxx"

#include <sstream>

namespace TMVA{
namespace Experimental{
namespace SOFIE{

template <typename T>
class ROperator_Identity final : public ROperator
{

private:

   bool fIsInputInitialized = false;
   bool fIsOutputInitialized = false; // the output is the same weight as the input
   std::string fNX;
   std::string fNY;
   std::vector<Dim> fShape;

public:
   ROperator_Identity(){}
   ROperator_Identity(std::string nameX, std::string nameY):
      fNX(UTILITY::Clean_name(nameX)), fNY(UTILITY::Clean_name(nameY)){
         fInputTensorNames = { fNX };
         fOutputTensorNames = { fNY };
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
      if (model.CheckIfTensorAlreadyExist(fNX) == false){
        throw std::runtime_error("TMVA SOFIE Identity Op Input Tensor is not found in model");
      }
      fShape = model.GetDimTensorShape(fNX);
      if (model.IsInitializedTensor(fNX)) {
         // we need to check if is a weight (initialized) or a constant tensor
         // in the first case we need to create a constant tensor with the output, in teh second we
         // need to generate the identy code in the GenerateInitCode
         if (model.IsConstantTensor(fNX)) {
            auto inputData = static_cast<T*>(model.GetInitializedTensorData(fNX).get());
            model.AddConstantTensor<T>(fNY, model.GetTensorShape(fNX), inputData);
            fIsOutputConstant = true;
         } else {
            // the output is the same weight under another name (exporters emit this for a
            // shared parameter); registering it as an initialized tensor keeps it resolvable
            // while the code is generated, as BatchNormalization needs its scale to be
            fIsOutputInitialized = true;
            model.AddInitializedTensor(fNY, model.GetTensorType(fNX), model.GetTensorShape(fNX),
                                       model.GetInitializedTensorData(fNX));
         }
      } else {
         model.AddIntermediateTensor(fNY, model.GetTensorType(fNX), fShape);
      }
   }

   std::string GenerateInitCode() override {
      // generate init code for identity operator
      if (!fIsInputInitialized) return "";
      std::stringstream out;
      out << "\n//------ IDENTITY\n";
      // just copy the tensor pointers
      out << SP << SP << "tensor_" << fNY << " = tensor_" << fNX << ";\n";
      return out.str();
   }


   std::string Generate(std::string OpName) override {
      if (fIsOutputConstant || fIsInputInitialized || fIsOutputInitialized)
         return "";
      OpName = "op_" + OpName;
      if (fShape.empty()) {
         throw std::runtime_error("TMVA SOFIE Operator Identity called to Generate without being initialized first");
      }
      std::stringstream out;
      out << "\n//------ IDENTITY\n";
      out << SP << "std::copy(tensor_" << fNX << ", tensor_" << fNX << " + " << ConvertDimShapeToLength(fShape)
          << ", tensor_" << fNY << ");\n";
      return out.str();
   }

};

}//SOFIE
}//Experimental
}//TMVA


#endif //TMVA_SOFIE_ROPERATOR_IDENTITY
