#ifndef TMVA_SOFIE_ROPERATOR_GELU
#define TMVA_SOFIE_ROPERATOR_GELU

#include "TMVA/SOFIE_common.hxx"
#include "TMVA/ROperator.hxx"
#include "TMVA/RModel.hxx"

#include <sstream>

namespace TMVA{
namespace Experimental{
namespace SOFIE{

template <typename T>
class ROperator_Gelu final : public ROperator
{

private:

   std::string fNX;
   std::string fNY;
   std::vector<Dim> fShape;

public:
   ROperator_Gelu(){}
   ROperator_Gelu(std::string nameX, std::string nameY):
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
      if (model.CheckIfTensorAlreadyExist(fNX) == false){   //input must be a graph input, or already initialized intermediate tensor
         throw std::runtime_error("TMVA SOFIE Gelu Op Input Tensor " + fNX + " is not found in model");
      }

      fShape = model.GetDimTensorShape(fNX);

      model.AddIntermediateTensor(fNY, model.GetTensorType(fNX), fShape);
      if (model.Verbose()) {
         std::cout << "Gelu : " << fNX << " -> " << fNY << " " << ConvertShapeToString(fShape) << std::endl;
      }
   }

   std::string Generate(std::string OpName) override {
     OpName = "op_" + OpName;
     if (fShape.empty()) {
        throw std::runtime_error("TMVA SOFIE Operator Gelu called to Generate without being initialized first");
     }
     std::stringstream out;
     auto length = ConvertDynamicShapeToLength(fShape);
     out << "\n//------ GELU\n";
     out << SP << "for (int id = 0; id < " << length << " ; id++){\n";
     out << SP << SP
         << "tensor_" << fNY << "[id] = 0.5 * tensor_" << fNX << "[id] * "
         << "(1 + std::tanh(0.7978845608 * "
         << "(tensor_" << fNX << "[id] + 0.044715 * "
         << "tensor_" << fNX << "[id] * tensor_" << fNX << "[id] * tensor_" << fNX << "[id])));\n";
     out << SP << "}\n";
     return out.str();
   }

};

}//SOFIE
}//Experimental
}//TMVA


#endif //TMVA_SOFIE_ROPERATOR_GELU
