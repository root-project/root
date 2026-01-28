#ifndef TMVA_SOFIE_ROPERATOR_SOFTPLUS
#define TMVA_SOFIE_ROPERATOR_SOFTPLUS

#include "TMVA/SOFIE_common.hxx"
#include "TMVA/ROperator.hxx"
#include "TMVA/RModel.hxx"

#include <sstream>

namespace TMVA{
namespace Experimental{
namespace SOFIE{

template <typename T>
class ROperator_Softplus final : public ROperator
{

private:

   std::string fNX;
   std::string fNY;
   std::vector<size_t> fShape;

public:
   ROperator_Softplus(){}
   ROperator_Softplus(std::string nameX, std::string nameY):
      fNX(UTILITY::Clean_name(nameX)), fNY(UTILITY::Clean_name(nameY)){
         fInputTensorNames = { fNX };
         fOutputTensorNames = { fNY };
      }

   std::vector<ETensorType> TypeInference(std::vector<ETensorType> input) override {
      return input;
   }

   std::vector<std::vector<size_t>> ShapeInference(std::vector<std::vector<size_t>> input) override {
      return input;
   }

   void Initialize(RModel& model) override {
      //input must be a graph input, or already initialized intermediate tensor.
      if (model.CheckIfTensorAlreadyExist(fNX) == false){
         throw std::runtime_error("TMVA SOFIE Softplus Op Input Tensor " + fNX + " is not found in model");
      }
      fShape = model.GetTensorShape(fNX);
      model.AddIntermediateTensor(fNY, model.GetTensorType(fNX), fShape);
   }

   std::string Generate(std::string OpName) override {
      OpName = "op_" + OpName;
      if (fShape.empty()){
         throw std::runtime_error("TMVA SOFIE Softplus operator called to Generate without being initialized first");
      }
      std::stringstream out;
      size_t length = ConvertShapeToLength(fShape);

      out << "\n//------ Softplus\n";
      out << SP << "for (int id = 0; id < " << length << " ; id++){\n";
      out << SP << SP << "float x = tensor_" << fNX << "[id];\n";
      out << SP << SP << "tensor_" << fNY << "[id] = (x >= 0x1.4000000000000p+4f) "
          << "? x : std::log1p(std::exp(x));\n";
      out << SP << "}\n";
      return out.str();
   }

   std::vector<std::string> GetStdLibs() override { return { std::string("cmath") };}
};

}//SOFIE
}//Experimental
}//TMVA


#endif //TMVA_SOFIE_ROPERATOR_SOFTPLUS
