#ifndef TMVA_SOFIE_ROPERATOR_RANGE
#define TMVA_SOFIE_ROPERATOR_RANGE

#include "TMVA/SOFIE_common.hxx"
#include "TMVA/ROperator.hxx"
#include "TMVA/RModel.hxx"

#include <sstream>

namespace TMVA{
namespace Experimental{
namespace SOFIE{

template <typename T>
class ROperator_Range final : public ROperator
{
private:

   std::string fNStart;
   std::string fNLimit;
   std::string fNDelta;
   std::string fNOutput;
   std::vector<Dim> fShape;
   std::string fType;

public:
   ROperator_Range(){}

   ROperator_Range(std::string start, std::string limit, std::string delta, std::string nameOutput):
      fNStart(start), fNLimit(limit), fNDelta(delta),
      fNOutput(UTILITY::Clean_name(nameOutput)) {
      if (std::is_same<T, float>::value) {
          fType = "float";
      } else if (std::is_same<T, int64_t>::value) {
          fType = "int64_t";
      }
      static_assert( (std::is_same_v<T, float> || std::is_same_v<T, int64_t>),
                  "TMVA::SOFIE - Unsupported type by Range operator");
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
      if (!model.CheckIfTensorAlreadyExist(fNStart)) {
         throw
            std::runtime_error("TMVA SOFIE Range Op Input Tensor " + fNStart + "is not found in model");
      }
      if (!model.CheckIfTensorAlreadyExist(fNLimit)) {
         throw
            std::runtime_error("TMVA SOFIE Range Op Input Tensor " + fNLimit + "is not found in model");
      }
      if (!model.CheckIfTensorAlreadyExist(fNDelta)) {
         throw
            std::runtime_error("TMVA SOFIE Range Op Input Tensor " + fNDelta + "is not found in model");
      }
      ETensorType type = ConvertStringToType(fType);
      fShape = {Dim{"range_size"}};
      model.AddDynamicTensor(fNOutput, type, fShape);
   }

   std::string Generate(std::string OpName) override {
      OpName = "op_" + OpName;
      if (fShape.empty()) {
         throw std::runtime_error("TMVA SOFIE Range operator called to Generate without being initialized first");
      }
      std::stringstream out;
      out << "\n//------ Range\n";
      std::string sizeName = fShape[0].param;
      out << SP << "size_t " << sizeName << " = static_cast<size_t>(std::max(std::ceil((static_cast<float>(*tensor_" << fNLimit << ") - static_cast<float>(*tensor_" << fNStart << ")) / static_cast<float>(*tensor_" << fNDelta << ")), 0.0f));\n";
      out << SP << "if (" << sizeName << " > " << "fTensor_" << fNOutput << ".size() ){\n";
      out << SP << SP << "fTensor_" << fNOutput << ".resize(" << sizeName << ");\n";
      // need to re-initialized pointer to tensor data
      out << SP << SP << "tensor_" << fNOutput << " = fTensor_" << fNOutput << ".data();\n";
      out << SP << "}\n";
      out << SP << "for (size_t i = 0; i < " << sizeName << "; i++) {\n";
      out << SP << SP << "fTensor_" << fNOutput << "[i] = *tensor_" << fNStart << " + i * (*tensor_" << fNDelta << ");\n";
      out << SP << "}\n";
      return out.str();
   }
};

}//SOFIE
}//Experimental
}//TMVA

#endif //TMVA_SOFIE_ROPERATOR_RANGE
