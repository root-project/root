#ifndef TMVA_SOFIE_ROPERATOR_TOPK
#define TMVA_SOFIE_ROPERATOR_TOPK

#include "TMVA/SOFIE_common.hxx"
#include "TMVA/ROperator.hxx"
#include "TMVA/RModel.hxx"

#include <vector>
#include <sstream>
#include <algorithm>

namespace TMVA{
namespace Experimental{
namespace SOFIE{

enum class ETopKOperator {Max, Min};

template<typename T, ETopKOperator Op>
struct TopKTraits {};

template<typename T>
struct TopKTraits<T, ETopKOperator::Max> {
   static const std::string Name() {return "TopK_Max";}
   static std::string Op(const std::string& res, const std::string& input, size_t k) {
      return "\t" + res + " = UTILITY::TopK<" + std::to_string(k) + ", " + typeid(T).name() + ", std::greater<T>>(" + input + ");\n";
   }
};

template<typename T>
struct TopKTraits<T, ETopKOperator::Min> {
   static const std::string Name() {return "TopK_Min";}
   static std::string Op(const std::string& res, const std::string& input, size_t k) {
      return "\t" + res + " = UTILITY::TopK<" + std::to_string(k) + ", " + typeid(T).name() + ", std::less<T>>(" + input + ");\n";
   }
};

template <typename T, ETopKOperator Op>
class ROperator_TopK final : public ROperator
{

private:

   int faxis = -1;
   int flargest = 1;
   int fsorted = 1;
   std::string fNX;
   std::string fNK;
   std::string fNY;
   std::vector<size_t> fShapeX;
   std::vector<size_t> fShapeY;

   bool fBroadcast = false;
   std::string fType;

public:
   ROperator_TopK(){}

   ROperator_TopK( int axis, int largest, int sorted, std::string& nameX, std::string& nameK std::string& nameY):
   faxis(axis), flargest(largest), fsorted(sorted),fNX(UTILITY::Clean_name(nameX)), fNK(UTILITY::Clean_name(nameK))
   fNY(UTILITY::Clean_name(nameY))
   {}

   // type of output given input
   std::vector<ETensorType> TypeInference(std::vector<ETensorType> input){
      ETensorType out = input[0];
      return {out};
   }

   // shape of output tensors given input tensors
   std::vector<std::vector<size_t>> ShapeInference(std::vector<std::vector<size_t>> input){
      
      auto ret = input; //suggest copy to compiler
      ret[0][faxis] = input[1];
      return ret;
   }

   void Initialize(RModel& model){
      if (!model.CheckIfTensorAlreadyExist(fNX)) {
         throw std::runtime_error("TMVA SOFIE TopK Op Input Tensor " + fNX + " is not found in model");
      }
      fShapeX = model.GetTensorShape(fNX);
      fShapeY = ShapeInference(fNX);
      model.AddIntermediateTensor(fNY, ETensorType::Vector, fShapeY);
   }

  std::string Generate(std::string OpName){
      OpName = "op_" + OpName;
      if (fShapeX.empty()) {
         throw std::runtime_error("TMVA SOFIE TopK operator called to Generate without being initialized first");
      }
      std::stringstream out;
      size_t length = ConvertShapeToLength(fShapeX);
      out << "\n//------ TOPK\n";
      out << SP << "for (int id = 0; id < " << length << " ; id++){\n";
      out << SP << SP << "tensor_" << fNY << "[id] = std::tanh(tensor_" << fNX << "[id]);\n";
      out << SP << "}\n";
      return out.str();
   }

};

} // namespace SOFIE
} // namespace Experimental
} // namespace TMVA

#endif