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

   std::string fNInput;
   std::string fNY;
   std::vector<size_t> fShapeInput;

   bool fBroadcast = false;
   std::string fType;

public:
   ROperator_TopK(){}

   ROperator_TopK(const std::string& nameInput, const std::string& nameY):
   fNInput(UTILITY::Clean_name(nameInput)),
   fNY(UTILITY::Clean_name(nameY))
   {}

   // type of output given input
   std::vector<ETensorType> TypeInference(std::vector<ETensorType> input){
      return std::vector<ETensorType>{ETensorType::Vector};
   }

   // shape of output tensors given input tensors
   std::vector<std::vector<size_t>> ShapeInference(std::vector<std::vector<size_t>> input){
      auto ret = std::vector<std::vector<size_t>>(1, {0});
      return ret;
   }

   void Initialize(RModel& model){
      if (!model.CheckIfTensorAlreadyExist(fNInput)) {
         throw std::runtime_error("TMVA SOFIE TopK Op Input Tensor " + fNInput + " is not found in model");
      }
      fShapeInput = model.GetTensorShape(fNInput);
      model.AddIntermediateTensor(fNY, ETensorType::Vector, {0});
   }

   void Compile(RModel& model){
      std::stringstream code;
      auto k = model.GetAttribute<size_t>("k");
      auto opCode = TopKTraits<T, Op>::Op(fNY, fNInput, k);
      code << opCode;
      model.AddCodeToFunction(model.GetCurrentFunctionName(), code.str());
   }

   void Execute(RModel& model){
      // nothing to do here, as the operation is already computed during Compile
   }
};

} // namespace SOFIE
} // namespace Experimental
} // namespace TMVA

#endif
