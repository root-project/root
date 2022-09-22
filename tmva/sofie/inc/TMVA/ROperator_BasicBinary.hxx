#ifndef TMVA_SOFIE_ROperator_BasicBinary
#define TMVA_SOFIE_ROperator_BasicBinary

#include "TMVA/SOFIE_common.hxx"
#include "TMVA/ROperator.hxx"
#include "TMVA/RModel.hxx"

#include <sstream>

namespace TMVA{
namespace Experimental{
namespace SOFIE{

enum EBasicBinaryOperator { Add, Sub, Mul, Div, Pow };

template <typename T, EBasicBinaryOperator Op1>
struct BinaryOperatorTrait {
   const char *Name() { return ""; }
   const char *Op() { return ""; }
};
template <typename T>
struct BinaryOperatorTrait<T, Add> {
   static const char *Name() { return "Add"; }
   static std::string Op(const std::string & t1, const std::string t2) { return t1 + " + " + t2; }
};

template <typename T>
struct BinaryOperatorTrait<T, Sub> {
   static const char *Name() { return "Sub"; }
   static std::string Op(const std::string & t1, const std::string t2) { return t1 + " - " + t2; }
};

template <typename T>
struct BinaryOperatorTrait<T, Mul> {
   static const char *Name() { return "Mul"; }
   static std::string Op(const std::string & t1, const std::string t2) { return t1 + " * " + t2; }
};

template <typename T>
struct BinaryOperatorTrait<T, Div> {
   static const char *Name() { return "Div"; }
   static std::string Op(const std::string & t1, const std::string t2) { return t1 + " / " + t2; }
};

template <typename T>
struct BinaryOperatorTrait<T, Pow> {
   static const char *Name() { return "Pow"; }
   static std::string Op(const std::string & t1, const std::string t2) { return "std::pow(" + t1 + "," + t2 + ")"; }
};

template<typename T, EBasicBinaryOperator Op>
class ROperator_BasicBinary final : public ROperator{
private:

   std::string fNA;
   std::string fNB;
   std::string fNY;

   std::vector<size_t> fShapeA;
   std::vector<size_t> fShapeB;
   std::vector<size_t> fShapeY;

public:
   ROperator_BasicBinary(){}
   ROperator_BasicBinary(std::string nameA, std::string nameB, std::string nameY):
      fNA(UTILITY::Clean_name(nameA)), fNB(UTILITY::Clean_name(nameB)), fNY(UTILITY::Clean_name(nameY)){}

   // type of output given input
   std::vector<ETensorType> TypeInference(std::vector<ETensorType> input){
      return input;
   }

   // shape of output tensors given input tensors
   std::vector<std::vector<size_t>> ShapeInference(std::vector<std::vector<size_t>> input){
      // assume now inputs have same shape (no broadcasting)
      auto ret = std::vector<std::vector<size_t>>(1, input[0]); // return vector size 1 with first input
      return ret;
   }

   void Initialize(RModel& model) {
      // input must be a graph input, or already initialized intermediate tensor
      if (model.CheckIfTensorAlreadyExist(fNA) == false){
         throw std::runtime_error(std::string("TMVA SOFIE Binary Op Input Tensor ") + fNA + "is not found in model");
      }
      if (model.CheckIfTensorAlreadyExist(fNB) == false) {
         throw std::runtime_error(std::string("TMVA SOFIE Binary Op Input Tensor ") + fNB + "is not found in model");
      }
      fShapeA = model.GetTensorShape(fNA);
      fShapeB = model.GetTensorShape(fNB);
      // If the shape of 2 tensors are not same we perform bidirectional broadcasting.
      size_t lengthA = ConvertShapeToLength(fShapeA);
      size_t lengthB = ConvertShapeToLength(fShapeB);
      if (fShapeA.size() < fShapeB.size()) {
         // Broadcast the shape of A to the shape of B from the left to the right
         fShapeA = UTILITY::BidirectionalBroadcastShape(fShapeA, fShapeB);
         fShapeY = fShapeB;
      } else if (fShapeA.size() > fShapeB.size()) {
         // Broadcast the shape of B to the shape of A from the left to the right
         fShapeB = UTILITY::BidirectionalBroadcastShape(fShapeB, fShapeA);
         fShapeY = fShapeA;
      }
      if (lengthA < lengthB) {
         // Broadcast A to B
         fShapeY = fShapeB;
         auto data = model.GetInitializedTensorData(fNA);
         std::shared_ptr<void> broadcastedData(
            UTILITY::BidirectionalBroadcast<float>(static_cast<float *>(data.get()), fShapeA, fShapeB),
            std::default_delete<float[]>());
         // Update the data and the shape of A
         model.UpdateInitializedTensor(fNA, model.GetTensorType(fNA), fShapeB, broadcastedData);
         fShapeA = fShapeB;
      } else if (lengthB < lengthA) {
         // Broadcast B to A
         fShapeY = fShapeA;
         auto data = model.GetInitializedTensorData(fNB);
         std::shared_ptr<void> broadcastedData(
            UTILITY::BidirectionalBroadcast<float>(static_cast<float *>(data.get()), fShapeB, fShapeA),
            std::default_delete<float[]>());
         // Update the data and the shape of B
         model.UpdateInitializedTensor(fNB, model.GetTensorType(fNB), fShapeA, broadcastedData);
         fShapeB = fShapeA;
      } else {
         fShapeY = fShapeA;
      }
      model.AddIntermediateTensor(fNY, model.GetTensorType(fNA), fShapeY);
   }

   std::string Generate(std::string OpName){
      OpName = "op_" + OpName;

      if (fShapeY.empty()) {
         throw std::runtime_error("TMVA SOFIE Binary Op called to Generate without being initialized first");
      }
      std::stringstream out;
      size_t length = ConvertShapeToLength(fShapeY);
      out << "\n//------ " + std::string(BinaryOperatorTrait<T,Op>::Name())+"\n";
      out << SP << "for (size_t id = 0; id < " << length << " ; id++){\n";
      out << SP << SP << "tensor_" << fNY << "[id] = " +
      std::string(BinaryOperatorTrait<T,Op>::Op( "tensor_" + fNA + "[id]" , "tensor_" + fNB + "[id]")) +  " ;\n";
      out << SP << "}\n";
      return out.str();
   }

};

}//SOFIE
}//Experimental
}//TMVA


#endif //TMVA_SOFIE_ROperator_BasicBinary
