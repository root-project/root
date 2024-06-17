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
struct BinaryOperatorTrait {};

template <typename T>
struct BinaryOperatorTrait<T, Add> {
   static const std::string Name() { return "Add"; }
   static std::string Op(const std::string & t1, const std::string t2) { return t1 + " + " + t2; }
};

template <typename T>
struct BinaryOperatorTrait<T, Sub> {
   static const std::string Name() { return "Sub"; }
   static std::string Op(const std::string & t1, const std::string t2) { return t1 + " - " + t2; }
};

template <typename T>
struct BinaryOperatorTrait<T, Mul> {
   static const std::string Name() { return "Mul"; }
   static std::string Op(const std::string & t1, const std::string t2) { return t1 + " * " + t2; }
};

template <typename T>
struct BinaryOperatorTrait<T, Div> {
   static const std::string Name() { return "Div"; }
   static std::string Op(const std::string & t1, const std::string t2) { return t1 + " / " + t2; }
};

template <typename T>
struct BinaryOperatorTrait<T, Pow> {
   static const std::string Name() { return "Pow"; }
   static std::string Op(const std::string & t1, const std::string t2) { return "std::pow(" + t1 + "," + t2 + ")"; }
};

template <typename T>
struct TensorType {};
template<>
struct TensorType<float> {
   static const std::string Name() { return "float"; }
};
template<>
struct TensorType<double> {
   static const std::string Name() { return "double"; }
};
template<>
struct TensorType<int64_t> {
   static const std::string Name() { return "int64_t"; }
};
template<>
struct TensorType<int32_t> {
   static const std::string Name() { return "int32_t"; }
};
template<>
struct TensorType<uint32_t> {
   static const std::string Name() { return "uint32_t"; }
};
template<>
struct TensorType<uint64_t> {
   static const std::string Name() { return "uint64_t"; }
};

template<typename T, EBasicBinaryOperator Op>
class ROperator_BasicBinary final : public ROperator{
private:

   std::string fNA;
   std::string fNB;
   std::string fNBroadcadstedA;
   std::string fNBroadcadstedB;
   std::string fNY;

   std::vector<size_t> fShapeA;
   std::vector<size_t> fShapeB;
   std::vector<size_t> fShapeY;

public:
   ROperator_BasicBinary(){}
   ROperator_BasicBinary(std::string nameA, std::string nameB, std::string nameY):
      fNA(UTILITY::Clean_name(nameA)), fNB(UTILITY::Clean_name(nameB)), fNY(UTILITY::Clean_name(nameY)){}

   // type of output given input
   std::vector<ETensorType> TypeInference(std::vector<ETensorType> input) override {
      return input;
   }

   // shape of output tensors given input tensors
   std::vector<std::vector<size_t>> ShapeInference(std::vector<std::vector<size_t>> input) override {
      // assume now inputs have same shape (no broadcasting)
      auto ret = std::vector<std::vector<size_t>>(1, input[0]); // return vector size 1 with first input
      return ret;
   }

   void Initialize(RModel& model) override {
      // input must be a graph input, or already initialized intermediate tensor
      if (!model.CheckIfTensorAlreadyExist(fNA)){
         throw std::runtime_error(std::string("TMVA SOFIE Binary Op Input Tensor ") + fNA + "is not found in model");
      }
      if (!model.CheckIfTensorAlreadyExist(fNB)) {
         throw std::runtime_error(std::string("TMVA SOFIE Binary Op Input Tensor ") + fNB + "is not found in model");
      }
      fShapeA = model.GetTensorShape(fNA);
      fShapeB = model.GetTensorShape(fNB);
      bool broadcast = !UTILITY::AreSameShape(fShapeA, fShapeB);
      if (broadcast) {
         // Y is the common shape of A and B
         fShapeY = UTILITY::UnidirectionalBroadcastShape(fShapeA, fShapeB);
         bool broadcastA = !UTILITY::AreSameShape(fShapeA, fShapeY);
         bool broadcastB = !UTILITY::AreSameShape(fShapeB, fShapeY);
         // Broadcast A to Y
         if (broadcastA) {
            if (model.IsInitializedTensor(fNA)) {
               auto data = model.GetInitializedTensorData(fNA);
               std::shared_ptr<void> broadcastedData(
                  UTILITY::UnidirectionalBroadcast<T>(static_cast<T *>(data.get()), fShapeA, fShapeY),
                  std::default_delete<T[]>());
               // Update the data and the shape of A
               model.UpdateInitializedTensor(fNA, model.GetTensorType(fNA), fShapeY, broadcastedData);
               fShapeA = fShapeY;
            } else {
               // Add an intermediate tensor for broadcasting A
               fNBroadcadstedA = "Broadcasted" + fNA;
               model.AddIntermediateTensor(fNBroadcadstedA, model.GetTensorType(fNA), fShapeY);
            }
         }
         // Broadcast B to Y
         if (broadcastB) {
            if (model.IsInitializedTensor(fNB)) {
               auto data = model.GetInitializedTensorData(fNB);
               std::shared_ptr<void> broadcastedData(
                  UTILITY::UnidirectionalBroadcast<T>(static_cast<T *>(data.get()), fShapeB, fShapeY),
                  std::default_delete<T[]>());
               // Update the data and the shape of B
               model.UpdateInitializedTensor(fNB, model.GetTensorType(fNB), fShapeY, broadcastedData);
               fShapeB = fShapeY;
            } else {
               // Add an intermediate tensor for broadcasting B
               fNBroadcadstedB = "Broadcasted" + fNB;
               model.AddIntermediateTensor(fNBroadcadstedB, model.GetTensorType(fNB), fShapeY);
            }
         }
      } else {
         fShapeY = fShapeA;
      }
      model.AddIntermediateTensor(fNY, model.GetTensorType(fNA), fShapeY);
   }

   std::string GenerateInitCode() override {
      std::stringstream out;
      return out.str();
   }

   std::string Generate(std::string OpName) override {
      OpName = "op_" + OpName;

      if (fShapeY.empty()) {
         throw std::runtime_error("TMVA SOFIE Binary Op called to Generate without being initialized first");
      }
      std::stringstream out;
      out << SP << "\n//------ " << BinaryOperatorTrait<T,Op>::Name() << "\n";
      size_t length = ConvertShapeToLength(fShapeY);
      std::string typeName = TensorType<T>::Name();
      // Broadcast A if it's uninitialized
      if (!fNBroadcadstedA.empty()) {
         out << SP << "// Broadcasting uninitialized tensor " << fNA << "\n";
         out << SP << "{\n";
         out << SP << SP << typeName << "* data = TMVA::Experimental::SOFIE::UTILITY::UnidirectionalBroadcast<" << typeName << ">(tensor_" << fNA << ", " << ConvertShapeToString(fShapeA) << ", " << ConvertShapeToString(fShapeY) << ");\n";
         out << SP << SP << "std::copy(data, data + " << length << ", tensor_" << fNBroadcadstedA << ");\n";
         out << SP << SP << "delete[] data;\n";
         out << SP << "}\n";
      }
      // Broadcast B if it's uninitialized
      if (!fNBroadcadstedB.empty()) {
         out << SP << "// Broadcasting uninitialized tensor " << fNB << "\n";
         out << SP << "{\n";
         out << SP << SP << typeName << "* data = TMVA::Experimental::SOFIE::UTILITY::UnidirectionalBroadcast<" << typeName << ">(tensor_" << fNB << ", " << ConvertShapeToString(fShapeB) << ", " << ConvertShapeToString(fShapeY) << ");\n";
         out << SP << SP << "std::copy(data, data + " << length << ", tensor_" << fNBroadcadstedB << ");\n";
         out << SP << SP << "delete[] data;\n";
         out << SP << "}\n";
      }
      const std::string& nameA = fNBroadcadstedA.empty()? fNA : fNBroadcadstedA;
      const std::string& nameB = fNBroadcadstedB.empty()? fNB : fNBroadcadstedB;
      out << SP << "for (size_t id = 0; id < " << length << " ; id++){\n";
      out << SP << SP << "tensor_" << fNY << "[id] = "  << BinaryOperatorTrait<T,Op>::Op( "tensor_" + nameA + "[id]" , "tensor_" + nameB + "[id]") <<  " ;\n";
      out << SP << "}\n";
      return out.str();
   }

   std::vector<std::string> GetStdLibs() override {
      if (Op == EBasicBinaryOperator::Pow) {
         return { std::string("cmath") };
      } else {
         return {};
      }
   }
};

}//SOFIE
}//Experimental
}//TMVA


#endif //TMVA_SOFIE_ROperator_BasicBinary
