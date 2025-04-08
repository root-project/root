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
   static T Func(T t1, T t2) {return  t1 + t2;}
};

template <typename T>
struct BinaryOperatorTrait<T, Sub> {
   static const std::string Name() { return "Sub"; }
   static std::string Op(const std::string & t1, const std::string t2) { return t1 + " - " + t2; }
   static T Func (T t1, T t2) { return t1 - t2;}
};

template <typename T>
struct BinaryOperatorTrait<T, Mul> {
   static const std::string Name() { return "Mul"; }
   static std::string Op(const std::string & t1, const std::string t2) { return t1 + " * " + t2; }
   static T Func (T t1, T t2) { return  t1 * t2;}
};

template <typename T>
struct BinaryOperatorTrait<T, Div> {
   static const std::string Name() { return "Div"; }
   static std::string Op(const std::string & t1, const std::string t2) { return t1 + " / " + t2; }
   static T Func (T t1, T t2) { return t1/t2;}
};

template <typename T>
struct BinaryOperatorTrait<T, Pow> {
   static const std::string Name() { return "Pow"; }
   static std::string Op(const std::string & t1, const std::string t2) { return "std::pow(" + t1 + "," + t2 + ")"; }
   static T Func (T t1, T t2) { return std::pow(t1,t2);}
};

template<typename T, EBasicBinaryOperator Op>
class ROperator_BasicBinary final : public ROperator{
private:

   std::string fNA;
   std::string fNB;
   std::string fNBroadcastedA;
   std::string fNBroadcastedB;
   std::string fNY;

   std::vector<size_t> fShapeA;
   std::vector<size_t> fShapeB;
   std::vector<size_t> fShapeY;

   std::vector<Dim> fDimShapeA;
   std::vector<Dim> fDimShapeB;
   std::vector<Dim> fDimShapeY;

public:
   ROperator_BasicBinary(){}
   ROperator_BasicBinary(std::string nameA, std::string nameB, std::string nameY):
      fNA(UTILITY::Clean_name(nameA)), fNB(UTILITY::Clean_name(nameB)), fNY(UTILITY::Clean_name(nameY)){
         fInputTensorNames = { fNA, fNB };
         fOutputTensorNames = { fNY };
      }

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
      if (model.IsDynamicTensor(fNA))
         fDimShapeA = model.GetDynamicTensorShape(fNA);
      else {
         fShapeA = model.GetTensorShape(fNA);
         fDimShapeA = ConvertShapeToDim(fShapeA);
      }
      if (model.IsDynamicTensor(fNB))
         fDimShapeB = model.GetDynamicTensorShape(fNB);
      else {
         fShapeB = model.GetTensorShape(fNB);
         fDimShapeB = ConvertShapeToDim(fShapeB);
      }
      // check if need to broadcast at initialization time if shapes are known and different
      // (we could broadcast the tensor tensor to maximum values of dynamic shapes - to be done)
      // case of known shapes
      if (!fShapeA.empty() && !fShapeB.empty()) {
         auto ret = UTILITY::MultidirectionalBroadcastShape(fShapeA, fShapeB);
         fShapeY = ret.second;
         bool broadcast =  ret.first > 0;
         if (broadcast) {
            // Y is the common shape of A and B
            bool broadcastA = ret.first > 1;
            bool broadcastB = ret.first == 1 || ret.first == 3;
            // Broadcast A to Y
            if (broadcastA) {
               fNBroadcastedA = "Broadcasted" + fNA + "to" + fNY;
               if (model.IsInitializedTensor(fNA)) {
                  auto data = model.GetInitializedTensorData(fNA);
                  std::shared_ptr<void> broadcastedData(
                     UTILITY::UnidirectionalBroadcast<T>(static_cast<T *>(data.get()), fShapeA, fShapeY),
                     std::default_delete<T[]>());
                  // Update the data and the shape of A
                  model.AddConstantTensor(fNBroadcastedA, model.GetTensorType(fNA), fShapeY, broadcastedData);
                  fShapeA = fShapeY;
               } else {
                  // Add an intermediate tensor for broadcasting A
                  model.AddIntermediateTensor(fNBroadcastedA, model.GetTensorType(fNA), fShapeY);
               }
            }
            // Broadcast B to Y
            if (broadcastB) {
               fNBroadcastedB = "Broadcasted" + fNB + "to" + fNY;
               if (model.IsInitializedTensor(fNB)) {
                  auto data = model.GetInitializedTensorData(fNB);
                  if (model.Verbose())
                     std::cout << "data B " << ConvertShapeToString(fShapeB) << " : "
                               << ConvertValuesToString(ConvertShapeToLength(fShapeB), static_cast<T *>(data.get()))
                               << std::endl;
                  std::shared_ptr<void> broadcastedData(
                     UTILITY::UnidirectionalBroadcast<T>(static_cast<T *>(data.get()), fShapeB, fShapeY),
                     std::default_delete<T[]>());
                  // do not update tensor B but add broadcasted one (since it can be input to some other operators)
                  if (model.Verbose())
                     std::cout << "broadcasted data B " << ConvertShapeToString(fShapeY) << " : "
                               << ConvertValuesToString(ConvertShapeToLength(fShapeY),
                                                        static_cast<T *>(broadcastedData.get()))
                               << std::endl;
                  model.AddConstantTensor(fNBroadcastedB, model.GetTensorType(fNB), fShapeY, broadcastedData);
                  fShapeB = fShapeY;
               } else {
                  // Add an intermediate tensor for broadcasting B
                  model.AddIntermediateTensor(fNBroadcastedB, model.GetTensorType(fNB), fShapeY);
               }
            }
         } else {
            fShapeY = fShapeA;
         }
         // check case of constant  output (if all inputs are defined)
         if (model.IsInitializedTensor(fNA) && model.IsInitializedTensor(fNB)) {
            const std::string &nameA = fNBroadcastedA.empty() ? fNA : fNBroadcastedA;
            const std::string &nameB = fNBroadcastedB.empty() ? fNB : fNBroadcastedB;
            auto dataA = static_cast<T *>(model.GetInitializedTensorData(nameA).get());
            auto dataB = static_cast<T *>(model.GetInitializedTensorData(nameB).get());
            std::vector<T> dataY(ConvertShapeToLength(fShapeY));
            for (size_t i = 0; i < dataY.size(); i++) {
               dataY[i] = BinaryOperatorTrait<T, Op>::Func(dataA[i], dataB[i]);
            }
            model.AddConstantTensor<T>(fNY, fShapeY, dataY.data());
            // flag tensors to not be written in a fil
            model.SetNotWritableInitializedTensor(nameA);
            model.SetNotWritableInitializedTensor(nameB);
            fIsOutputConstant = true;
            if (model.Verbose())
               std::cout << "Binary op ---> " << fNY << "  " << ConvertShapeToString(fShapeY) << " : "
                         << ConvertValuesToString(dataY) << std::endl;
         } else {
            model.AddIntermediateTensor(fNY, model.GetTensorType(fNA), fShapeY);
         }
         // we convert non-dim shapes to Dim shapes
         fDimShapeY = ConvertShapeToDim(fShapeY);
      } // endif of non-parametric shapes
      else {
         // case A or B have dynamic shapes. We need to broadcast if shape are not same
         auto ret = UTILITY::MultidirectionalBroadcastShape(fDimShapeA, fDimShapeB);
         fDimShapeY = ret.second;
         if (ret.first > 1) {
            // case we broadcast A
            fNBroadcastedA = "Broadcasted" + fNA + "to" + fNY;
            model.AddIntermediateTensor(fNBroadcastedA, model.GetTensorType(fNA), fDimShapeY);
         }
         if (ret.first == 1 || ret.first == 3)  {
            // case we broadcast B
            fNBroadcastedB = "Broadcasted" + fNB + "to" + fNY;
            model.AddIntermediateTensor(fNBroadcastedB, model.GetTensorType(fNB), fDimShapeY);
         }
         // add output tensor
         model.AddIntermediateTensor(fNY, model.GetTensorType(fNA), fDimShapeY);
      }
   }

   std::string GenerateInitCode() override {
      std::stringstream out;
      return out.str();
   }

   std::string Generate(std::string OpName) override {

      if (fIsOutputConstant) return "";

      OpName = "op_" + OpName;

      if (fDimShapeY.empty()) {
         throw std::runtime_error("TMVA SOFIE Binary Op called to Generate without being initialized first");
      }
      std::stringstream out;
      out << SP << "\n//------ " << BinaryOperatorTrait<T,Op>::Name() << "\n";
      auto length = ConvertDimShapeToLength(fDimShapeY);
      std::string typeName = TensorType<T>::Name();
      // Broadcast A if it's uninitialized
      // use broadcasting function where we pass an already allocated tensor to minimize memory allocations
      if (!fNBroadcastedA.empty()) {
         out << SP << "// Broadcasting uninitialized tensor " << fNA << "\n";
         out << SP  << "TMVA::Experimental::SOFIE::UTILITY::UnidirectionalBroadcast<" << typeName << ">(tensor_" << fNA << ", "
                  << ConvertDimShapeToString(fDimShapeA) << ", " << ConvertDimShapeToString(fDimShapeY)
                  << ", fTensor_" << fNBroadcastedA << ");\n";
      }
      // Broadcast B if it's uninitialized
      if (!fNBroadcastedB.empty()) {
         out << SP << "// Broadcasting uninitialized tensor " << fNB << "\n";
         out << SP << "TMVA::Experimental::SOFIE::UTILITY::UnidirectionalBroadcast<" << typeName << ">(tensor_" << fNB << ", "
                  << ConvertDimShapeToString(fDimShapeB) << ", " << ConvertDimShapeToString(fDimShapeY)
                  << ", fTensor_" << fNBroadcastedB << ");\n";
      }
      const std::string& nameA = fNBroadcastedA.empty()? fNA : fNBroadcastedA;
      const std::string& nameB = fNBroadcastedB.empty()? fNB : fNBroadcastedB;
      out << SP << "for (size_t id = 0; id < " << length << " ; id++){\n";
      out << SP << SP << "tensor_" << fNY << "[id] = "
               << BinaryOperatorTrait<T,Op>::Op( "tensor_" + nameA + "[id]" , "tensor_" + nameB + "[id]")
               <<  " ;\n";
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
