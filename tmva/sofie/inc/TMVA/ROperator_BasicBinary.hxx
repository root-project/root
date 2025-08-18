#ifndef TMVA_SOFIE_ROperator_BasicBinary
#define TMVA_SOFIE_ROperator_BasicBinary

#include "TMVA/SOFIE_common.hxx"
#include "TMVA/ROperator.hxx"
#include "TMVA/RModel.hxx"

#include <sstream>

namespace TMVA{
namespace Experimental{
namespace SOFIE{

enum EBasicBinaryOperator { Add, Sub, Mul, Div, Pow, Mod, FMod };

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
template <typename T>
struct BinaryOperatorTrait<T, Mod> {
   static const std::string Name() { return "Mod"; }
   static std::string Op(const std::string & t1, const std::string t2) { return "(" + t1 + " % " + t2 + ")"; }
   static T Func (T t1, T t2) { return std::pow(t1,t2);}
};
template <typename T>
struct BinaryOperatorTrait<T, FMod> {
   static const std::string Name() { return "FMod"; }
   static std::string Op(const std::string & t1, const std::string t2) { return "std::fmod(" + t1 + "," + t2 + ")"; }
   static T Func (T t1, T t2) { return std::pow(t1,t2);}
};

template<typename T, EBasicBinaryOperator Op>
class ROperator_BasicBinary final : public ROperator{
private:

   int fBroadcastFlag = 0;
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
      int dynamicInputs = 0;
      if (model.IsDynamicTensor(fNA)) {
         fDimShapeA = model.GetDynamicTensorShape(fNA);
         dynamicInputs |= 1;
      } else {
         fShapeA = model.GetTensorShape(fNA);
         fDimShapeA = ConvertShapeToDim(fShapeA);
      }
      if (model.IsDynamicTensor(fNB)) {
         dynamicInputs |= 2;
         fDimShapeB = model.GetDynamicTensorShape(fNB);
      } else {
         fShapeB = model.GetTensorShape(fNB);
         fDimShapeB = ConvertShapeToDim(fShapeB);
      }
      if (dynamicInputs & 1 && model.Verbose() )
         std::cout <<  BinaryOperatorTrait<T, Op>::Name() << " : input " << fNA << " is dynamic " << ConvertShapeToString(fDimShapeA) << "  ";
      if (dynamicInputs & 2 && model.Verbose())
         std::cout <<  BinaryOperatorTrait<T, Op>::Name() << " : input " << fNB << " is dynamic " << ConvertShapeToString(fDimShapeB) << "  ";
      std::cout << std::endl;
      // check if need to broadcast at initialization time if shapes are known and different
      // (we could broadcast the tensor tensor to maximum values of dynamic shapes - to be done)
      // case of known shapes
      if (dynamicInputs == 0) {
         auto ret = UTILITY::MultidirectionalBroadcastShape(fShapeA, fShapeB);
         fBroadcastFlag = ret.first;
         fShapeY = ret.second;
         bool broadcast =  ret.first > 0;
         if (broadcast) {
            // Y is the common shape of A and B
            bool broadcastA = ret.first & 2;
            bool broadcastB = ret.first & 1;
            // Broadcast A to Y
            if (broadcastA) {
               fNBroadcastedA = "Broadcasted" + fNA + "to" + fNY;
               if (model.IsConstantTensor(fNA)) {
                  auto data = model.GetInitializedTensorData(fNA);
                  std::shared_ptr<void> broadcastedData(
                     UTILITY::UnidirectionalBroadcast<T>(static_cast<T *>(data.get()), fShapeA, fShapeY),
                     std::default_delete<T[]>());
                  // Update the data and the shape of A
                  model.AddConstantTensor(fNBroadcastedA, model.GetTensorType(fNA), fShapeY, broadcastedData);
                  fShapeA = fShapeY;
                  fDimShapeA = ConvertShapeToDim(fShapeA);
               } else {
                  // Add an intermediate tensor for broadcasting A
                  model.AddIntermediateTensor(fNBroadcastedA, model.GetTensorType(fNA), fShapeY);
               }
            }
            // Broadcast B to Y
            if (broadcastB) {
               fNBroadcastedB = "Broadcasted" + fNB + "to" + fNY;
               if (model.IsConstantTensor(fNB)) {
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
                  fDimShapeB = ConvertShapeToDim(fShapeB);
               } else {
                  // Add an intermediate tensor for broadcasting B
                  model.AddIntermediateTensor(fNBroadcastedB, model.GetTensorType(fNB), fShapeY);
               }
            }
         } else {
            fShapeY = fShapeA;
         }
         // check case of constant  output (if all inputs are defined)
         if (model.IsConstantTensor(fNA) && model.IsConstantTensor(fNB)) {
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
            if (model.Verbose()) {
               std::cout << BinaryOperatorTrait<T, Op>::Name() << " : " << fNA <<  "  " << ConvertShapeToString(fShapeA)
                     << " , " << fNB << "  " << ConvertShapeToString(fShapeB) << " ---> " << fNY
                     << "  " << ConvertShapeToString(fShapeY) << " : " << ConvertValuesToString(dataY) << std::endl;
            }
         } else {
            model.AddIntermediateTensor(fNY, model.GetTensorType(fNA), fShapeY);
            if (model.Verbose()) {
               std::cout << BinaryOperatorTrait<T, Op>::Name() << " : " << fNA <<  "  " << ConvertShapeToString(fShapeA)
                     << " , " << fNB << "  " << ConvertShapeToString(fShapeB) << " ---> " << fNY
                     << "  " << ConvertShapeToString(fShapeY) << std::endl;
            }
         }
         // we convert non-dim shapes to Dim shapes
         fDimShapeY = ConvertShapeToDim(fShapeY);
      } // endif of non-parametric shapes
      else {
         // case A or B have dynamic shapes. We need to broadcast if shape are not same
         auto ret = UTILITY::MultidirectionalBroadcastShape(fDimShapeA, fDimShapeB);
         fBroadcastFlag = ret.first;
         fDimShapeY = ret.second;
         // case of all parametric shapes and MultiDirectionalBroadcastShape  return the max of the 2
         // need to do before we declare the output tensor shape and the broadcasted ones
         if (ret.first & 4) {
            // check if one of the parameter is an input dimension
            // define function to find this
            auto IsInputDimParam = [&](const std::string & p) {
               auto inputNames = model.GetInputTensorNames();
               for (auto & input : inputNames) {
                  for (auto & i_s : model.GetDimTensorShape(input)) {
                     if (i_s.isParam && i_s.param == p) return true;
                  }
               }
               return false;
            };
            for (size_t i = 0; i < fDimShapeY.size(); i++) {
               auto & s = fDimShapeY[i];
               if (s.isParam && s.param.find("std::max") != std::string::npos) {
                  if (IsInputDimParam(fDimShapeA[i].param)) {
                     // case dim is 1 we indicate that the input parameter is equal to 1
                     if (fDimShapeA[i].dim != 1)
                        s = fDimShapeA[i];
                     else
                        s = fDimShapeB[i];
                  } else if (IsInputDimParam(fDimShapeB[i].param)) {
                     if (fDimShapeB[i].dim != 1)
                        s = fDimShapeB[i];
                     else
                        s = fDimShapeA[i];
                  }
               }
            }
         }
         if (ret.first & 2) {
            // case we broadcast A
            fNBroadcastedA = "Broadcasted" + fNA + "to" + fNY;
            model.AddIntermediateTensor(fNBroadcastedA, model.GetTensorType(fNA), fDimShapeY);
         }
         if (ret.first & 1)  {
            // case we broadcast B
            fNBroadcastedB = "Broadcasted" + fNB + "to" + fNY;
            model.AddIntermediateTensor(fNBroadcastedB, model.GetTensorType(fNB), fDimShapeY);
         }

         model.AddIntermediateTensor(fNY, model.GetTensorType(fNA), fDimShapeY);
         if (model.Verbose()) {
            std::cout << BinaryOperatorTrait<T, Op>::Name() << " : " << ConvertShapeToString(fDimShapeA) << " , "
                      << ConvertShapeToString(fDimShapeB) << " --> " << ConvertShapeToString(fDimShapeY) << std::endl;
         }
      }
   }

   std::string GenerateInitCode() override {
      std::stringstream out;
      return out.str();
   }

   std::string Generate(std::string opName) override {

      if (fIsOutputConstant) return "";

      opName = "op_" + opName;

      if (fDimShapeY.empty()) {
         throw std::runtime_error("TMVA SOFIE Binary Op called to Generate without being initialized first");
      }
      std::stringstream out;
      out << SP << "\n//------ " << BinaryOperatorTrait<T,Op>::Name() << "\n";
      auto length = ConvertDimShapeToLength(fDimShapeY);
      std::string typeName = TensorType<T>::Name();
      // we need to check if we can broadcast (case flag has bit 4 set)
      if (fBroadcastFlag & 4) {
         // need to check if shapes are the same
         auto lengthA = ConvertDimShapeToLength(fDimShapeA);
         auto lengthB = ConvertDimShapeToLength(fDimShapeB);
         out << SP << "if (" << lengthA << "!=" << lengthB << ") {\n";
         // check if A->B or B->A
         //bool broadcastable = true;
         for (size_t i = 0; i < fDimShapeY.size(); i++) {
            if (fBroadcastFlag & 5 && fDimShapeY[i] == fDimShapeA[i] && fDimShapeA[i].dim > 1 && fDimShapeB[i].isParam) {
               // B->A B[i] needs to be 1
               out << SP << SP << "if (" << fDimShapeB[i] << "!= 1)\n";
               out << SP << SP << SP << "throw std::runtime_error(\"SOFIE - Cannot broadcast B->A in operator "
                                        << opName << "\");\n";
            }
            if (fBroadcastFlag & 6 && fDimShapeY[i] == fDimShapeB[i] && fDimShapeB[i].dim > 1 && fDimShapeA[i].isParam) {
               //A-> B A[i] needs to be 1
               out << SP << SP << "if (" << fDimShapeA[i] << "!= 1)\n";
               out << SP << SP << SP << "throw std::runtime_error(\"SOFIE - Cannot broadcast A->B in operator "
                                        << opName << "\");\n";
            }
            else if (fDimShapeA[i].isParam && fDimShapeB[i].isParam) {
               // both shapes are parametric and we broadcast to maximum
               // we allocate here output vector
               out << SP << SP << "if (" << fDimShapeA[i] << " != " << fDimShapeB[i] << " && ("
                   << fDimShapeA[i] << " != 1 || " << fDimShapeB[i] << " != 1))\n";
               out << SP << SP << "throw std::runtime_error(\"SOFIE - Cannot broadcast shapes in operator "
                                        << opName << "\");\n";
            }
         }
      } else {
         out << SP << "{\n";
      }
      // Broadcast A if it's uninitialized
      // use broadcasting function where we pass an already allocated tensor to minimize memory allocations
      // in case of A is  constant tensor is automatically broadcasted in Initialize() then shapeA == shapeY
      if (!fNBroadcastedA.empty() && (fDimShapeA != fDimShapeY)) {
         out << SP << SP << "// Broadcasting uninitialized tensor " << fNA << "\n";
         out << SP << SP  << "TMVA::Experimental::SOFIE::UTILITY::UnidirectionalBroadcast<" << typeName << ">(tensor_" << fNA << ", "
                  << ConvertDimShapeToString(fDimShapeA) << ", " << ConvertDimShapeToString(fDimShapeY)
                  << ", fTensor_" << fNBroadcastedA << ");\n";
      }
      // Broadcast B if it's uninitialized
      if (!fNBroadcastedB.empty() && (fDimShapeB != fDimShapeY)) {
         out << SP << SP << "// Broadcasting uninitialized tensor " << fNB << "\n";
         out << SP << SP << "TMVA::Experimental::SOFIE::UTILITY::UnidirectionalBroadcast<" << typeName << ">(tensor_" << fNB << ", "
                  << ConvertDimShapeToString(fDimShapeB) << ", " << ConvertDimShapeToString(fDimShapeY)
                  << ", fTensor_" << fNBroadcastedB << ");\n";
      }
      out << SP << "}\n"; // end if on broadcasting
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
