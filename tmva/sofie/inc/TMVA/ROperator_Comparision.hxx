
#ifndef TMVA_SOFIE_ROperator_Comparision
#define TMVA_SOFIE_ROperator_Comparision

#include "TMVA/SOFIE_common.hxx"
#include "TMVA/ROperator.hxx"
#include "TMVA/RModel.hxx"

#include <sstream>

namespace TMVA{
namespace Experimental{
namespace SOFIE{

enum EComparisionOperator { Eq, Less, LessEq, Greater, GreaterEq };

template <typename T, EComparisionOperator Op1>
struct ComparisionTrait{};

template <typename T>
struct ComparisionTrait<T, Eq> {
   static const std::string Name() { return "Equal"; }
   static std::string Op(const std::string & t1, const std::string t2) { return t1 + " == " +  t2 + " ? true : false "; }
   static bool Result(T v1, T v2) { return v1 == v2;}
};

template <typename T>
struct ComparisionTrait<T, Less> {
   static const std::string Name() { return "Less"; }
   static std::string Op(const std::string & t1, const std::string t2) { return t1 + " < " + t2 + " ? true : false "; }
   static bool Result(T v1, T v2) { return v1 < v2;}
};

template <typename T>
struct ComparisionTrait<T, LessEq> {
   static const std::string Name() { return "LessOrEqual"; }
   static std::string Op(const std::string & t1, const std::string t2) { return t1 + " <= " +  t2 + " ? true : false ";  }
   static bool Result(T v1, T v2) { return v1 <= v2;}
};

template <typename T>
struct ComparisionTrait<T, Greater> {
   static const std::string Name() { return "Greater"; }
   static std::string Op(const std::string & t1, const std::string t2) { return  t1 + " > " +  t2 + " ? true : false "; }
   static bool Result(T v1, T v2) { return v1 > v2;}
};

template <typename T>
struct ComparisionTrait<T, GreaterEq> {
   static const std::string Name() { return "GreaterOrEqual"; }
   static std::string Op(const std::string & t1, const std::string t2) { return t1 + " >= " +  t2 + " ? true : false " ; }
   static bool Result(T v1, T v2) { return v1 >= v2;}
};

template<typename T, EComparisionOperator Op>
class ROperator_Comparision final : public ROperator{
private:

   bool fIsModelOutput = false;
   std::string fNX1;
   std::string fNX2;
   std::string fNY;
   std::vector<size_t> fShapeX1;
   std::vector<size_t> fShapeX2;
   std::vector<size_t> fShapeY;
   std::string fNBroadcastedX1;
   std::string fNBroadcastedX2;
   ETensorType fTensorType1 = ETensorType::UNDEFINED;
   ETensorType fTensorType2 = ETensorType::UNDEFINED;
   bool fBroadcast = false;


public:
   ROperator_Comparision(){}
   ROperator_Comparision(const std::string & nameX1, const std::string & nameX2, const std::string & nameY):
      fNX1(UTILITY::Clean_name(nameX1)), fNX2(UTILITY::Clean_name(nameX2)), fNY(UTILITY::Clean_name(nameY)){
         fInputTensorNames = { fNX1, fNX2 };
         
         // output will be a boolean vector so should not be considered for memory optimized pool
         fOutputTensorNames = { fNY };
      }

   // type of output given input
   std::vector<ETensorType> TypeInference(std::vector<ETensorType> input) override {
      return input;
   }

   // shape of output tensors given input tensors
   std::vector<std::vector<size_t>> ShapeInference(std::vector<std::vector<size_t>> input) override {
      auto ret = input; // return vector size 1 with first input
      return ret;
   }

   void Initialize(RModel& model) override {
      // input must be a graph input, or already initialized intermediate tensor
      if (!model.CheckIfTensorAlreadyExist(fNX1)){
         throw std::runtime_error(std::string("TMVA SOFIE Comparision Op Input Tensor ") + fNX1 + "is not found in model");
      }
      if (!model.CheckIfTensorAlreadyExist(fNX2)) {
         throw std::runtime_error(std::string("TMVA SOFIE Comparision Op Input Tensor ") + fNX2 + "is not found in model");
      }
      fShapeX1 = model.GetTensorShape(fNX1);
      fShapeX2 = model.GetTensorShape(fNX2);
      fTensorType1 = model.GetTensorType(fNX1);
      fTensorType2 = model.GetTensorType(fNX2);
      bool broadcast = !UTILITY::AreSameShape(fShapeX1, fShapeX2);
      if (broadcast) {
         // Y is the common shape of A and B
         fShapeY = UTILITY::UnidirectionalBroadcastShape(fShapeX1, fShapeX2);
         bool broadcastX1 = !UTILITY::AreSameShape(fShapeX1, fShapeY);
         bool broadcastX2 = !UTILITY::AreSameShape(fShapeX2, fShapeY);
         // Broadcast A to Y
         if (broadcastX1) {
            if (model.IsInitializedTensor(fNX1)) {
               auto data = model.GetInitializedTensorData(fNX1);
               std::shared_ptr<void> broadcastedData(
                  UTILITY::UnidirectionalBroadcast<T>(static_cast<T *>(data.get()), fShapeX1, fShapeY),
                  std::default_delete<T[]>());
               // Update the data and the shape of A
               model.UpdateInitializedTensor(fNX1, model.GetTensorType(fNX1), fShapeY, broadcastedData);
               fShapeX1 = fShapeY;
            } else {
               // Add an intermediate tensor for broadcasting A
               fNBroadcastedX1 = "Broadcasted" + fNX1;
               model.AddIntermediateTensor(fNBroadcastedX1, model.GetTensorType(fNX1), fShapeY);
            }
         }
         // Broadcast B to Y
         if (broadcastX2) {
            if (model.IsInitializedTensor(fNX2)) {
               auto data = model.GetInitializedTensorData(fNX2);
               std::shared_ptr<void> broadcastedData(
                  UTILITY::UnidirectionalBroadcast<T>(static_cast<T *>(data.get()), fShapeX2, fShapeY),
                  std::default_delete<T[]>());
               // Update the data and the shape of B
               model.UpdateInitializedTensor(fNX2, model.GetTensorType(fNX2), fShapeY, broadcastedData);
               fShapeX2 = fShapeY;
            } else {
               // Add an intermediate tensor for broadcasting B
               fNBroadcastedX2 = "Broadcasted" + fNX2;
               model.AddIntermediateTensor(fNBroadcastedX2, model.GetTensorType(fNX2), fShapeY);
            }
         }
      } else {
         fShapeY = fShapeX1;
      }
      // case of constant tensors
      if (model.IsInitializedTensor(fNX1) && model.IsInitializedTensor(fNX2) ) {
         fIsOutputConstant = true;
         auto data1 = static_cast<T *>(model.GetInitializedTensorData(fNX1).get());
         auto data2 = static_cast<T *>(model.GetInitializedTensorData(fNX2).get());
         size_t length = ConvertShapeToLength(fShapeY);
         bool * outData = new bool[length];
         for (size_t i = 0; i < length; i++)
            outData[i] = ComparisionTrait<T,Op>::Result(data1[i], data2[i]);
         model.AddConstantTensor(fNY, fShapeY, outData);
         if (model.Verbose())
            std::cout <<  ComparisionTrait<T,Op>::Name() << " op ---> " << fNY << "  " << ConvertShapeToString(fShapeY) << " : "
               << ConvertValuesToString(length,outData) << std::endl;
         delete [] outData;
      } else {
         model.AddIntermediateTensor(fNY, ETensorType::BOOL , fShapeY);
      }
      // check if this is not output operators to add a specific line for definining the tensor_xxx variable
      const auto & outputTensorNames = model.GetOutputTensorNames();
      fIsModelOutput = false;
      if (std::find(outputTensorNames.begin(), outputTensorNames.end(), fNY) != outputTensorNames.end())
         fIsModelOutput = true;
   }

   std::string Generate(std::string OpName) override {
      if (fIsOutputConstant) return "";
      OpName = "op_" + OpName;

     if (fShapeY.empty()) {
         throw std::runtime_error("TMVA SOFIE Comparision Op called to Generate without being initialized first");
      }
      std::stringstream out;
      out << SP << "\n//------ " << ComparisionTrait<T,Op>::Name() << "\n";
      size_t length = ConvertShapeToLength(fShapeY);
      // Broadcast A if it's uninitialized
      if (!fNBroadcastedX1.empty()) {
         std::string type1 = ConvertTypeToString(fTensorType1);
         out << SP << "// Broadcasting uninitialized tensor " << fNX1 << "\n";
         out << SP << "{\n";
         out << SP << SP << type1 << "* data = TMVA::Experimental::SOFIE::UTILITY::UnidirectionalBroadcast<" << type1 << ">(tensor_" << fNX1 << ", " << ConvertShapeToString(fShapeX1) << ", " << ConvertShapeToString(fShapeY) << ");\n";
         out << SP << SP << "std::copy(data, data + " << length << ", tensor_" << fNBroadcastedX1 << ");\n";
         out << SP << SP << "delete[] data;\n";
         out << SP << "}\n";
      }
      // Broadcast B if it's uninitialized
      if (!fNBroadcastedX2.empty()) {
         std::string type2 = ConvertTypeToString(fTensorType2);
         out << SP << "// Broadcasting uninitialized tensor " << fNX2 << "\n";
         out << SP << "{\n";
         out << SP << SP << type2 << "* data = TMVA::Experimental::SOFIE::UTILITY::UnidirectionalBroadcast<" << type2 << ">(tensor_" << fNX2 << ", " << ConvertShapeToString(fShapeX2) << ", " << ConvertShapeToString(fShapeY) << ");\n";
         out << SP << SP << "std::copy(data, data + " << length << ", tensor_" << fNBroadcastedX2 << ");\n";
         out << SP << SP << "delete[] data;\n";
         out << SP << "}\n";
      }
      const std::string& nameX1 = fNBroadcastedX1.empty()? fNX1 : fNBroadcastedX1;
      const std::string& nameX2 = fNBroadcastedX2.empty()? fNX2 : fNBroadcastedX2;

      out << SP << "for (size_t id = 0; id < " << length << " ; id++){\n";
      out << SP << SP << "fTensor_" << fNY << "[id] = " << ComparisionTrait<T,Op>::Op( "tensor_" + nameX1 + "[id]" , "tensor_" + nameX2 + "[id]") <<  " ;\n";
      out << SP << "}\n";
      // since output is a boolean need to add the tensor_xxx variable since it is not defined as a pointer to a boolean std::vector
      if (!fIsModelOutput)
         out << SP << "const std::vector<bool> & tensor_" << fNY << " = fTensor_" << fNY << ";\n";

      return out.str();
   }

};

}//SOFIE
}//Experimental
}//TMVA


#endif //TMVA_SOFIE_ROperator_Comparision
