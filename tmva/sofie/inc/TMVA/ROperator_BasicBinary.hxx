#ifndef TMVA_SOFIE_ROperator_BasicBinary
#define TMVA_SOFIE_ROperator_BasicBinary

#include "TMVA/SOFIE_common.hxx"
#include "TMVA/ROperator.hxx"
#include "TMVA/RModel.hxx"

#include <sstream>

namespace TMVA{
namespace Experimental{
namespace SOFIE{

enum EBasicBinaryOperator { Add, Sub, Mul, Div };

template <typename T, EBasicBinaryOperator Op1>
struct BinaryOperatorTrait {
   const char *Name() { return ""; }
   const char *Op() { return ""; }
};
template <typename T>
struct BinaryOperatorTrait<T, Add> {
   static const char *Name() { return "Add"; }
   static const char *Op() { return "+"; }
};

template <typename T>
struct BinaryOperatorTrait<T, Sub> {
   static const char *Name() { return "Sub"; }
   static const char *Op() { return "-"; }
};

template <typename T>
struct BinaryOperatorTrait<T, Mul> {
   static const char *Name() { return "Mul"; }
   static const char *Op() { return "*"; }
};

template <typename T>
struct BinaryOperatorTrait<T, Div> {
   static const char *Name() { return "Div"; }
   static const char *Op() { return "/"; }
};

template<typename T, EBasicBinaryOperator Op>
class ROperator_BasicBinary final : public ROperator{
private:

   std::string fNX1;
   std::string fNX2;
   std::string fNY;
   std::vector<size_t> fShape;

   // template <typename T, EBasicBinaryOperator Op1>
   // BinaryOperatorTrait<T,Op1> *s;

public:
   ROperator_BasicBinary(){}
   ROperator_BasicBinary(std::string nameX1, std::string nameX2, std::string nameY):
      fNX1(UTILITY::Clean_name(nameX1)), fNX2(UTILITY::Clean_name(nameX2)), fNY(UTILITY::Clean_name(nameY)){}

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

   void Initialize(RModel& model){
      // input must be a graph input, or already initialized intermediate tensor
      if (model.CheckIfTensorAlreadyExist(fNX1) == false){
         throw std::runtime_error(std::string("TMVA SOFIE Binary Op Input Tensor ") + fNX1 + "is not found in model");
      }
      if (model.CheckIfTensorAlreadyExist(fNX2) == false) {
         throw std::runtime_error(std::string("TMVA SOFIE Binary Op Input Tensor ") + fNX2 + "is not found in model");
      }
      auto shapeX1 = model.GetTensorShape(fNX1);
      auto shapeX2 = model.GetTensorShape(fNX2);
      // assume same shape X1 and X2
      if (shapeX1 != shapeX2) {
         fShape = UTILITY::Multidirectional_broadcast(shapeX1,shapeX2);
         size_t length1 = ConvertShapeToLength(shapeX1);
         size_t length2 = ConvertShapeToLength(shapeX2);
         size_t output_length = ConvertShapeToLength(fShape);
         if(length1 != length2 || length1 != output_length){
            throw std::runtime_error(std::string("TMVA SOFIE Binary Op does not support input tensors with different lengths. The output tensor should also have the same length as the input tensors."));
         }
      }
      else if(shapeX1 == shapeX2){
         fShape = shapeX1;
      }   
      model.AddIntermediateTensor(fNY, model.GetTensorType(fNX1), fShape);
   }


   std::string Generate(std::string OpName){
      OpName = "op_" + OpName;

      if (fShape.empty()) {
         throw std::runtime_error("TMVA SOFIE Binary Op called to Generate without being initialized first");
      }
      std::stringstream out;
      // int length = 1;
      // for(auto& i: fShape){
      //    length *= i;
      // }
      size_t length = ConvertShapeToLength(fShape);
      out << "\n//------ " + std::string(BinaryOperatorTrait<T,Op>::Name())+"\n";
      out << SP << "for (size_t id = 0; id < " << length << " ; id++){\n";
      out << SP << SP << "tensor_" << fNY << "[id] = tensor_" << fNX1 << "[id]" +
      std::string(BinaryOperatorTrait<T,Op>::Op()) +  "tensor_" << fNX2 << "[id];\n";
      out << SP << "}\n";
      return out.str();
   }

};

}//SOFIE
}//Experimental
}//TMVA


#endif //TMVA_SOFIE_ROperator_BasicBinary