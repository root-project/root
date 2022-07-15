#ifndef TMVA_SOFIE_ROPERATOR_Pow
#define TMVA_SOFIE_ROPERATOR_Pow

#include "TMVA/SOFIE_common.hxx"
#include "TMVA/ROperator.hxx"
#include "TMVA/RModel.hxx"

#include <sstream>

namespace TMVA{
namespace Experimental{
namespace SOFIE{

template <typename T>
class ROperator_Pow final : public ROperator
{

private:

   std::string fNX1;
   std::string fNX2;
   std::string fNY;
   std::vector<size_t> fShape;

public:
public:
   ROperator_Pow(){}
   ROperator_Pow(std::string nameX1, std::string nameX2, std::string nameY):
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
      // If the shape of 2 tensors are not same we perform multi-directional Broadcasting.
      // We only support tensors with same length and the resultant output length should also be same.
      if (shapeX1 != shapeX2) {
         fShape = UTILITY::Multidirectional_broadcast(shapeX1,shapeX2);
         size_t length1 = ConvertShapeToLength(shapeX1);
         size_t length2 = ConvertShapeToLength(shapeX2);
         size_t output_length = ConvertShapeToLength(fShape);
         if(length1 != length2 || length1 != output_length){
            throw std::runtime_error(std::string("TMVA SOFIE Binary Op does not support input tensors with different lengths. The output tensor should also have the same length as the input tensors."));
         }
      }
      // If both the tensors have same shape then assign the same shape to resultant output.
      else if(shapeX1 == shapeX2){
         fShape = shapeX1;
      }   
      model.AddIntermediateTensor(fNY, model.GetTensorType(fNX1), fShape);
   }


   std::string Generate(std::string OpName){
      OpName = "op_" + OpName;
      if (fShape.empty()) {
         throw std::runtime_error("TMVA SOFIE Pow Op called to Generate without being initialized first");
      }
      std::stringstream out;
      size_t length = ConvertShapeToLength(fShape);
      out << "\n//------ POW\n";
      out << SP << "for (int id = 0; id < " << length << " ; id++){\n";
      out << SP << SP << "tensor_" << fNY << "[id] = std::pow(tensor_" << fNX1 << "[id] , tensor_" << fNX2 << "[id]);\n";
      out << SP << "}\n";
      return out.str();
   }

};

}//SOFIE
}//Experimental
}//TMVA


#endif //TMVA_SOFIE_ROPERATOR_Pow

