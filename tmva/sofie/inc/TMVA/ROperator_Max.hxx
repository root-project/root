#ifndef TMVA_SOFIE_ROPERATOR_MAX
#define TMVA_SOFIE_ROPERATOR_MAX

#include "TMVA/SOFIE_common.hxx"
#include "TMVA/ROperator.hxx"
#include "TMVA/RModel.hxx"

#include <sstream>
#include <algorithm>

namespace TMVA{
namespace Experimental{
namespace SOFIE{

template <typename T>
class ROperator_Max final : public ROperator
{

private:

   std::vector<std::string> fInputNames;
   std::string fNY;
   std::vector<std::vector<size_t> > fShapeInputs;
   std::vector<size_t> fShape;

public:
   ROperator_Max(){}
   ROperator_Max( const std::vector<std::string> & inputNames, std::string nameY):
   fNY(UTILITY::Clean_name(nameY)){
      fInputNames.reserve(inputNames.size());
      for (auto & name : inputNames)
         fInputNames.push_back(UTILITY::Clean_name(name));
   }

   // type of output given input 
   std::vector<ETensorType> TypeInference(std::vector<ETensorType> input){
      return input;
   }

   // shape of output tensors given input tensors
   std::vector<std::vector<size_t>> ShapeInference(std::vector<std::vector<size_t>> input){
      auto ret = std::vector<std::vector<size_t>>(1, input[0]); 
      return ret;
   }

   void Initialize(RModel& model){
      // input must be a graph input, or already initialized intermediate tensor

      for (auto &it : fInputNames) {
         if (model.CheckIfTensorAlreadyExist(it) == false) {
            throw std::runtime_error("TMVA SOFIE Max Op Input Tensor " + it + " is not found in model");
         }
         fShapeInputs.push_back(model.GetTensorShape(it));
      }
      for(size_t i=0; i < fShapeInputs.size()-1; i++){
         // input must be a graph input, or already initialized intermediate tensor
         auto shapeX1 = fShapeInputs[i];
         auto shapeX2 = fShapeInputs[i+1];
         // If the shape of 2 tensors are not same we perform multi-directional Broadcasting.
         // We only support tensors with same length and the resultant output length should also be same.
         if (shapeX1 != shapeX2) {
            fShape = UTILITY::Multidirectional_broadcast(shapeX1,shapeX2);
            size_t length1 = ConvertShapeToLength(shapeX1);
            size_t length2 = ConvertShapeToLength(shapeX2);
            size_t output_length = ConvertShapeToLength(fShape);
            if(length1 != length2 || length1 != output_length){
               throw std::runtime_error(std::string("TMVA SOFIE Max Op does not support input tensors with different lengths. The output tensor should also have the same length as the input tensors."));
            }
         }
         // If both the tensors have same shape then assign the same shape to resultant output.
         else if(shapeX1 == shapeX2){
            fShape = shapeX1;
         }
      }
      model.AddIntermediateTensor(fNY, model.GetTensorType(fInputNames[0]), fShape);
      model.AddNeededStdLib("cmath");
   }


   std::string Generate(std::string OpName){
      OpName = "op_" + OpName;
      if (fShape.empty()) {
         throw std::runtime_error("TMVA SOFIE Max called to Generate without being initialized first");
      }
      std::stringstream out;
      size_t length = ConvertShapeToLength(fShape);
      out << "\n//------ Max\n";
      out << SP << "for (size_t id = 0; id < " << length << " ; id++){\n";
      if(fInputNames.size() == 1){
         out << SP << SP <<"tensor_" << fNY << "[id] = tensor_" << fInputNames[0] << "[id];\n";
      }
      else if(fInputNames.size() == 2){
         out << SP << SP <<"tensor_" << fNY << "[id] = std::max({tensor_" << fInputNames[0] << "[id],tensor_" << fInputNames[1] << "[id]});\n";
      }
      else{
         out << SP << SP <<"tensor_" << fNY << "[id] = std::max({";
         size_t j = 0;
         for ( j = 0; j < fInputNames.size()-1; j++){
            out << "tensor_" << fInputNames[j] << "[id],";
         }
         out << "tensor_" << fInputNames[j] << "[id]});\n";
      }
      out << SP << "}\n";
      return out.str();
   }

};

}//SOFIE
}//Experimental
}//TMVA


#endif //TMVA_SOFIE_ROPERATOR_Max