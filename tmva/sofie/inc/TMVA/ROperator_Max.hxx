#ifndef TMVA_SOFIE_ROPERATOR_MAX
#define TMVA_SOFIE_ROPERATOR_MAX

#include "TMVA/SOFIE_common.hxx"
#include "TMVA/ROperator.hxx"
#include "TMVA/RModel.hxx"

#include <sstream>

namespace TMVA{
namespace Experimental{
namespace SOFIE{

template <typename T>
class ROperator_Max final : public ROperator
{

private:

   std::string fNX1;
   std::string fNX2;
   std::string fNY;
   std::vector<size_t> fShape;

public:
   ROperator_Max(){}
   ROperator_Max(std::string nameX1, std::string nameX2, std::string nameY):
      fNX1(UTILITY::Clean_name(nameX1)), fNX2(UTILITY::Clean_name(nameX2)), fNY(UTILITY::Clean_name(nameY)){}

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
      if (model.CheckIfTensorAlreadyExist(fNX1) == false){
         throw std::runtime_error(std::string("TMVA SOFIE Max Op Input Tensor ") + fNX1 + "is not found in model");
      }
      if (model.CheckIfTensorAlreadyExist(fNX2) == false) {
         throw std::runtime_error(std::string("TMVA SOFIE Max Op Input Tensor ") + fNX2 + "is not found in model");
      }
      auto shapeX1 = model.GetTensorShape(fNX1);
      auto shapeX2 = model.GetTensorShape(fNX2);
      // assume same shape X1 and X2 
      if (shapeX1 != shapeX2) {
         std::string msg = "TMVA SOFIE Max Op: Support only inputs with same shape, shape 1 is " +
                           ConvertShapeToString(shapeX1) + "shape 2 is " + ConvertShapeToString(shapeX2);
         throw std::runtime_error(msg);
      }
      fShape = shapeX1;
      model.AddIntermediateTensor(fNY, model.GetTensorType(fNX1), fShape);
   }


   std::string Generate(std::string OpName){
      OpName = "op_" + OpName;
      if (fShape.empty()) {
         throw std::runtime_error("TMVA SOFIE Max called to Generate without being initialized first");
      }
      std::stringstream out;
      // int length = 1;
      // for(auto& i: fShape){
      //    length *= i;
      // }
      size_t length = ConvertShapeToLength(fShape);
      out << "\n//------ Max\n";
      out << SP << "for (size_t id = 0; id < " << length << " ; id++){\n";
      out << SP << SP << "tensor_" << fNY << "[id] = (tensor_" << fNX1 << "[id] > tensor_" << fNX2 << "[id]) ? (tensor_" << fNX1 << "[id]) : (tensor_" << fNX2 << "[id]);\n";
      out << SP << "}\n";
      return out.str();
   }

};

}//SOFIE
}//Experimental
}//TMVA


#endif //TMVA_SOFIE_ROPERATOR_Max