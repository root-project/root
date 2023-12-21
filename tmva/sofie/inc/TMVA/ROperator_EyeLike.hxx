#ifndef TMVA_SOFIE_ROPERATOR_EyeLike
#define TMVA_SOFIE_ROPERATOR_EyeLike

#include "TMVA/SOFIE_common.hxx"
#include "TMVA/ROperator.hxx"
#include "TMVA/RModel.hxx"

#include <sstream>

namespace TMVA{
namespace Experimental{
namespace SOFIE{

template <typename T>
class ROperator_EyeLike final : public ROperator
{

private:

   int fdtype = static_cast<int>(ETensorType::FLOAT); //Default value
   int fk = 0;
   std::string fNX;
   std::string fNY;
   std::vector<size_t> fShape;

public:
   ROperator_EyeLike(){}
   ROperator_EyeLike(int dtype, int k, std::string nameX, std::string nameY):
      fdtype(dtype), fk(k), fNX(UTILITY::Clean_name(nameX)), fNY(UTILITY::Clean_name(nameY)){}

   std::vector<ETensorType> TypeInference(std::vector<ETensorType> input){
      return input;
   }

   std::vector<std::vector<size_t>> ShapeInference(std::vector<std::vector<size_t>> input){
      auto ret = input; //suggest copy to compiler
      return ret;
   }

   void Initialize(RModel& model){
      if (model.CheckIfTensorAlreadyExist(fNX) == false){   //input must be a graph input, or already initialized intermediate tensor
         throw std::runtime_error("TMVA SOFIE EyeLike Op Input Tensor is not found in model");
      }
      fShape = model.GetTensorShape(fNX);
      if(fdtype){
        ETensorType extractedType = static_cast<ETensorType>(fdtype);
        model.AddIntermediateTensor(fNY, extractedType, fShape);
      }
      else{
        model.AddIntermediateTensor(fNY, model.GetTensorType(fNX), fShape);
      }
   }


   std::string Generate(std::string OpName){
      OpName = "op_" + OpName;
      if (fShape.empty()){
         throw std::runtime_error("TMVA SOFIE Operator EyeLike called to Generate without being initialized first");
      }
      std::stringstream out;
      int length = 1;
      for(auto& i: fShape){
         length *= i;
      }
         out << SP << "///--------EyeLike operator\n" << std::endl;
         out << SP << "for (int i = 0; i < " << ConvertShapeToString(fShape) << "[0]; i++) {" << std::endl;
         out << SP << SP << "for (int j = 0; j < " << ConvertShapeToString(fShape) << "[1]; j++) {" << std::endl;
         out << SP << SP << SP << fNY << "[i * " << ConvertShapeToString(fShape) << "[1] + j] = (i + " << fk << " == j) ? 1.0 : 0.0;" << std::endl;
         out << SP << SP << "}" << std::endl;
         out << SP << "}" << std::endl;
         return out.str();
   }

};

}//SOFIE
}//Experimental
}//TMVA


#endif //TMVA_SOFIE_ROPERATOR_EyeLike