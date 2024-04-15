#ifndef TMVA_SOFIE_ROPERATOR_EyeLike
#define TMVA_SOFIE_ROPERATOR_EyeLike

#include "TMVA/SOFIE_common.hxx"
#include "TMVA/ROperator.hxx"
#include "TMVA/RModel.hxx"

#include <sstream>

namespace TMVA{
namespace Experimental{
namespace SOFIE{

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
      if (fShape.size() != 2)
         throw std::runtime_error("TMVA SOFIE EyeLike Op Input Tensor is not of rank 2");

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
      auto length = ConvertShapeToLength(fShape);
      auto stride = TMVA::Experimental::SOFIE::UTILITY::ComputeStrideFromShape(fShape);
      std::stringstream out;
      out << SP << "///--------EyeLike operator\n" << std::endl;
      // add a dummy statement to avoid warning for unused input
      out << SP << "(void) tensor_" << fNX << ";\n";

      out << SP << "fTensor_" << fNY << ".assign(" << length << ", 0);\n";
      out << SP << "for (int i = 0; i < " << fShape[0] << "; i++) {\n";
      out << SP << SP << "int j = i +" << fk << ";\n";
      out << SP << SP << "if (j >= 0 && j < " << fShape[1] << ")\n";
      out << SP << SP << SP << "tensor_" << fNY << "[i * " << fShape[1] << "+ j] = 1;\n";
      out << SP << "}\n";
      return out.str();
   }

};

}//SOFIE
}//Experimental
}//TMVA


#endif //TMVA_SOFIE_ROPERATOR_EyeLike