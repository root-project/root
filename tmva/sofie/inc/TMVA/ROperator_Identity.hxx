#ifndef TMVA_SOFIE_ROPERATOR_IDENTITY
#define TMVA_SOFIE_ROPERATOR_IDENTITY

#include "TMVA/SOFIE_common.hxx"
#include "TMVA/ROperator.hxx"
#include "TMVA/RModel.hxx"

#include <sstream>

namespace TMVA{
namespace Experimental{
namespace SOFIE{

template <typename T>
class ROperator_Identity final : public ROperator
{

private:

   std::string fNX;
   std::string fNY;
   std::vector<size_t> fShape;

public:
   ROperator_Identity(){}
   ROperator_Identity(std::string nameX, std::string nameY):
      fNX(UTILITY::Clean_name(nameX)), fNY(UTILITY::Clean_name(nameY)){}

   std::vector<ETensorType> TypeInference(std::vector<ETensorType> input){
      return input;
   }

   std::vector<std::vector<size_t>> ShapeInference(std::vector<std::vector<size_t>> input){
      auto ret = input; //suggest copy to compiler
      return ret;
   }

   void Initialize(RModel& model){
       //input must be a graph input, or already initialized intermediate tensor
      if (model.CheckIfTensorAlreadyExist(fNX) == false){
        throw std::runtime_error("TMVA SOFIE Identity Op Input Tensor is not found in model");
      }
      fShape = model.GetTensorShape(fNX);
      model.AddIntermediateTensor(fNY, model.GetTensorType(fNX), fShape);
   }


   std::string Generate(std::string OpName){
      OpName = "op_" + OpName;
      if (fShape.empty()) {
         throw std::runtime_error("TMVA SOFIE Operator Identity called to Generate without being initialized first");
      }
      std::stringstream out;
      out << "\n//------ IDENTITY\n";
      // just copy the tensor pointers
      out << SP << SP << "tensor_" << fNY << " = tensor_" << fNX << ";\n";
      return out.str();
   }

   std::string GenerateGPU(std::string OpName) {
      OpName = "op_" + OpName;
      if (fShape.empty()) {
         throw std::runtime_error("TMVA SOFIE Operator Identity called to Generate without being initialized first");
      }
      std::stringstream out;
      out << "\n" << SP*3 << "//------ IDENTITY\n";
      out << SP*3 << "q.submit([&](cl::sycl::handler &cgh){\n";
      out << SP*4 << "auto acc_tensor_" << fNX << " = cl::sycl::accessor{buf_tensor_" << fNX;
      out << ", cgh, cl::sycl::read_only};\n";
      out << SP*4 << "auto acc_tensor_" << fNY << " = cl::sycl::accessor{buf_tensor_" << fNY;
      out << ", cgh, cl::sycl::write_only, cl::sycl::no_init};\n\n";
      out << SP*4 << "cgh.copy_in_device(acc_tensor_" << fNX << ", acc_tensor_" << fNY << ");\n";
      out << SP*4 << "});\n";
      out << SP*3 << "});\n";
   
      return out.str();
   }

};

}//SOFIE
}//Experimental
}//TMVA


#endif //TMVA_SOFIE_ROPERATOR_IDENTITY
