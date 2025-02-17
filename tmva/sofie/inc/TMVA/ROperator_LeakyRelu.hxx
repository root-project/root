#ifndef TMVA_SOFIE_ROPERATOR_LeakyRelu
#define TMVA_SOFIE_ROPERATOR_LeakyRelu

#include "TMVA/SOFIE_common.hxx"
#include "TMVA/ROperator.hxx"
#include "TMVA/RModel.hxx"

#include <sstream>

namespace TMVA{
namespace Experimental{
namespace SOFIE{

template <typename T>
class ROperator_LeakyRelu final : public ROperator
{

private:

   /* Attributes*/
   float falpha=0.01; //default value
   std::string fNX;
   std::string fNY;
   std::vector<size_t> fShape;
   std::string fType;

public:
   ROperator_LeakyRelu(){}
   ROperator_LeakyRelu(float alpha,std::string nameX, std::string nameY):
   falpha(alpha),fNX(UTILITY::Clean_name(nameX)), fNY(UTILITY::Clean_name(nameY))
   {
      if(std::is_same<T, float>::value){
         fType = "float";
      }
		else{
			throw
				std::runtime_error("TMVA SOFIE Encountered unsupported type parsing a Leaky Relu operator");
		}
   }

   std::vector<ETensorType> TypeInference(std::vector<ETensorType> input){
      return input;
   }

   std::vector<std::vector<size_t>> ShapeInference(std::vector<std::vector<size_t>> input){
      auto ret = input; //suggest copy to compiler
      return ret;
   }

   void Initialize(RModel& model){
      if (model.CheckIfTensorAlreadyExist(fNX) == false){   //input must be a graph input, or already initialized intermediate tensor
         throw std::runtime_error("TMVA SOFIE Leaky Relu Op Input Tensor is not found in model");
      }
      fShape = model.GetTensorShape(fNX);
      model.AddIntermediateTensor(fNY, model.GetTensorType(fNX), fShape);
   }


   std::string Generate(std::string OpName){
      OpName = "op_" + OpName;
      if (fShape.empty()) {
         throw std::runtime_error("TMVA SOFIE Operator Leaky Relu called to Generate without being initialized first");
      }
      std::stringstream out;
      size_t length = ConvertShapeToLength(fShape);

      out << SP << "constexpr float " << OpName << "_alpha = " << std::setprecision(std::numeric_limits<float>::max_digits10) << falpha << ";\n";

      out << "\n//------ LEAKY RELU\n";
      out << SP << "for (int id = 0; id < " << length << " ; id++){\n";
      out << SP << SP << "tensor_" << fNY << "[id] = ((tensor_" << fNX << "[id] >= 0 )? tensor_" << fNX << "[id] : "<< OpName << "_alpha * tensor_"<< fNX<<"[id]);\n";
      out << SP << "}\n";
      return out.str();
   }

   std::string GenerateGPU(std::string OpName, std::string gemm, std::string copy, 
   std::string axpy, std::string transpose, std::string nontrans, std::string trans, std::string copy_batch, std::string scal) {
      OpName = "op_" + OpName;
      if (fShape.empty()) {
         throw std::runtime_error("TMVA SOFIE Operator Relu called to Generate without being initialized first");
      }
      std::stringstream out;
      int length = 1;
      for(auto& i: fShape){
         length *= i;
      }

      out << "\n" << SP*3 << "//------ LEAKY RELU\n";
      out << SP*3 << "float " << OpName << "_alpha = " << std::setprecision(std::numeric_limits<float>::max_digits10) << falpha << ";\n";
      
      out << SP*3 << "q.submit([&](cl::sycl::handler &cgh){\n";
      out << SP*4 << "auto acc_tensor_" << fNX << " = cl::sycl::accessor{buf_tensor_" << fNX;
      out << ", cgh, cl::sycl::read_only};\n";
      out << SP*4 << "auto acc_tensor_" << fNY << " = cl::sycl::accessor{buf_tensor_" << fNY;
      out << ", cgh, cl::sycl::write_only, cl::sycl::no_init};\n\n";
      out << SP*4 << "cgh.parallel_for<class " << OpName << ">(cl::sycl::range<1>(" << std::to_string(length);
      out << "), [=](cl::sycl::id<1> id){\n";
      out << SP*5 << "acc_tensor_" << fNY << "[id] = cl::sycl::max(acc_tensor_" << fNX << "[id], ";
      out << OpName << "_alpha * acc_tensor_" << fNX << "[id]);\n";
      out << SP*4 << "});\n";
      out << SP*3 << "});\n";

      return out.str();
   }

};

}//SOFIE
}//Experimental
}//TMVA


#endif //TMVA_SOFIE_ROPERATOR_LeakyRelu
