#ifndef TMVA_SOFIE_ROPERATOR_Cast
#define TMVA_SOFIE_ROPERATOR_Cast

#include "TMVA/SOFIE_common.hxx"
#include "TMVA/ROperator.hxx"
#include "TMVA/RModel.hxx"

#include <sstream>

namespace TMVA{
namespace Experimental{
namespace SOFIE{


template <typename T>
class ROperator_Cast final : public ROperator
{

private:

   std::string fNX;
   std::string fNY;
   std::vector<size_t> fShape;
   std::string fAttrType = "float";

public:
   ROperator_Cast(){}
   ROperator_Cast(std::string attr_type,std::string nameX, std::string nameY):
   fNX(UTILITY::Clean_name(nameX)), fNY(UTILITY::Clean_name(nameY)),
   fAttrType(attr_type) {}

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
        throw std::runtime_error("TMVA SOFIE Cast Op Input Tensor is not found in model");
      }
      fShape = model.GetTensorShape(fNX);
      model.AddIntermediateTensor(fNY, ConvertStringToType(fAttrType), fShape);
   }


   std::string Generate(std::string OpName){
      OpName = "op_" + OpName;
      if (fShape.empty()) {
         throw std::runtime_error("TMVA SOFIE Cast called to Generate without being initialized first");
      }
      std::stringstream out;
      size_t length = ConvertShapeToLength(fShape);

      // out << SP << ETensorType << " " << OpName << "_attr = "  << fattr << ";\n";
      out << "\n//------ CAST\n";
      out << SP << "for (int id = 0; id < " << length << " ; id++){\n";

      out << SP << SP << "tensor_" << fNY << "[id] = static_cast<"<< fAttrType << ">(tensor_" << fNX << "[id]);\n";

      out << SP << "}\n";
      return out.str();
   }

   std::string GenerateGPU(std::string OpName) {
      OpName = "op_" + OpName;
      if (fShape.empty()) {
         throw std::runtime_error("TMVA SOFIE Cast called to Generate without being initialized first");
      }
      std::stringstream out;
      size_t length = ConvertShapeToLength(fShape);

      out << "\n" << SP*3 << "//------ CAST\n";
      out << SP*3 << "q.submit([&](cl::sycl::handler& cgh){\n";
      out << SP*4 << "auto acc_tensor_" << fNX << " = cl::sycl::accessor{buf_tensor_" << fNX;
      out << ", cgh, cl::sycl::read_only};\n";
      out << SP*4 << "auto acc_tensor_" << fNY << "= cl::sycl::accessor{buf_tensor_" << fNY;
      out << ", cgh, cl::sycl::write_only};\n";
      out << SP*4 << "cgh.parallel_for<class " << OpName << ">(cl::sycl::range<1>(" << length;
      out << "), [=](cl::sycl::id<1> id){\n";
      out << SP*5 << "acc_tensor_" << fNY << "[id] = static_cast<" << fAttrType << ">(acc_tensor_";
      out << fNX << "[id]);\n";
      out << SP*4 << "});\n";
      out << SP*3 << "});\n";
      
      return out.str();
   }

};

}//SOFIE
}//Experimental
}//TMVA


#endif //TMVA_SOFIE_ROPERATOR_Cast
