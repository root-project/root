#ifndef TMVA_SOFIE_ROPERATOR_Reduce
#define TMVA_SOFIE_ROPERATOR_Reduce

#include "TMVA/SOFIE_common.hxx"
#include "TMVA/ROperator.hxx"
#include "TMVA/RModel.hxx"

#include <memory>
#include <sstream>
#include <algorithm>
#include <stdexcept>
#include <vector>
#include <cassert>

namespace TMVA{
namespace Experimental{
namespace SOFIE{

enum EReduceOpMode { ReduceMean, ReduceSumsquare, ReduceProd, InvalidReduceOp };

template <typename T, EReduceOpMode Op>
class ROperator_Reduce final : public ROperator
{
private:
    /* Attributes*/
    int fkeepdims = 1; //default value
    int fAttrAxes;
    EReduceOpMode fReduceOpMode;
    std::string fNX;
    std::string fNY;
    std::vector<size_t> fShapeX;
    std::vector<size_t> fShapeY;
   

public:

   std::string Name() {
      if (fReduceOpMode == ReduceMean)  return "ReduceMean";
      else if (fReduceOpMode == ReduceSumsquare )  return "ReduceSumsquare";
      else if (fReduceOpMode == ReduceProd ) return "ReduceProd";
      return "Invalid";
   }

   ROperator_Reduce(){}   
   ROperator_Reduce(int keepdims,int attrAxes,std::string nameX, std::string nameY):
   fkeepdims(keepdims), fAttrAxes(attrAxes), fNX(UTILITY::Clean_name(nameX)), fNY(UTILITY::Clean_name(nameY)) {}

   // type of output given input
   std::vector<ETensorType> TypeInference(std::vector<ETensorType> input){
      return input;
   }

   // shape of output tensors given input tensors
   std::vector<std::vector<size_t>> ShapeInference(std::vector<std::vector<size_t>> input){
      auto ret = input; //suggest copy to compiler
      ret[0][fAttrAxes] = 1;
      return ret;
   }
    void Initialize(RModel& model){

        fUseSession = model.UseSession();

        if (model.CheckIfTensorAlreadyExist(fNX) == false){   //input must be a graph input, or already initialized intermediate tensor
            throw std::runtime_error("TMVA SOFIE Reduce Op Input Tensor " + fNX + " is not found in model");
        }
        fShapeX = model.GetTensorShape(fNX);
         // find shape of Y and add it in the list of intermediate tensors
         fShapeY = ShapeInference({fShapeX})[0];
         model.AddIntermediateTensor(fNY, model.GetTensorType(fNX), fShapeY);

    }

    std::string Generate(std::string OpName){
      OpName = "op_" + OpName;
      if (fShapeX.empty() || fShapeY.empty()) {
         throw std::runtime_error("TMVA SOFIE Reduce Op called to Generate without being initialized first");
      }

      size_t outputLength = TMVA::Experimental::SOFIE::ConvertShapeToLength(fShapeY);

      auto inputStrides = TMVA::Experimental::SOFIE::UTILITY::ComputeStrideFromShape(fShapeX);
      auto outputStrides = TMVA::Experimental::SOFIE::UTILITY::ComputeStrideFromShape(fShapeY);
      
      // write here according to size of shape
      // in generation code can be done automatically
      // i0 =  i / s0 ; i1 = (i % s0) / s1 ; i2 = ( (i % s0) % s1 ) / s2 and so on
      // and we have for the inverse
      // i = i0 * s0 + i1 * s1 + i2 * s2 + i3 * s3 ....

      // don't need to divide by last stride s[n-1] since it is 1 by definition
      
      std::stringstream out;
      out << "\n//----  operator " << Name() << "  " << OpName << "\n";
      out << SP << "for (size_t i = 0; i < " << outputLength << "; i++) {\n";
      
      // write here according to size of shape
      // in generation code can be done automatically
      // i0 =  i / s0 ; i1 = (i % s0) / s1 ; i2 = ( (i % s0) % s1 ) / s2 and so on
      // and we have for the inverse
      // i = i0 * s0 + i1 * s1 + i2 * s2 + i3 * s3 ....

      // don't need to divide by last stride s[n-1] since it is 1 by definition

      size_t dim = fShapeX.size();   // this is the input dimension (e.g. 2, 3 or 4 or more)
      out << SP << "std::vector<size_t> outputStrides = {" ;
      for (size_t k = 0; k < dim; k++) {
      out << outputStrides[k] ; 
      if (k < dim-1) 
         out << "  ,";
      else 
         out << " };\n";
      }
      // no compute indices as function of strides
      // as in the example I have sent you
      
      for (size_t k = 0; k < dim; k++) {
         size_t j;
         out << SP << "size_t idx_" << k <<" = i;\n";
         for(j = 0; j < k; j++ )
         out << SP << "idx_" << k << " = idx_" << k <<" % outputStrides[" << j << "];\n" ;

         out << SP << "idx_" << k << " = idx_" << k << "/ outputStrides[" << j << "];\n";
      }

      // out << SP << "assert(idx[" << fAttrAxes << "] == 0);\n";  // we can avoid computing this for the reduction axis which by definition is always zero

      out << SP << "float sum = 0;\n";
      out << SP << SP << " for (size_t k = 0; k < " << fShapeX[fAttrAxes] <<"; k++) { \n";
      out << SP << SP << "  idx_" << fAttrAxes << " = k;\n";
       // compute input index j 
      out << SP << "std::vector<size_t> inputStrides = {" ;
      for (size_t k = 0; k < dim; k++) {
      out << inputStrides[k] ; 
      if (k < dim-1) 
         out << "  ,";
      else 
         out << " };\n";
      }
      out << SP << SP << "size_t l = 0;\n";
      for (size_t m = 0; m < dim; m++) {
         size_t n;
         for(n = 0; n < m; n++ )
         out << SP << "l += idx_" << n << " * inputStrides[" << n << "];\n";

         out << SP << "l +=  idx_" << m << ";\n";
      }
      if(fReduceOpMode == ReduceMean){
         out << SP << SP << "sum += tensor_" << fNX << "[l];\n";
         out << SP << SP << "};\n"; 
      }
      else if(fReduceOpMode == ReduceSumsquare){
         out << SP << SP << "sum += tensor_" << fNX << "[l] * tensor_" << fNX << "[l];\n";
         out << SP << SP << "};\n"; 
      }
      else if(fReduceOpMode == ReduceProd){
         out << SP << SP << "sum *= tensor_" << fNX << "[l];\n";
         out << SP << SP << "};\n"; 
      }
      out << SP << SP << "float average = sum/(float)" << fShapeX[fAttrAxes] << ";\n";
      out << SP << SP << "tensor_" << fNY << "[i] = average;\n";
      out << SP << "};\n";   
      out << SP << "}\n";
      return out.str();
   }

};

}//SOFIE
}//Experimental
}//TMVA


#endif //TMVA_SOFIE_ROPERATOR_Reduce

