
// #ifndef TMVA_SOFIE_ROPERATOR_Reduce
// #define TMVA_SOFIE_ROPERATOR_Reduce

// #include "TMVA/SOFIE_common.hxx"
// #include "TMVA/ROperator.hxx"
// #include "TMVA/RModel.hxx"

// #include <memory>
// #include <sstream>
// #include <algorithm>
// #include <stdexcept>
// #include <vector>
// #include <cassert>

// namespace TMVA{
// namespace Experimental{
// namespace SOFIE{

// enum ReduceOpMode { ReduceMean, ReduceSumsquare, ReduceProd };

// template <typename T, ReduceOpMode Op1>
// struct ReduceOperatorTrait {
//    const char *Name() { return ""; }
// };
// template <typename T>
// struct ReduceOperatorTrait <T, ReduceMean> {
//    static const char *Name() { return "ReduceMean"; }
// };

// template <typename T>
// struct ReduceOperatorTrait <T, ReduceProd> {
//    static const char *Name() { return "ReduceProd"; }
// };

// template <typename T>
// struct ReduceOperatorTrait <T, ReduceSumsquare> {
//    static const char *Name() { return "ReduceSumsquare"; }
// };

// template <typename T, ReduceOpMode Op>
// class ROperator_Reduce final : public ROperator
// {
// private:
//     /* Attributes*/
//     int fAxis = 1;
//     ReduceOpMode fReduceMode;
//     int fkeepdims = 1; //default value
//     std::string fNX;
//     std::string fNY;
//     std::vector<size_t> fShapeX;
//     std::vector<size_t> fShapeY;

// public:

//    ROperator_Reduce(){}   
//    ROperator_Reduce(int keepdims,int axis,std::string nameX, std::string nameY):
//    fkeepdims(keepdims), fAxis(axis), fNX(UTILITY::Clean_name(nameX)), fNY(UTILITY::Clean_name(nameY)) {}

//    // type of output given input
//    std::vector<ETensorType> TypeInference(std::vector<ETensorType> input){
//       return input;
//    }

//    // shape of output tensors given input tensors
//    std::vector<std::vector<size_t>> ShapeInference(std::vector<std::vector<size_t>> input){
//       // assume now inputs have same shape (no broadcasting)
//       auto ret = std::vector<std::vector<size_t>>(1, input[0]); // return vector size 1 with first input
//       return ret;
//    }
//     void Initialize(RModel& model){

//         fUseSession = model.UseSession();

//         if (model.CheckIfTensorAlreadyExist(fNX) == false){   //input must be a graph input, or already initialized intermediate tensor
//             throw std::runtime_error("TMVA SOFIE Reduce Op Input Tensor " + fNX + " is not found in model");
//         }
//         fShapeX = model.GetTensorShape(fNX);
//         // find shape of Y and add it in the list of intermediate tensors
//         fShapeY = ShapeInference({fShapeX})[0];
//         model.AddIntermediateTensor(fNY, model.GetTensorType(fNX), fShapeY);
//     }

//     std::string Generate(std::string OpName){
//       OpName = "op_" + OpName;
//       if (fShapeX.empty() || fShapeY.empty()) {
//          throw std::runtime_error("TMVA SOFIE Reduce Op called to Generate without being initialized first");
//       }

//       size_t outputLength = TMVA::Experimental::SOFIE::ConvertShapeToLength(fShapeY);

//       auto inputStrides = TMVA::Experimental::SOFIE::UTILITY::ComputeStrideFromShape(fShapeX);
//       auto outputStrides = TMVA::Experimental::SOFIE::UTILITY::ComputeStrideFromShape(fShapeY);

//       size_t dim = fShapeY.size();
//       std::vector<size_t> idx(dim);

//    std::stringstream out;
//    for (size_t i = 0; i < outputLength; i++) {
      
//       if (dim == 2) {
//          idx[0] = i / outputStrides[0];
//          idx[1] = i % outputStrides[0];
//       }
//       if (dim == 3) {
//          idx[0] = i / outputStrides[0];
//          idx[1] = (i % outputStrides[0]) / outputStrides[1];
//          idx[2] = (i % outputStrides[0]) % outputStrides[1];
//       }
//       if (dim == 4) {
//          idx[0] = i / outputStrides[0];
//          idx[1] = (i % outputStrides[0]) / outputStrides[1];
//          idx[2] = ((i % outputStrides[0]) % outputStrides[1]) / outputStrides[2]; 
//          idx[3] = ((i % outputStrides[0]) % outputStrides[1]) % outputStrides[2];
//       }

//       assert(idx[fAxis] == 0);  // we can avoid computing this for the reduction axis which by definition is always zero 
      
//       out << SP << "float sum = 0;\n";
//       // float sum = 0;
//       for (size_t k = 0; k < fShapeX[fAxis]; k++) {
//          idx[fAxis] = k;
//          // compute input index j 
//          size_t j = 0;
//          if (dim == 2) j = idx[0]*inputStrides[0] + idx[1];
//          if (dim == 3) j = idx[0]*inputStrides[0] + idx[1]* inputStrides[1] + idx[2];
//          if (dim == 4) j = idx[0]*inputStrides[0] + idx[1]* inputStrides[1] + idx[2]*inputStrides[2] + idx[3];

//          out << SP << SP << "sum +=  tensor_" << fNX[j] << ";\n";
//       }
//       out << SP << "float average = sum/float(" << fShapeX[fAxis] << ")\n;";
//       out << SP << "tensor_" << fNY[i] << " = average;\n"; 
//    }
//       return out.str();
//    }

// };

// }//SOFIE
// }//Experimental
// }//TMVA


// #endif //TMVA_SOFIE_ROPERATOR_Reduce

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

template <typename T, ReduceOpMode Op1>
struct ReduceOperatorTrait {
   const char *Name() { return ""; }
};
template <typename T>
struct ReduceOperatorTrait <T, ReduceMean> {
   static const char *Name() { return "ReduceMean"; }
};

template <typename T>
struct ReduceOperatorTrait <T, ReduceProd> {
   static const char *Name() { return "ReduceProd"; }
};

template <typename T>
struct ReduceOperatorTrait <T, ReduceSumsquare> {
   static const char *Name() { return "ReduceSumsquare"; }
};

template <typename T, ReduceOpMode Op>
class ROperator_Reduce final : public ROperator
{
private:
    /* Attributes*/
    int fkeepdims = 1; //default value
    std::string fNX;
    std::string fNY;
    std::vector<size_t> fShapeX;
    std::vector<size_t> fShapeY;
    int fAttrAxes;

public:
   ROperator_Reduce(){}   
   ROperator_Reduce(int keepdims,int attrAxes,std::string nameX, std::string nameY):
   fkeepdims(keepdims), fAttrAxes(attrAxes), fNX(UTILITY::Clean_name(nameX)), fNY(UTILITY::Clean_name(nameY)) {}

   // type of output given input
   std::vector<ETensorType> TypeInference(std::vector<ETensorType> input){
      return input;
   }

   // shape of output tensors given input tensors
   std::vector<std::vector<size_t>> ShapeInference(std::vector<std::vector<size_t>> input){
      
      // std::vector<std::vector<size_t>> ret;
      // auto & input_shape = input[0];
      // auto ret = std::vector<std::vector<size_t>>(1, input[0]); // return vector size 1 with first input
      // return ret;
      auto ret = input; //suggest copy to compiler
      ret[fAttrAxes] = 1;
      return ret;
   }
    void Initialize(RModel& model){

        fUseSession = model.UseSession();

        if (model.CheckIfTensorAlreadyExist(fNX) == false){   //input must be a graph input, or already initialized intermediate tensor
            throw std::runtime_error("TMVA SOFIE Reduce Op Input Tensor " + fNX + " is not found in model");
        }
        fShapeX = model.GetTensorShape(fNX);
        // find shape of Y and add it in the list of intermediate tensors
        fShapeY = ShapeInference(fShapeX);
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
      out << "\n//----  operator " << std::string(ReduceOperatorTrait<T,Op>::Name()) << "  " << OpName << "\n";
      out << SP << "size_t dim = " << fShapeY.size() << ";\n";

      out << SP << "std::vector<size_t> idx(dim);";

      
      out << SP << "for (size_t i = 0; i < " << outputLength << "; i++) {\n";
      
      // write here according to size of shape
      // in generation code can be done automatically
      // i0 =  i / s0 ; i1 = (i % s0) / s1 ; i2 = ( (i % s0) % s1 ) / s2 and so on
      // and we have for the inverse
      // i = i0 * s0 + i1 * s1 + i2 * s2 + i3 * s3 ....

      // don't need to divide by last stride s[n-1] since it is 1 by definition
      
      out << SP << SP << "idx[j] = i;\n";
      out << SP << SP << "size_t k = 0;\n";
      out << SP << SP << SP << "for(k=0; k < dim-1; k++){\n";
      out << SP << SP << SP << "idx[k] = idx[k] %" << outputStrides << "[k];\n";
      out << SP << SP << SP << "};\n"; 
      out << SP << SP << "idx[j] = idx[j] /" << outputStrides << "[k];\n";
      

      out << SP << "assert(idx[" << fAttrAxes << "] == 0);\n";  // we can avoid computing this for the reduction axis which by definition is always zero

      out << SP << "float sum = 0;\n";
      out << SP << SP << " for (size_t k = 0; k < inputShape[" << fAttrAxes << "]; k++) { \n";
      out << SP << SP << "  idx[" << fAttrAxes << "] = k;\n";
       // compute input index j 
      out << SP << SP << "size_t l = 0;\n";
      out << SP << SP << "size_t m = 0;\n";
      out << SP << SP << SP << "for(m=0; m < dim-1; m++){\n";
      out << SP << SP << SP << "l += idx[m] *" << inputStrides << "[m];\n";
      out << SP << SP << SP << "};\n"; 
      out << SP << SP << "l += idx[m];\n";
      out << SP << SP << "sum += tensor_" << fNX << "[l];\n";
      out << SP << SP << "};\n"; 
      out << SP << SP << "float average = sum/float(inputShape[" << fAttrAxes << "]);\n";
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