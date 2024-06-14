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

enum EReduceOpMode { ReduceMean, ReduceSum, ReduceSumsquare, ReduceProd, InvalidReduceOp };

template <typename T, EReduceOpMode Op>
class ROperator_Reduce final : public ROperator
{
private:
    /* Attributes*/
    int fkeepdims = 1; //default value
    std::vector<int64_t> fAttrAxes;
    EReduceOpMode fReduceOpMode;
    std::string fNX;
    std::string fNAxes;
    std::string fNY;
    std::vector<size_t> fShapeX;
    std::vector<size_t> fShapeY;
    std::vector<size_t> fShapeYNotPruned; // needed for fKeepdims=0


public:

   std::string Name() {
      if (fReduceOpMode == ReduceMean)  return "ReduceMean";
      else if (fReduceOpMode == ReduceSumsquare )  return "ReduceSumsquare";
      else if (fReduceOpMode == ReduceProd ) return "ReduceProd";
      else if (fReduceOpMode == ReduceSum) return "ReduceSum";
      return "Invalid";
   }

   ROperator_Reduce(){}
   ROperator_Reduce(int keepdims, std::vector<int64_t> attrAxes, std::string nameX, std::string nameAxes, std::string nameY):
   fkeepdims(keepdims), fAttrAxes(attrAxes), fNX(UTILITY::Clean_name(nameX)), fNAxes(UTILITY::Clean_name(nameAxes)), fNY(UTILITY::Clean_name(nameY)) {
      fReduceOpMode = Op;
   }

   // type of output given input
   std::vector<ETensorType> TypeInference(std::vector<ETensorType> input){
      return input;
   }

   // shape of output tensors given input tensors
   std::vector<std::vector<size_t>> ShapeInference(std::vector<std::vector<size_t>> input){
      auto ret = input; //suggest copy to compiler
      auto & outputShape = ret[0];
      for (size_t j = 0; j < fAttrAxes.size(); j++) {
         if (fAttrAxes[j] < 0) fAttrAxes[j] += outputShape.size();
         if (fAttrAxes[j] < 0 || (size_t) fAttrAxes[j] >= outputShape.size() )
            throw std::runtime_error("TMVA SOFIE Reduce Op - invalid axes values " + std::to_string(fAttrAxes[j]));
         // set to 1 the reduced dims
         outputShape[fAttrAxes[j]] = 1;
      }
      fShapeYNotPruned = outputShape;
      // in case of pruning dimension we need to sort axes attributes
      if (fkeepdims == 0) {
         auto ax = fAttrAxes;
         std::sort(ax.begin(), ax.end());
         for (size_t j = 0; j < ax.size(); j++) {
            // erase reduced dimensions, but keep last one
            if (outputShape.size() > 1) {
               outputShape.erase(outputShape.begin() + ax[j]);
               for (size_t k = j+1; k < ax.size(); k++)
                  ax[k] -= 1;  // decrease by one since we have removed a value
            }
         }
      }
      return ret;
   }
   void Initialize(RModel &model) {

      fUseSession = model.UseSession();

      if (!model.CheckIfTensorAlreadyExist(fNX)) {
         // input must be a graph input, or already initialized intermediate tensor
         throw std::runtime_error("TMVA SOFIE Reduce Op Input Tensor " + fNX + " is not found in model");
      }
      fShapeX = model.GetTensorShape(fNX);
      // check if tensor with axes is provided
      if (!fNAxes.empty()) {
         auto ax_shptr = model.GetInitializedTensorData(fNAxes);
         auto ax_ptr = static_cast<int64_t *>(ax_shptr.get());
         auto ax_shape = model.GetTensorShape(fNAxes);
         size_t ax_length = ConvertShapeToLength(ax_shape);
         fAttrAxes = std::vector<int64_t>(ax_ptr, ax_ptr+ax_length);
      } else if (fAttrAxes.empty()) {
         // in case no axes is passed assume full reduction
         fAttrAxes.resize(fShapeX.size());
         for (size_t i = 0; i < fAttrAxes.size(); i++)
            fAttrAxes[i] = i;
      }
      // find shape of Y and add it in the list of intermediate tensors
      fShapeY = ShapeInference({fShapeX})[0];
      model.AddIntermediateTensor(fNY, model.GetTensorType(fNX), fShapeY);

      //std::cout << "Reduce operator - axis = " << fAttrAxes[0] << " shape x " << ConvertShapeToString(fShapeX)
      //          << " output shape " << ConvertShapeToString(fShapeY) << std::endl;
   }

   std::string Generate(std::string OpName){
      OpName = "op_" + OpName;
      if (fShapeX.empty() || fShapeY.empty()) {
         throw std::runtime_error("TMVA SOFIE Reduce Op called to Generate without being initialized first");
      }

      size_t inputLength = TMVA::Experimental::SOFIE::ConvertShapeToLength(fShapeX);
      size_t outputLength = TMVA::Experimental::SOFIE::ConvertShapeToLength(fShapeY);

      auto inputStrides = TMVA::Experimental::SOFIE::UTILITY::ComputeStrideFromShape(fShapeX);
      // output stride (or not pruned vector)
      auto outputStrides = TMVA::Experimental::SOFIE::UTILITY::ComputeStrideFromShape(fShapeYNotPruned);

      // write here according to size of shape
      // in generation code can be done automatically
      // i0 =  i / stride0  % shape0; i1 = i / stride1 % shape1 and so on
      // and we have for the inverse
      // i = i0 * s0 + i1 * s1 + i2 * s2 + i3 * s3 ....

      // don't need to divide by last stride s[n-1] since it is 1 by definition

      std::stringstream out;
      out << "\n//----  operator " << Name() << "  " << OpName << "\n";
      // reset output tensors
      if (fReduceOpMode == ReduceProd)
         out << SP << "fTensor_" << fNY << ".assign(" << outputLength << ",1);\n";
      else
         out << SP << "fTensor_" << fNY << ".assign(" << outputLength << ",0);\n";

      out << SP << "for (size_t i = 0; i < " << inputLength << "; i++) {\n";

      size_t dim = fShapeX.size();   // this is the input dimension (e.g. 2, 3 or 4 or more)

      // here we find output index
      out << SP << SP << "size_t outputIndex = 0;\n";
      for (size_t k = 0; k < dim; k++) {
         if (std::find(fAttrAxes.begin(), fAttrAxes.end(), k) == fAttrAxes.end()) {
            // do for not reducing axes
            out << SP << SP << "size_t i_" << k << " = i / " << inputStrides[k] << " % " << fShapeX[k] << ";\n";
            out << SP << SP << "outputIndex += i_" << k << " * " << outputStrides[k] << ";\n";
         }
      }
      // now compute reduction
      out << SP << SP << "// compute reduction....\n";
      if (fReduceOpMode == ReduceProd)
         out << SP << SP << "tensor_" << fNY << "[outputIndex] *= tensor_" << fNX << "[i];\n";
      else if (fReduceOpMode == ReduceSum || fReduceOpMode == ReduceMean)
         out << SP << SP << "tensor_" << fNY << "[outputIndex] += tensor_" << fNX << "[i];\n";
      else if(fReduceOpMode == ReduceSumsquare){
         out << SP << SP << "tensor_" << fNY << "[outputIndex] += tensor_" << fNX << "[i] * tensor_" << fNX << "[i];\n";
      }
      out << SP << "}\n";  // end loop on input elements
      //normalize for reduced mean
      if(fReduceOpMode == ReduceMean){
         size_t reducedLength = inputLength/outputLength;
         out << SP << "for (size_t i = 0; i < " << outputLength << "; i++) {\n";
         out << SP << SP << "tensor_" << fNY << "[i] /= static_cast<float>(" << reducedLength << ");\n";
         out << SP  << "}\n";
      }

      return out.str();
   }

};

}//SOFIE
}//Experimental
}//TMVA


#endif //TMVA_SOFIE_ROPERATOR_Reduce

