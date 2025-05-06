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

enum EReduceOpMode { ReduceMean, ReduceSum, ReduceSumSquare, ReduceProd, InvalidReduceOp };

template <typename T, EReduceOpMode Op>
class ROperator_Reduce final : public ROperator
{
private:
    /* Attributes*/
    bool fInputDimShape = false;
    int fkeepdims = 1; //default value
    std::vector<int64_t> fAttrAxes;
    EReduceOpMode fReduceOpMode;
    std::string fNX;
    std::string fNAxes;
    std::string fNY;
    std::vector<Dim> fShapeX;
    std::vector<Dim> fShapeY;
    std::vector<Dim> fShapeYNotPruned; // needed for fKeepdims=0


public:

   std::string Name() {
      if (fReduceOpMode == ReduceMean)  return "ReduceMean";
      else if (fReduceOpMode == ReduceSumSquare )  return "ReduceSumSquare";
      else if (fReduceOpMode == ReduceProd ) return "ReduceProd";
      else if (fReduceOpMode == ReduceSum) return "ReduceSum";
      return "Invalid";
   }

   ROperator_Reduce(){}
   ROperator_Reduce(int keepdims, std::vector<int64_t> attrAxes, std::string nameX, std::string nameAxes, std::string nameY):
   fkeepdims(keepdims), fAttrAxes(attrAxes), fNX(UTILITY::Clean_name(nameX)), fNAxes(UTILITY::Clean_name(nameAxes)), fNY(UTILITY::Clean_name(nameY)) {
      fReduceOpMode = Op;

      fInputTensorNames = { fNX };
      if(!fNAxes.empty()){
         fInputTensorNames.emplace_back(fNAxes);
      }

      fOutputTensorNames = { fNY };
   }

   // shape of output tensors given input tensors
   std::vector<Dim> DoShapeInference(const std::vector<Dim> &  input)  {
      auto ret = input; //suggest copy to compiler
      auto & outputShape = ret;
      for (size_t j = 0; j < fAttrAxes.size(); j++) {
         if (fAttrAxes[j] < 0) fAttrAxes[j] += outputShape.size();
         if (fAttrAxes[j] < 0 || (size_t) fAttrAxes[j] >= outputShape.size() )
            throw std::runtime_error("TMVA SOFIE Reduce Op - invalid axes values " + std::to_string(fAttrAxes[j]));
         // set to 1 the reduced dims
         outputShape[fAttrAxes[j]] = Dim{1};
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
   void Initialize(RModel& model) override {

      fUseSession = model.UseSession();

      if (!model.CheckIfTensorAlreadyExist(fNX)) {
         // input must be a graph input, or already initialized intermediate tensor
         throw std::runtime_error("TMVA SOFIE Reduce Op Input Tensor " + fNX + " is not found in model");
      }
      fShapeX = model.GetDimTensorShape(fNX);
      if (model.IsDynamicTensor(fNX))
         fInputDimShape = true;
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
      fShapeY = DoShapeInference(fShapeX);
      model.AddIntermediateTensor(fNY, model.GetTensorType(fNX), fShapeY);
      if (model.Verbose()){
         std::cout << Name() << " : " << fNX << " -> " << fNY << " shape " << ConvertShapeToString(fShapeY) << std::endl;
      }
      model.AddNeededStdLib("algorithm");
   }

   std::string Generate(std::string opName) override {
      opName = "op_" + opName;
      if (fShapeX.empty() || fShapeY.empty()) {
         throw std::runtime_error("TMVA SOFIE Reduce Op called to Generate without being initialized first");
      }

      auto inputLength = TMVA::Experimental::SOFIE::ConvertDimShapeToLength(fShapeX);
      auto outputLength = TMVA::Experimental::SOFIE::ConvertDimShapeToLength(fShapeY);

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
      out << "\n//----  operator " << Name() << "  " << opName << "\n";
      // check where is reduced axes are first or last one. In these case we can do a faster implementation
      enum EReduceDim {kFirst, kLast, kMiddle};
      EReduceDim reduceDims = kLast;
      int kmin = fShapeX.size()-fAttrAxes.size();
      for (int k = fShapeX.size()-1; k >= kmin; k--) {
         // if k is not a reduced axis is not last ones
         if (std::find(fAttrAxes.begin(), fAttrAxes.end(), k) == fAttrAxes.end()) {
            reduceDims = kMiddle;
            break;
         }
      }
      if (reduceDims == kMiddle) {
         reduceDims = kFirst;
         // check if at the beginning
         for (size_t k = 0; k < fAttrAxes.size(); k++) {
            // if k is not a reduced axis is not first ones
            if (std::find(fAttrAxes.begin(), fAttrAxes.end(), k) == fAttrAxes.end()) {
               reduceDims = kMiddle;
               break;
            }
         }
      }
      std::string reducedLength;
      if (fInputDimShape) {
         reducedLength = "reducedLength_" + opName;
         out << SP << "size_t " << reducedLength << " = " <<  inputLength << " / " << outputLength << ";\n";
      } else {
         int rLength = std::stoi(inputLength) / std::stoi(outputLength);
         reducedLength = std::to_string(rLength);
      }
      if (reduceDims == kLast) {
         //std::cout << "reduction for operator " << opName << " is last" << std::endl;
         // new faster implementation using a single loop
         // faster to loop first on reduced dimension and then output
         // reset output tensors

         // loop on output dimensions
         out << SP << "for (size_t i = 0; i < " << outputLength << "; i++) {\n";
         // loop on reduce dimensions
         std::string startingValue = (fReduceOpMode == ReduceProd) ? "1" : "0";
         out << SP << SP << "tensor_" << fNY << "[i] = " << startingValue << ";\n";
         out << SP << SP << "for (size_t j = 0; j < " << reducedLength << "; j++) {\n";

         if (fReduceOpMode == ReduceProd)
            out << SP << SP << SP <<  "tensor_" << fNY << "[i] *= tensor_" << fNX << "[i * " << reducedLength << " + j];\n";
         else if (fReduceOpMode == ReduceSum || fReduceOpMode == ReduceMean)
            out << SP << SP << SP <<  "tensor_" << fNY << "[i] += tensor_" << fNX << "[i * " << reducedLength << " + j];\n";
         else if(fReduceOpMode == ReduceSumSquare)
            out << SP << SP << SP <<  "tensor_" << fNY << "[i] += tensor_" << fNX << "[i * " << reducedLength << " + j] * tensor_"
                                    << fNX << "[i * " << reducedLength << " + j];\n";
         out << SP << SP << "}\n"; // end j loop
         if(fReduceOpMode == ReduceMean)
            out << SP << SP << "tensor_" << fNY << "[i] /= static_cast<float>(" << reducedLength << ");\n";

         out << SP << "}\n"; // end i loop
      } else if (reduceDims == kFirst) {
         //std::cout << "reduction for operator " << opName << " is first" << std::endl;
         // case reduction is at beginning
         // reset output tensors
         if (fReduceOpMode == ReduceProd)
            out << SP << "std::fill(tensor_" << fNY <<", tensor_"<< fNY <<" + "<< outputLength << ", 1);\n";
         else
            out << SP << "std::fill(tensor_" << fNY <<", tensor_"<< fNY <<" + "<< outputLength << ", 0);\n";

         out << SP << "for (size_t i = 0; i < " << reducedLength << "; i++) {\n";
         out << SP << SP << "for (size_t j = 0; j < " << outputLength << "; j++) {\n";

         if (fReduceOpMode == ReduceProd)
            out << SP << SP << SP << "tensor_" << fNY << "[j] *= tensor_" << fNX << "[i * " << outputLength << " + j];\n";
         else if (fReduceOpMode == ReduceSum || fReduceOpMode == ReduceMean)
            out << SP << SP << SP << "tensor_" << fNY << "[j] += tensor_" << fNX << "[i * " << outputLength << " + j];\n";
         else if(fReduceOpMode == ReduceSumSquare)
            out << SP << SP << SP << "tensor_" << fNY << "[j] += tensor_" << fNX << "[i * " << outputLength << " + j] * tensor_"
                                    << fNX << "[i * " << outputLength << " + j];\n";
         out << SP << SP << "}\n"; // end j loop
         out << SP  << "}\n"; // end i loop
         if(fReduceOpMode == ReduceMean) {
            out << SP  << "for (size_t j = 0; i < " << outputLength << "; j++) {\n";
            out << SP << SP << "tensor_" << fNY << "[j] /= static_cast<float>(" << reducedLength << ");\n";
            out << SP << "}\n"; // end j loop
         }
      }
      else
      { // standard case
         //std::cout << "reduction for operator " << opName << " is middle" << std::endl;
         // reset output tensors
         if (fReduceOpMode == ReduceProd)
            out << SP << "std::fill(tensor_" << fNY <<", tensor_"<< fNY <<" + "<< outputLength << ", 1);\n";
         else
            out << SP << "std::fill(tensor_" << fNY <<", tensor_"<< fNY <<" + "<< outputLength << ",0);\n";

         out << SP << "for (size_t i = 0; i < " << inputLength << "; i++) {\n";

         size_t dim = fShapeX.size(); // this is the input dimension (e.g. 2, 3 or 4 or more)

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
         else if (fReduceOpMode == ReduceSumSquare) {
            out << SP << SP << "tensor_" << fNY << "[outputIndex] += tensor_" << fNX << "[i] * tensor_" << fNX
                << "[i];\n";
         }
         out << SP << "}\n"; // end loop on input elements
         // normalize for reduced mean
         if (fReduceOpMode == ReduceMean) {
            out << SP << "for (size_t i = 0; i < " << outputLength << "; i++) {\n";
            out << SP << SP << "tensor_" << fNY << "[i] /= static_cast<float>(" << reducedLength << ");\n";
            out << SP << "}\n";
         }
      }

      return out.str();
   }

};

}//SOFIE
}//Experimental
}//TMVA


#endif //TMVA_SOFIE_ROPERATOR_Reduce

