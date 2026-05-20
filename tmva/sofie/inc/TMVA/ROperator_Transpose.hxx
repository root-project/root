#ifndef TMVA_SOFIE_ROPERATOR_TRANSPOSE
#define TMVA_SOFIE_ROPERATOR_TRANSPOSE

#include "TMVA/SOFIE_common.hxx"
#include "TMVA/ROperator.hxx"
#include "TMVA/RModel.hxx"

#include <sstream>
#include <cassert>

namespace TMVA{
namespace Experimental{
namespace SOFIE{



class ROperator_Transpose final : public ROperator
{

private:

   std::vector<int64_t> fAttrPerm;

   std::string fNX;
   std::string fNY;
   std::vector<Dim> fShapeX;
   std::vector<Dim> fShapeY;

public:

   ROperator_Transpose(){}
   ROperator_Transpose(std::vector<int64_t> attr_perm, std::string nameData, std::string nameOutput):
      fAttrPerm(attr_perm), fNX(UTILITY::Clean_name(nameData)), fNY(UTILITY::Clean_name(nameOutput)) {
            fInputTensorNames = { fNX };
            fOutputTensorNames = { fNY };
   }


   std::vector<ETensorType> TypeInference(std::vector<ETensorType> input) override {
      return input;
   }

   std::vector<std::vector<size_t>> ShapeInference(std::vector<std::vector<size_t>> input) override {
      if (input.size() > 1) throw std::runtime_error("TMVA SOFIE Tranpose Op Shape Inference only need 1 input tensor");
      auto& data = input[0];
      if (fAttrPerm.size() != data.size() )
         throw std::runtime_error("TMVA SOFIE Tranpose Op - Invalid axes attributes");

      std::vector<size_t> output_shape(fAttrPerm.size());
      for (size_t i = 0; i < fAttrPerm.size(); i++){
         output_shape[i] = data[fAttrPerm[i]];
      }
      std::vector<std::vector<size_t>> ret;
      ret.push_back(output_shape);
      return ret;
   }

   template<class T>
   void ProcessInitializedTensor(RModel& model) {
      // case input is a constant or initialized tensor we perform here the transpose
      // here we know the shape
      auto shapeX = ConvertShapeToInt(fShapeX);
      auto shapeY = ConvertShapeToInt(fShapeY);
      fIsOutputConstant = true;

      auto inStrides = UTILITY::ComputeStrideFromShape(shapeX);
      auto outStrides = UTILITY::ComputeStrideFromShape(shapeY);
      size_t length = ConvertShapeToLength(shapeY);
      auto inputData = static_cast<T *>(model.GetInitializedTensorData(fNX).get());
      size_t dim = fShapeX.size();
      std::vector<size_t> outputIdx(dim);
      std::vector<T> outputData(length);
      for (size_t i = 0; i < length; i++) {
         outputIdx[0] = i / outStrides[0];
         for (size_t j = 1; j < dim; j++) {
            outputIdx[j] = (i % outStrides[j - 1]) / outStrides[j];
         }
         // compute input index
         size_t inputIndex = 0;
         for (size_t j = 0; j < dim; j++) {
            // find value in fAtrrPerm corresponding to j
            int k = std::find(fAttrPerm.begin(), fAttrPerm.end(), j) - fAttrPerm.begin();
            inputIndex += outputIdx[k] * inStrides[j];
         }
         outputData[i] = inputData[inputIndex];
      }
      model.AddConstantTensor<T>(fNY, shapeY, outputData.data());
      if (model.Verbose()) {
         std::cout << "Transpose: output is a constant tensor " << ConvertShapeToString(shapeY) << " : "
                   << ConvertValuesToString(outputData) << std::endl;
      }
   }

   void Initialize(RModel& model) override {
      if (model.CheckIfTensorAlreadyExist(fNX) == false){   //input must be a graph input, or already initialized intermediate tensor
         std::cout<<"Input tensor for transpose: "<<fNX<<'\n';
         throw std::runtime_error("TMVA SOFIE Tranpose Op Input Tensor is not found in model");
      }
      fShapeX = model.GetDimTensorShape(fNX);
      if (fAttrPerm.empty()){
         fAttrPerm.reserve(fShapeX.size());
         for (int i = fShapeX.size() - 1; i >= 0; i--){
            fAttrPerm.push_back(i);
         }
      }

      // inference of output shape
      if (fAttrPerm.size() != fShapeX.size() )
         throw std::runtime_error("TMVA SOFIE Tranpose Op - Invalid axes attributes");

      fShapeY.resize(fAttrPerm.size());
      for (size_t i = 0; i < fAttrPerm.size(); i++){
         fShapeY[i] = fShapeX[fAttrPerm[i]];
      }

      if (model.IsInitializedTensor(fNX) ) {
         auto type = model.GetTensorType(fNX);
         switch(type) {
            case ETensorType::FLOAT:
               ProcessInitializedTensor<float>(model);
               break;
            case ETensorType::INT64:
               ProcessInitializedTensor<int64_t>(model);
               break;
            case ETensorType::BOOL:
               ProcessInitializedTensor<uint8_t>(model);
               break;
            case ETensorType::UINT8:
               ProcessInitializedTensor<uint8_t>(model);
               break;
            default:
               std::cout << "Transpose - no support for initialized tensor of type " << ConvertTypeToString(type) << std::endl;
         }
         return;
      }
      // case of intermediate tensors (non constant)
      model.AddIntermediateTensor(fNY, model.GetTensorType(fNX), fShapeY);
      if (model.Verbose()) {
         std::cout << "Transpose ---> " << fNY << " " <<  ConvertDimShapeToString(fShapeY) << std::endl;
      }
   }

   std::string Generate(std::string opName) override {
      if (fIsOutputConstant) return "";  //no op for constant tensors
      opName = "op_" + opName;
      if (fShapeX.empty() || fShapeY.empty()){
         throw std::runtime_error("TMVA SOFIE Transpose Op called to Generate without being initialized first");
      }
      auto stridesX = UTILITY::ComputeStrideFromShape(fShapeX);
      auto stridesY = UTILITY::ComputeStrideFromShape(fShapeY);

      auto intShapeX = ConvertShapeToInt(fShapeX);
      size_t rank = fShapeX.size();
      bool isDynamic = (intShapeX.empty() && rank > 0);

      std::string constQualifier = (isDynamic) ? "const" : "constexpr";

      std::stringstream out;

      out << SP << "///------- Transpose operator " << opName << ConvertDimShapeToString(fShapeX)
                  << " --> " << ConvertDimShapeToString(fShapeY) << std::endl;

      // Implement more efficient implementation of transpose operator using strides
      // For 2-dim rank tensors we could have an optimised implementation for rank = 2 tensors using Tiles

      // General implementation : start pre-computing strides as const expr
      // Emit strides for X (input) and Y (output) as constexpr
      out << SP << "{\n";
      out << SP << SP << "// Pre-baked input strides (row-major)\n";
      out << SP << SP << constQualifier << " size_t " << opName << "_strX[] = {";
      for (size_t i = 0; i < rank; ++i)
         out << stridesX[i] << (i + 1 < rank ? ", " : "");
      out << "};\n";

      out << SP << SP << "// Pre-baked output strides (row-major)\n";
      out << SP << SP << constQualifier << " size_t " << opName << "_strY[] = {";
      for (size_t i = 0; i < rank; ++i)
         out << stridesY[i] << (i + 1 < rank ? ", " : "");
      out << "};\n\n";

      // Check if last perm axis == rank-1 (contiguous inner axis fast path)
      bool innerContiguous = (fAttrPerm.back() == (int64_t) (rank - 1));
      size_t outerRank    = innerContiguous ? rank - 1 : rank;
      size_t  innerSize    = innerContiguous ? (isDynamic ? 0 : intShapeX[fAttrPerm[rank - 1]])
                             : 1;

      if (innerContiguous && !isDynamic && innerSize > 1) {
         // ---- Fast path: innermost axis is contiguous in source -----
         out << SP << SP
             << "// Fast path: last permuted axis is contiguous in source\n";
         out << SP << SP
             << "// Inner " << innerSize << " elements copied with pointer arithmetic\n";

         // Nested loops over all axes except the last
         EmitNestedLoops(out, outerRank, fShapeY);

         // Compute flat src and dst offsets for the current outer indices
         out << SP << SP << SP << "size_t src_off = ";
         for (size_t i = 0; i < outerRank; ++i) {
            out << "idx_" << i << " * " << opName << "_strX["
                << fAttrPerm[i] << "]";
            if (i + 1 < outerRank) out << " + ";
         }
         out << ";\n";

         out << SP << SP << SP << "size_t dst_off = ";
         for (size_t i = 0; i < outerRank; ++i) {
            out << "idx_" << i << " * " << opName << "_strY[" << i << "]";
            if (i + 1 < outerRank) out << " + ";
         }
         out << ";\n";

         // Inner memcpy-style copy over the contiguous axis
         out << SP << SP << SP
             << "std::copy(tensor_" << fNX << " + src_off, "
             << "tensor_" << fNX << " + src_off + " << innerSize << ", "
             << "tensor_" << fNY << " + dst_off);\n";

         CloseNestedLoops(out, outerRank);

      } else {

       // ---- General path: per-element index arithmetic -------------
         out << SP << SP << "// General N-D transpose\n";

         EmitNestedLoops(out, rank, fShapeY);

         // Flat source index: sum over perm[i] * strideX[perm[i]]
         out << SP << SP << SP << "size_t src_idx = ";
         for (size_t i = 0; i < rank; ++i) {
            out << "idx_" << i << " * " << opName << "_strX[" << fAttrPerm[i] << "]";
            if (i + 1 < rank) out << " + ";
         }
         out << ";\n";

         // Flat destination index: sum over i * strideY[i]
         out << SP << SP << SP << "size_t dst_idx = ";
         for (size_t i = 0; i < rank; ++i) {
            out << "idx_" << i << " * " << opName << "_strY[" << i << "]";
            if (i + 1 < rank) out << " + ";
         }
         out << ";\n";

         out << SP << SP << SP
             << "tensor_" << fNY << "[dst_idx] = "
             << "tensor_" << fNX << "[src_idx];\n";

         CloseNestedLoops(out, rank);

      }

      out << SP << "}\n";
      return out.str();
   }


};

}//SOFIE
}//Experimental
}//TMVA


#endif //TMVA_SOFIE_ROPERATOR_TRANSPOSE
