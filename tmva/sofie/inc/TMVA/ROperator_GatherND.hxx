#ifndef TMVA_SOFIE_ROPERATOR_GatherND
#define TMVA_SOFIE_ROPERATOR_GatherND

#include "TMVA/SOFIE_common.hxx"
#include "TMVA/ROperator.hxx"
#include "TMVA/RModel.hxx"

#include <sstream>
#include <stdexcept>
#include <string>

namespace TMVA{
namespace Experimental{
namespace SOFIE{

class ROperator_GatherND final : public ROperator
{
private:

   size_t fBatchDims = 0;
   std::string fNX;
   std::string fNIndices;
   std::string fNY;

   std::vector<Dim> fShapeX;
   std::vector<Dim> fShapeIndices;
   std::vector<Dim> fShapeY;

   std::vector<int64_t> fIndices;  // indices vector in case they are known at initialization

   std::string fType;

public:
   ROperator_GatherND(){}
   ROperator_GatherND(int batch_dims, std::string nameX, std::string nameIndices, std::string nameY):
      fBatchDims(batch_dims), fNX(UTILITY::Clean_name(nameX)), fNIndices(UTILITY::Clean_name(nameIndices)), fNY(UTILITY::Clean_name(nameY)) {
         fInputTensorNames = { fNX, fNIndices };
         fOutputTensorNames = { fNY };
   }

   void Initialize(RModel& model) override {
      if (!model.CheckIfTensorAlreadyExist(fNX)) {
         throw std::runtime_error("TMVA SOFIE GatherND Op Input Tensor " + fNX + " is not found in model");
      }
      fShapeX = model.GetDimTensorShape(fNX);
      if (model.Verbose())
         std::cout << "GatherND - initial shape " << ConvertShapeToString(fShapeX) << " shape of indices "
               << ConvertShapeToString(model.GetDimTensorShape(fNIndices)) << std::endl;
      //  fShapeIndices can be  dynamic
      fShapeIndices = model.GetDimTensorShape(fNIndices);
      size_t q = fShapeIndices.size();
      // Axis in range [0, r) where r=rank(X)
      size_t r = fShapeX.size();

      if (q < 1) {
         throw std::runtime_error("TMVA SOFIE GatherND : rank of Indices is < 1");
      }
      if (r < 1) {
         throw std::runtime_error("TMVA SOFIE GatherND : rank of input tensor is < 1");
      }
      if (fBatchDims >= std::min(q,r)) {
         throw std::runtime_error("TMVA SOFIE GatherND : invalid batch dim value");
      }
      if (fBatchDims > 0) {
         for (size_t i = 0; i < fBatchDims; i++) {
            if (fShapeX[i] != fShapeIndices[i]) {
               std::cout << " input shape " << ConvertShapeToString(fShapeX) << " "
                         << " index shape " << ConvertShapeToString(fShapeIndices) << std::endl;
               throw std::runtime_error("TMVA SOFIE GatherND : invalid input or index shape for " + std::to_string(i));
            }
         }
      }

      //general case. Assumption is that last dimension of index shape is known (is not dynamic)
      if (fShapeIndices.back().isParam)
         throw std::runtime_error("TMVA SOFIE GatherND : Index_shape(-1) is not known");

      // output shape size (output rank)
      // is (q-1)+r -index_shape[-1]
      size_t last_index_shape = fShapeIndices.back().dim;
      if (last_index_shape < 1 || last_index_shape > r - fBatchDims) {
         throw std::runtime_error("TMVA SOFIE GatherND : Index_shape(-1) has wrong value " +
            std::to_string(last_index_shape));
      }

      size_t output_rank = r + q -1 - last_index_shape - fBatchDims;
      //fShapeY.resize(output_rank);
      // first index shape dimensions are same in output
      fShapeY = std::vector<Dim>(fShapeIndices.begin(), fShapeIndices.end() - 1);
      fShapeY.insert(fShapeY.end(), fShapeX.begin() + fBatchDims + last_index_shape, fShapeX.end());
      if (fShapeY.size() != output_rank) {
         std::cout << " input shape " << ConvertShapeToString(fShapeX) << " "
                         << " index shape " << ConvertShapeToString(fShapeIndices)
                         << " output shape " << ConvertShapeToString(fShapeY)
                         << " and output rank should be " << output_rank << std::endl;
         throw std::runtime_error("TMVA SOFIE GatherND : Something is wrong in initialization ");
      }

      if (!fIsOutputConstant) {
         // Add output tensor
         model.AddIntermediateTensor(fNY, model.GetTensorType(fNX), fShapeY);
         fType = ConvertTypeToString(model.GetTensorType(fNX));
         if (model.Verbose())
               std::cout <<  "GatherND: input " << fNX << " " << ConvertShapeToString(fShapeX) << " indices " << fNIndices << ConvertShapeToString(fShapeIndices)
                         << " -> " << fNY << " with shape " << ConvertShapeToString(fShapeY) << std::endl;
      }



      // // case indices tensor is initialized
      // if (model.IsInitializedTensor(fNIndices)) {
      //     // empty shape Indices is a scalar value for the indices
      //    size_t indicesLength = ConvertShapeToLength(model.GetTensorShape(fNIndices));
      //    int64_t* indicesData = static_cast<int64_t*>(model.GetInitializedTensorData(fNIndices).get());
      //    //flag index tensor as not writable (not sure this is needed since index tensor might be used in generated code)
      //    model.SetNotWritableInitializedTensor(fNIndices);
      //    // update indices data in case of negative dim values
      //    for (size_t i = 0; i < indicesLength; i++) {
      //       // move this at generation time?
      //       if (!fShapeX[fAttrAxis].isParam) {
      //          if (indicesData[i] < 0) {
      //             indicesData[i] += fShapeX[fAttrAxis].dim;
      //          }
      //       }
      //    }
      //    // Save in a vector GatherND Indices of size q
      //    fIndices = std::vector<int64_t>(indicesData, indicesData + indicesLength);
      // }

      // case input is known (type is an integer) and input indices is a scalar (or vector of size 1)
      // if (model.IsInitializedTensor(fNX) && q <= 1 && r == 1 && fIndices.size() > 0) {
      //    auto shapeX = ConvertShapeToInt(fShapeX);  // we assume model is not dynamic
      //    auto shapeY = ConvertShapeToInt(fShapeY);
      //    if (model.GetTensorType(fNX) == ETensorType::INT64) {
      //       auto inputData = static_cast<int64_t*>(model.GetInitializedTensorData(fNX).get());
      //       // if q <=1 and r = 1 output length = 1 (it is a scalar)
      //       std::vector<int64_t> outputData(1); //ConvertShapeToLength(shapeY));
      //       outputData[0] = inputData[fIndices[0]];
      //       model.AddConstantTensor(fNY, shapeY, outputData.data());
      //       if (model.Verbose())
      //          std::cout << "GatherND: " << fNX << " " << ConvertShapeToString(shapeX) << " -> " << fNY << " with shape " << ConvertShapeToString(shapeY)
      //              << " and values " << ConvertValuesToString(outputData) << " (constant) " << std::endl;
      //       fIsOutputConstant = true;
      //    }
      // }
      // // case input is a shape tensor  (r is == 1 by definition) and indices are known
      // else if (model.IsShapeTensor(fNX) && q <=1  && fIndices.size() > 0) {
      //    auto inputData = model.GetShapeTensorValues(fNX);
      //    // if r == 1 and q<=1 then output length is 1 (is a scalar or tensor of size1)
      //    std::vector<Dim> outputData(1);
      //    outputData[0] = inputData[fIndices[0]];
      //    if (outputData[0].isParam) {
      //       fIsOutputConstant = true;
      //       // shapeY can be scalar or vector of size1
      //       model.AddShapeTensor(fNY, outputData, fShapeY.size() == 0);
      //       if (model.Verbose())
      //          std::cout << "GatherND: " << fNX << " " << ConvertShapeToString(fShapeX) << " -> " << fNY << " with shape " << ConvertShapeToString(fShapeY)
      //              << " and values " << ConvertShapeToString(outputData) << " (shape) " << std::endl;
      //    } else {
      //       int64_t value = static_cast<int64_t>(outputData[0].dim);
      //       auto shapeY = ConvertShapeToInt(fShapeY);
      //       model.AddConstantTensor(fNY, shapeY, &value);
      //       fIsOutputConstant = true;
      //       if (model.Verbose())
      //          std::cout << "GatherND: " << fNX << " " << ConvertShapeToString(fShapeX) << " -> " << fNY << " with shape " << ConvertShapeToString(fShapeY)
      //              << " and values {" << value <<  "} (constant) " << std::endl;
      //    }
      // }

   }

   std::string Generate(std::string opName) override {
      if (fIsOutputConstant) {
         // no code to generate here for constant output. Tensor output is defined in Session constructor
         return "//---------------------------------------\n";
      }
      opName = "op_" + opName;
      std::stringstream out;
      out << "//--------- GatherND " << opName << " --> " << ConvertShapeToString(fShapeY) << "\n";
      // The shape of the output is q + r - 1
      size_t r = fShapeX.size();
      // Indices of shape q
      size_t q = fShapeIndices.size();
      // Strides
      auto stridesX = UTILITY::ComputeStrideFromShape(fShapeX);
      auto stridesY = UTILITY::ComputeStrideFromShape(fShapeY);
      auto stridesIndices = UTILITY::ComputeStrideFromShape(fShapeIndices);

      // case input_index_shape == rank of input
      size_t ss = fShapeIndices.back().dim;

      // check for negative indices
      auto indicesLength = ConvertDimShapeToLength(fShapeIndices);
      out << SP << "for (size_t i = 0; i < " << indicesLength << "; i++) {\n";
      out << SP << SP << "if (tensor_" << fNIndices << "[i] < 0 ) {\n";
      // corresponding input shape is  i % strides[N-1]
      out << SP << SP << SP << "size_t s_i = " << fShapeX[fBatchDims] << ";\n";
      for (size_t j = 1; j < ss; j++) {
         out << SP << SP << SP << "if (i % " << ss << " == " << j << ") s_i = " <<  fShapeX[fBatchDims+j] << ";\n";
      }
      out << SP << SP << SP << "const_cast<int64_t &>(tensor_" << fNIndices << "[i]) +=  s_i;\n";
      out << SP << SP << "}\n";
      out << SP << "}\n";
      // loop on batch dims
      std::string outIndex;
      std::string inIndex;
      std::string idIndex;
      for (size_t j = 0; j < fBatchDims; j++) {
         std::string index = "i_" + std::to_string(j);
         for (size_t k = 0; k <= j; k++)
            out << SP;
         out << "for (size_t " << index << " = 0; " << index << " < " << fShapeY[j] << "; " << index << "++) {\n";
         if (j > 0) {
            outIndex += " + ";
            inIndex += " + ";
            idIndex += " + ";
         }
         outIndex += index;
         if (stridesY[j].GetVal() != "1")
            outIndex += " * " + stridesY[j].GetVal();
         inIndex += index;
         if (stridesX[j].GetVal() != "1")
            inIndex += " * " + stridesX[j].GetVal();
         idIndex += index;
         if (stridesIndices[j].GetVal() != "1")
            idIndex += " * " + stridesIndices[j].GetVal();
      }
      // loop between b and q-1
      for (size_t j = fBatchDims; j < q - 1; j++) {
         std::string index = "i_" + std::to_string(j);
         for (size_t k = 0; k <= j; k++) out << SP;
         out << "for (size_t " << index << " = 0; " << index << " < " << fShapeY[j] << "; " << index << "++) {\n";
         if (j > 0) {
            outIndex += " + ";
            idIndex += " + ";
         }
         outIndex += index;
         if (stridesY[j].GetVal() != "1")
            outIndex += " * " + stridesY[j].GetVal();
         idIndex += index;
         if (stridesIndices[j].GetVal() != "1")
            idIndex += " * " + stridesIndices[j].GetVal();
      }
      for (size_t k = 0; k <= q - 1; k++) out << SP;
      out << "size_t inputIndex = " << inIndex;
      std::string indexIndex = idIndex;
      for (size_t l = 0; l < ss; l++) {
         if (l > 0)
            indexIndex = idIndex + " + " + std::to_string(l);
         // compute input index using index tensors
         if (!indexIndex.empty() || l>0)
            out << " + ";
         out << "tensor_" << fNIndices << "[" << indexIndex << "]";
         if (stridesX[fBatchDims + l].GetVal() != "1") out
             << " * " << stridesX[fBatchDims + l];
      }
      out << ";\n";
      for (size_t k = 0; k <= q - 1; k++) out << SP;
      // case slice is a scalar
      if (ss == r - fBatchDims) {
         out << "tensor_" << fNY << "[" << outIndex << "] = "
             << "tensor_" << fNX << "[inputIndex];\n";
      } else {
         // we make a copy of slice
         out << "std::copy(tensor_" << fNX << " + inputIndex, tensor_" << fNX << " + inputIndex + "
             << stridesX[fBatchDims + ss - 1] << ","
             << "tensor_" << fNY << "+" << outIndex << ");\n";
      }
      // close the loops

      // end loops j_k, j_{k + 1}, ..., j_{r - 2}
      for (size_t j = q-1; j > 0; j--) {
         for (size_t k = 0; k <j; k++) out << SP;
         out << "}\n";
      }

      return out.str();
   }

};

}//SOFIE
}//Experimental
}//TMVA


#endif //TMVA_SOFIE_ROPERATOR_RELU
