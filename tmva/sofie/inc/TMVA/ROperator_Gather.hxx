#ifndef TMVA_SOFIE_ROPERATOR_GATHER
#define TMVA_SOFIE_ROPERATOR_GATHER

#include "TMVA/SOFIE_common.hxx"
#include "TMVA/ROperator.hxx"
#include "TMVA/RModel.hxx"

#include <sstream>
#include <stdexcept>
#include <string>

namespace TMVA{
namespace Experimental{
namespace SOFIE{

class ROperator_Gather final : public ROperator
{
private:

   int64_t fAttrAxis = 0;

   std::string fNX;
   std::string fNIndices;
   std::string fNY;

   std::vector<size_t> fShapeX;
   std::vector<size_t> fShapeIndices;
   std::vector<size_t> fShapeY;

   std::vector<int64_t> fIndices;

   std::string fType;

public:
   ROperator_Gather(){}
   ROperator_Gather(int64_t attrAxis, std::string nameX, std::string nameIndices, std::string nameY):
      fAttrAxis(attrAxis), fNX(UTILITY::Clean_name(nameX)), fNIndices(UTILITY::Clean_name(nameIndices)), fNY(UTILITY::Clean_name(nameY)) {
   }

   std::vector<ETensorType> TypeInference(std::vector<ETensorType> input){
      return input;
   }

   std::vector<std::vector<size_t>> ShapeInference(std::vector<std::vector<size_t>> input){
      auto ret = input;
      return ret;
   }

   void Initialize(RModel& model) {
      if (!model.CheckIfTensorAlreadyExist(fNX)) {
         throw std::runtime_error("TMVA SOFIE Gather Op Input Tensor " + fNX + " is not found in model");
      }
      fShapeX = model.GetTensorShape(fNX);
      if (!model.IsInitializedTensor(fNIndices)) {
         throw
            std::runtime_error("TMVA::SOFIE - Tensor " + fNIndices + " is not initialized.");
      }
      int64_t* indicesData = static_cast<int64_t*>(model.GetInitializedTensorData(fNIndices).get());
      fShapeIndices = model.GetTensorShape(fNIndices);
      size_t q = fShapeIndices.size();
      // Axis in range [0, r) where r=rank(X)
      size_t r = fShapeX.size();
      // Set the axis
      if (fAttrAxis < 0) {
         fAttrAxis = fAttrAxis + int64_t(r);
      }
      // Indices of size q
      // empty fShapeIndices is a scalar value for the indices
      size_t indicesLength = ConvertShapeToLength(fShapeIndices);
      fIndices = std::vector<int64_t>(indicesData, indicesData + indicesLength);
      for (size_t i = 0; i < indicesLength; i++) {
         if (fIndices[i] < 0) {
            fIndices[i] += fShapeX[fAttrAxis];
         }
      }
      // Output shape
      if (fShapeY.empty()) {
         fShapeY.resize(q + r - 1);
         if (fAttrAxis > 0) {
            // Copy shape of X[0, ..., axis) to Shape of Y[0, ..., axis)
            std::copy(fShapeX.begin(), fShapeX.begin() + fAttrAxis, fShapeY.begin());
         }
         // Set shape of Y[axis, ..., axis + q)
         for (size_t i = 0; i < q; i++) {
            fShapeY[fAttrAxis + i] = fShapeIndices[i];
         }
         // Copy shape of X[axis + 1, ..., axis + r) to shape of Y[axis + q, ... q + r - 1)
         std::copy(fShapeX.begin() + fAttrAxis + 1, fShapeX.end(), fShapeY.begin() + fAttrAxis + q);
      }
      // Add output tensor
      model.AddIntermediateTensor(fNY, model.GetTensorType(fNX), fShapeY);
      fType = ConvertTypeToString(model.GetTensorType(fNX));
   }

   std::string Generate(std::string OpName) {
      OpName = "op_" + OpName;
      std::stringstream out;
      // The shape of the output is q + r - 1
      size_t r = fShapeX.size();
      // Indices of shape q
      size_t q = fShapeIndices.size();
      // Strides
      std::vector<size_t> stridesX = UTILITY::ComputeStrideFromShape(fShapeX);
      std::vector<size_t> stridesY = UTILITY::ComputeStrideFromShape(fShapeY);
      std::vector<size_t> stridesIndices = UTILITY::ComputeStrideFromShape(fShapeIndices);
      // Indices vector
      out << SP << "std::vector<int64_t> " << OpName << "_indices = {";
      size_t indicesLength = ConvertShapeToLength(fShapeIndices);
      for (size_t i = 0; i < indicesLength; i++) {
         out << fIndices[i] << (i + 1 < indicesLength? ", " : "};\n");
      }
      // Fill the output Y[j_0, j_1, ..., j_{axis - 1}, i_0, i_1, ..., i_{q - 1}, j_{axis + 1}, ..., j_{r - 1}]
      // [0 ... axis) [axis ... axis + q) [axis + q ... q + r - 1)
      // iterate in [0 ... axis) [0 ... q) [axis ... r - 1)
      // for j_0, j_1, ..., j_{axis-1}
      for (size_t j = 0; j < size_t(fAttrAxis); j++) {
         std::string index = "j_" + std::to_string(j);
         out << SP << "for (size_t " << index << " = 0; " << index << " < " << fShapeY[j] << "; " << index << "++) {\n";
      }
      // for i_0, i_1, ..., i_{q - 1}
      for (size_t i = 0; i < q; i++) {
         std::string index = "i_" + std::to_string(i);
         out << SP << SP << "for (size_t " << index << " = " << 0 << "; " << index << " < " << fShapeIndices[i] << "; " << index << "++) {\n";
      }
      // for j_axis, j_{axis + 1}, ..., j_{r - 1}
      for (size_t j = fAttrAxis; j + 1 < r; j++) {
         std::string index = "j_" + std::to_string(j);
         out << SP << SP << SP << "for (size_t " << index << " = 0; " << index << " < " << fShapeY[q + j] << "; " << index << "++) {\n";
      }

      out << SP << SP << SP << "size_t y_index = 0;\n";
      for (size_t j = 0; j < size_t(fAttrAxis); j++) {
         out << SP << SP << SP << "y_index += j_" + std::to_string(j) + " * " << stridesY[j] << ";\n";
      }
      for (size_t i = 0; i < q; i++) {
         out << SP << SP << SP << "y_index += i_" + std::to_string(i) + " * " << stridesY[fAttrAxis + i] << ";\n";
      }
      for (size_t j = fAttrAxis; j + 1 < r; j++) {
         out << SP << SP << SP << "y_index += j_" + std::to_string(j) + " * " << stridesY[q + j] << ";\n";
      }
      // Indices
      out << SP << SP << SP << "size_t i_index = 0;\n";
      for (size_t i = 0; i < q; i++) {
         out << SP << SP << SP << "i_index += i_" + std::to_string(i) + " * " << stridesIndices[i] << ";\n";
      }
      // K
      out << SP << SP << SP << "size_t k = static_cast<size_t>(" << OpName << "_indices[i_index]" << ");\n";
      // Input
      out << SP << SP << SP << "size_t x_index = k * " << stridesX[fAttrAxis] << ";\n";
      for (size_t j = 0; j < size_t(fAttrAxis); j++) {
         out << SP << SP << SP << "x_index += j_" + std::to_string(j) + " * " << stridesX[j] << ";\n";
      }
      for (size_t j = fAttrAxis + 1; j < r; j++) {
         out << SP << SP << SP << "x_index += j_" + std::to_string(j - 1) + " * " << stridesX[j] << ";\n";
      }
      out << SP << SP << SP << "tensor_" << fNY << "[y_index] = tensor_" << fNX << "[x_index];\n";

      // end loops j_k, j_{k + 1}, ..., j_{r - 2}
      for (size_t j = fAttrAxis; j + 1 < r; j++) {
         out << SP << SP << SP << "}\n";
      }
      // end loops i_0, i_1, ..., i_{q - 1}
      for (size_t i = 0; i < q; i++) {
         out << SP << SP << "}\n";
      }
      // end loops j_0, j_1, ..., j_{axis - 1}
      for (size_t j = 0; j < size_t(fAttrAxis); j++) {
         out << SP << "}\n";
      }

      return out.str();
   }

};

}//SOFIE
}//Experimental
}//TMVA


#endif //TMVA_SOFIE_ROPERATOR_RELU
