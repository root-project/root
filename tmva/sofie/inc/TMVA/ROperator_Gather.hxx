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

   std::vector<Dim> fShapeX;
   std::vector<Dim> fShapeIndices;
   std::vector<Dim> fShapeY;

   std::vector<int64_t> fIndices;  // indices vector in case they are known at initialization

   std::string fType;

public:
   ROperator_Gather(){}
   ROperator_Gather(int64_t attrAxis, std::string nameX, std::string nameIndices, std::string nameY):
      fAttrAxis(attrAxis), fNX(UTILITY::Clean_name(nameX)), fNIndices(UTILITY::Clean_name(nameIndices)), fNY(UTILITY::Clean_name(nameY)) {
         fInputTensorNames = { fNX, fNIndices };
         fOutputTensorNames = { fNY };
   }

   std::vector<ETensorType> TypeInference(std::vector<ETensorType> input) override {
      return input;
   }

   std::vector<std::vector<size_t>> ShapeInference(std::vector<std::vector<size_t>> input) override {
      auto ret = input;
      return ret;
   }

   void Initialize(RModel& model) override {
      if (!model.CheckIfTensorAlreadyExist(fNX)) {
         throw std::runtime_error("TMVA SOFIE Gather Op Input Tensor " + fNX + " is not found in model");
      }
      fShapeX = model.GetDimTensorShape(fNX);
      if (model.Verbose())
         std::cout << "Gather - initial shape " << ConvertShapeToString(fShapeX) << " shape of indices "
               << ConvertShapeToString(model.GetDimTensorShape(fNIndices)) << std::endl;
      //  fShapeIndices can be  dynamic
      fShapeIndices = model.GetDimTensorShape(fNIndices);
      size_t q = fShapeIndices.size();
      // Axis in range [0, r) where r=rank(X)
      size_t r = fShapeX.size();
       // Set the axis
      if (fAttrAxis < 0) {
         fAttrAxis = fAttrAxis + int64_t(r);
      }


      // case indices tensor is initialized
      if (model.IsInitializedTensor(fNIndices)) {
          // empty shape Indices is a scalar value for the indices
         size_t indicesLength = ConvertShapeToLength(model.GetTensorShape(fNIndices));
         int64_t* indicesData = static_cast<int64_t*>(model.GetInitializedTensorData(fNIndices).get());
         //flag index tensor as not writable (not sure this is needed since index tensor might be used in generated code)
         model.SetNotWritableInitializedTensor(fNIndices);
         // update indices data in case of negative dim values
         for (size_t i = 0; i < indicesLength; i++) {
            // move this at generation time?
            if (!fShapeX[fAttrAxis].isParam) {
               if (indicesData[i] < 0) {
                  indicesData[i] += fShapeX[fAttrAxis].dim;
               }
            }
         }
         // Save in a vector gather Indices of size q
         fIndices = std::vector<int64_t>(indicesData, indicesData + indicesLength);
      }
      // Output shape
      if (model.Verbose())
         std::cout << "Gather: q and r " << q << " " << r << " shape indices " << ConvertShapeToString(fShapeIndices) << std::endl;

      if (fShapeY.empty()) {
         fShapeY.resize(q + r - 1);
         if (fAttrAxis > 0) {
            // Copy shape of X[0, ..., axis-1) to Shape of Y[0, ..., axis-1)
            std::copy(fShapeX.begin(), fShapeX.begin() + fAttrAxis, fShapeY.begin());
         }
         // Set shape of Y[axis, ..., axis + q)
         for (size_t i = 0; i < q; i++) {
            fShapeY[fAttrAxis + i] = Dim{ fShapeIndices[i]};
         }
         // Copy shape of X[axis + 1, ..., r) to shape of Y[axis + q, ... q + r - 1)
         std::copy(fShapeX.begin() + fAttrAxis + 1, fShapeX.end(), fShapeY.begin() + fAttrAxis + q);
      }
      // case input is known (type is an integer) and input indices is a scalar (or vector of size 1)
      if (model.IsInitializedTensor(fNX) && q <= 1 && r == 1 && fIndices.size() > 0) {
         auto shapeX = ConvertShapeToInt(fShapeX);  // we assume model is not dynamic
         auto shapeY = ConvertShapeToInt(fShapeY);
         if (model.GetTensorType(fNX) == ETensorType::INT64) {
            auto inputData = static_cast<int64_t*>(model.GetInitializedTensorData(fNX).get());
            // if q <=1 and r = 1 output length = 1 (it is a scalar)
            std::vector<int64_t> outputData(1); //ConvertShapeToLength(shapeY));
            outputData[0] = inputData[fIndices[0]];
            model.AddConstantTensor(fNY, shapeY, outputData.data());
            if (model.Verbose())
               std::cout << "Gather: " << fNX << " " << ConvertShapeToString(shapeX) << " -> " << fNY << " with shape " << ConvertShapeToString(shapeY)
                   << " and values " << ConvertValuesToString(outputData) << " (constant) " << std::endl;
            fIsOutputConstant = true;
         }
      }
      // case input is a shape tensor  (r is == 1 by definition) and indices are known
      else if (model.IsShapeTensor(fNX) && q <=1  && fIndices.size() > 0) {
         auto inputData = model.GetShapeTensorValues(fNX);
         // if r == 1 and q<=1 then output length is 1 (is a scalar or tensor of size1)
         std::vector<Dim> outputData(1);
         outputData[0] = inputData[fIndices[0]];
         if (outputData[0].isParam) {
            fIsOutputConstant = true;
            // shapeY can be scalar or vector of size1
            model.AddShapeTensor(fNY, outputData, fShapeY.size() == 0);
            if (model.Verbose())
               std::cout << "Gather: " << fNX << " " << ConvertShapeToString(fShapeX) << " -> " << fNY << " with shape " << ConvertShapeToString(fShapeY)
                   << " and values " << ConvertShapeToString(outputData) << " (shape) " << std::endl;
         } else {
            int64_t value = static_cast<int64_t>(outputData[0].dim);
            auto shapeY = ConvertShapeToInt(fShapeY);
            model.AddConstantTensor(fNY, shapeY, &value);
            fIsOutputConstant = true;
            if (model.Verbose())
               std::cout << "Gather: " << fNX << " " << ConvertShapeToString(fShapeX) << " -> " << fNY << " with shape " << ConvertShapeToString(fShapeY)
                   << " and values {" << value <<  "} (constant) " << std::endl;
         }
      }
      if (!fIsOutputConstant) {
         // Add output tensor
         model.AddIntermediateTensor(fNY, model.GetTensorType(fNX), fShapeY);
         fType = ConvertTypeToString(model.GetTensorType(fNX));
         if (model.Verbose())
               std::cout <<  "Gather: input " << fNX << " " << ConvertShapeToString(fShapeX) << " indices " << fNIndices << ConvertShapeToString(fShapeIndices)
                         << " -> " << fNY << " with shape " << ConvertShapeToString(fShapeY) << std::endl;
      }
   }

   std::string Generate(std::string opName) override {
      if (fIsOutputConstant) {
         // no code to generate here for constant output. Tensor output is defined in Session constructor
         return "//---------------------------------------\n";
      }
      opName = "op_" + opName;
      std::stringstream out;
      out << "//--------- Gather " << opName << " --> " << ConvertShapeToString(fShapeY) << "\n";
      // The shape of the output is q + r - 1
      size_t r = fShapeX.size();
      // Indices of shape q
      size_t q = fShapeIndices.size();
      // Strides
      auto stridesX = UTILITY::ComputeStrideFromShape(fShapeX);
      auto stridesY = UTILITY::ComputeStrideFromShape(fShapeY);
      auto stridesIndices = UTILITY::ComputeStrideFromShape(fShapeIndices);

      // case fIndices is not known we need to correct for negative axis indices at run-time
      if (fIndices.empty()) {
         auto indicesLength = ConvertDimShapeToLength(fShapeIndices);
         out << SP << "// correct in case of negative gather indices\n";
         out << SP << "for (size_t i = 0; i < " << indicesLength << "; i++){\n";
         out << SP << SP << "if (tensor_" << fNIndices << "[i] < 0)\n";
         out << SP << SP << SP <<  "tensor_" << fNIndices << "[i] += " << fShapeX[fAttrAxis] << ";\n";
         out << SP << "}\n";
      }

      // Fill the output Y[j_0, j_1, ..., j_{axis - 1}, i_0, i_1, ..., i_{q - 1}, j_{axis + 1}, ..., j_{r - 1}]
      // [0 ... axis) [axis ... axis + q) [axis + q ... q + r - 1)
      // iterate in [0 ... axis) [0 ... q) [axis ... r - 1)
      // for j_0, j_1, ..., j_{axis-1}

      for (size_t j = 0; j < size_t(fAttrAxis); j++) {
         std::string index = "j_" + std::to_string(j);
         for (size_t k = 0; k <= j; k++) out << SP;
         out << "for (size_t " << index << " = 0; " << index << " < " << fShapeY[j] << "; " << index << "++) {\n";
      }
      // for i_0, i_1, ..., i_{q - 1}
      for (size_t i = 0; i < q; i++) {
         std::string index = "i_" + std::to_string(i);
         for (size_t k = 0; k <= i + fAttrAxis; k++) out << SP;
         out << "for (size_t " << index << " = " << 0 << "; " << index << " < " << fShapeIndices[i] << "; " << index << "++) {\n";
      }
      // for j_axis, j_{axis + 1}, ..., j_{r - 1}
      for (size_t j = fAttrAxis; j + 1 < r; j++) {
         std::string index = "j_" + std::to_string(q+j); // annotate index using output axis
         for (size_t k = 0; k <= q + j; k++) out << SP;
         out << "for (size_t " << index << " = 0; " << index << " < " << fShapeY[q + j] << "; " << index << "++) {\n";
      }

      // add a scope for local variables in case above loop are not done
      if (fAttrAxis == 0 && q == 0 && r <= 1)
         out << SP << "{   // scalar case \n";

      // output index
      for (size_t k = 0; k < q + r; k++) out << SP;
      out << "size_t y_index = ";
      for (size_t j = 0; j < size_t(fAttrAxis); j++) {
         if (j > 0) out << " + ";
         out << "j_" << j;
         if (stridesY[j].dim != 1) out << " * " << stridesY[j];
      }
      for (size_t i = 0; i < q; i++) {
         if (fAttrAxis + i > 0) out << " + ";
         out << "i_" << i;
         if (stridesY[fAttrAxis + i].dim != 1) out << " * " << stridesY[fAttrAxis + i];
      }
      for (size_t j = fAttrAxis; j + 1 < r; j++) {
         if (j + q > 0) out << " + ";
         out << "j_" << q+j;
         if (stridesY[q+j].dim != 1) out << " * " << stridesY[q+j];
      }
      // empty case
      if (fAttrAxis == 0 && q == 0 && r <= 1)
         out << "0";
      out << ";\n";

      // input Indices
      for (size_t k = 0; k < q + r; k++) out << SP;
      out << "size_t i_index = ";
      for (size_t i = 0; i < q; i++) {
         if (i > 0) out << " + ";
         out << "i_" << i;
         if (stridesIndices[i].dim != 1) out << " * " << stridesIndices[i];
      }
      // empty case
      if (q == 0)
         out << "0";
      out << ";\n";

      // K
      for (size_t k = 0; k < q + r; k++) out << SP;
      out << "size_t k = static_cast<size_t>(" << "tensor_" << fNIndices << "[i_index]" << ");\n";
      // Input
      for (size_t k = 0; k < q + r; k++) out << SP;
      out << "size_t x_index = k";
      if (stridesX[fAttrAxis].dim != 1) out << " * " << stridesX[fAttrAxis];
      for (size_t j = 0; j < size_t(fAttrAxis); j++) {
         out << " + ";
         out << " j_" << j;
         if (stridesX[j].dim != 1) out << " * " << stridesX[j];
      }
      // for input corresponding stride is axis+1,.... r
      // loop is on j from fAttrAxis, so consider stridesX[j+1]
      for (size_t j = fAttrAxis; j+1 < r; j++) {
         out << " + ";
         out << " j_" << q+j;
         if (stridesX[j+1].dim != 1) out << " * " << stridesX[j+1];
      }
      out << ";\n";
      for (size_t k = 0; k < q + r; k++) out << SP;
      out << "tensor_" << fNY << "[y_index] = tensor_" << fNX << "[x_index];\n";

      // end loops j_k, j_{k + 1}, ..., j_{r - 2}
      for (size_t j = q+r-1; j > 0; j--) {
         for (size_t k = 0; k <j; k++) out << SP;
         out << "}\n";
      }
      // close empty scope if it was opened
      if (q == 0 && fAttrAxis == 0 && r <= 1)
         out << SP << "}   // close Gather scope for scalar case \n";


      return out.str();
   }

};

}//SOFIE
}//Experimental
}//TMVA


#endif //TMVA_SOFIE_ROPERATOR_RELU
