#ifndef TMVA_SOFIE_ROPERATOR_LAYERNORMALIZATION
#define TMVA_SOFIE_ROPERATOR_LAYERNORMALIZATION

#include "TMVA/RModel.hxx"
#include "TMVA/SOFIE_common.hxx"

#include <sstream>
#include <string>

namespace TMVA {
namespace Experimental {
namespace SOFIE {

template <typename T>
class ROperator_LayerNormalization : public ROperator {
private:
   int64_t fAttrAxis;
   float fAttrEpsilon;
   size_t fAttrStashType;

   std::string fNX;
   std::string fNScale;
   std::string fNB;
   std::string fNY;
   std::string fNMean;
   std::string fNInvStdDev;

   std::string fNCastedX;
   std::string fNNormalizedX;
   std::string fNBroadcastedB;

   std::vector<size_t> fShapeX;
   std::vector<size_t> fShapeScale;
   std::vector<size_t> fShapeB;
   std::vector<size_t> fShapeY;
   std::vector<size_t> fShapeMean;
   std::vector<size_t> fShapeInvStdDev;

   size_t fAxis; // axis in [0, size)
   size_t fSize; // Size of the input
   // size_t fAxisDim;
   size_t fLength; // Length of the input X

   std::vector<size_t> fNormalizedShape;
   std::vector<size_t> fAxesShape;
   size_t fNormalizedLength;
   size_t fAxesLength;

   std::string fType;

public:
   ROperator_LayerNormalization() {}

   ROperator_LayerNormalization(int64_t axis, float epsilon, size_t stashType, const std::string &nameX,
                                const std::string &nameScale, const std::string &nameB, const std::string &nameY,
                                const std::string &nameMean, const std::string &nameInvStdDev)
      : fAttrAxis(axis), fAttrEpsilon(epsilon), fAttrStashType(stashType), fNX(nameX), fNScale(nameScale), fNB(nameB),
        fNY(nameY), fNMean(nameMean), fNInvStdDev(nameInvStdDev)
   {
   }

   std::vector<std::vector<size_t>> ShapeInference(std::vector<std::vector<size_t>> input) override { return input; }

   std::vector<ETensorType> TypeInference(std::vector<ETensorType> input) override { return input; }

   void Initialize(RModel &model) override
   {
      if (!model.CheckIfTensorAlreadyExist(fNX)) {
         throw std::runtime_error("TMVA::SOFIE - Tensor " + fNX + " not found.");
      }
      fShapeX = model.GetTensorShape(fNX);
      fShapeY = fShapeX;
      model.AddIntermediateTensor(fNY, model.GetTensorType(fNX), fShapeY);
      // Type of the output
      fType = ConvertTypeToString(model.GetTensorType(fNX));
      // Size of the input
      fSize = fShapeX.size();
      // Axis in [0, size)
      fAxis = (fAttrAxis < 0) ? fSize + fAttrAxis : fAttrAxis;
      // Shape of fShapeX[0, ..., fAxis)
      fAxesShape = std::vector<size_t>(fShapeX.begin(), fShapeX.begin() + fAxis);
      // Lenght of the axes
      fAxesLength = ConvertShapeToLength(fAxesShape);
      // Shape of fShapeX[fAxis, ..., fSize)
      fNormalizedShape = std::vector<size_t>(fShapeX.begin() + fAxis, fShapeX.end());
      // Length of the normalized axis
      fNormalizedLength = ConvertShapeToLength(fNormalizedShape);
      // length of the input
      fLength = ConvertShapeToLength(fShapeX);
      // Type of mean and std
      ETensorType type = (fAttrStashType == 1) ? ETensorType::FLOAT : model.GetTensorType(fNX);
      // Mean
      if (fNMean.empty()) {
         fNMean = "Mean" + fNX;
         model.AddIntermediateTensor(fNMean, type, {fAxesLength});
      }
      // Inverse Standard Deviation
      if (fNInvStdDev.empty()) {
         fNInvStdDev = "InvStdDev" + fNX;
         model.AddIntermediateTensor(fNInvStdDev, type, {fAxesLength});
      }
      // Cast X to float
      if (fAttrStashType == 1 && model.GetTensorType(fNX) != ETensorType::FLOAT) {
         fNCastedX = "Casted" + fNX;
         model.AddIntermediateTensor(fNCastedX, ETensorType::FLOAT, fShapeX);
         fNNormalizedX = "Normalized" + fNX;
         model.AddIntermediateTensor(fNNormalizedX, ETensorType::FLOAT, fShapeX);
      }
      // Broadcast the bias
      if (!fNB.empty()) {
         fShapeB = model.GetTensorShape(fNB);
         size_t lengthB = ConvertShapeToLength(fShapeB);
         if (lengthB < fLength) {
            fNBroadcastedB = "Broadcasted" + fNB;
            model.AddIntermediateTensor(fNBroadcastedB, ConvertStringToType(fType), fShapeX);
         }
      }
   }

   std::string GenerateInitCode() override
   {
      std::stringstream out;
      if (!fNBroadcastedB.empty()) {
         out << SP << "// Broadcasting the bias of LayerNormlization op\n";
         out << SP << "{\n";
         out << SP << SP << "float* data = TMVA::Experimental::SOFIE::UTILITY::UnidirectionalBroadcast<float>(tensor_";
         out << fNB << ", " << ConvertShapeToString(fShapeB) << ", " << ConvertShapeToString(fShapeX) << ");\n";
         out << SP << "std::copy(data, data + " << fLength << ", tensor_" << fNBroadcastedB << ");\n";
         out << SP << "delete[] data;\n";
         out << SP << "}\n";
      }
      return out.str();
   }

   std::string Generate(std::string OpName) override
   {
      OpName = "op_" + OpName;
      if (fShapeX.empty()) {
         throw std::runtime_error("TMVA::SOFIE LayerNormalization operator " + OpName +
                                  " called to generate without beging initialized first.");
      }
      if (fShapeX.size() > 5) {
         throw std::runtime_error("TMVA::SOFIE LayerNormalization operator not "
                                  "implemented for input tensor of size > 5.");
      }

      std::stringstream out;

      out << SP << "// Operator " << OpName << "\n";

      // Loop over all the normalized axes i.e. [axis, ..., size)
      out << SP << "std::vector<size_t> " << OpName << "_InputShape ({";
      for (size_t i = 0; i < fSize; i++) {
         out << fShapeX[i];
         if (i + 1 < fSize) {
            out << ",";
         }
      }
      out << "});\n";
      std::string inputShape = OpName + "_InputShape";

      auto strides = UTILITY::ComputeStrideFromShape(fShapeX);
      std::string InputIndex = "axis_0 * " + std::to_string(strides[0]);
      for (size_t i = 1; i < fSize; i++) {
         InputIndex += " + axis_" + std::to_string(i) + " * " + std::to_string(strides[i]);
      }

      auto axesStrides = UTILITY::ComputeStrideFromShape(fAxesShape);
      std::string axesIndex = "axis_" + std::to_string(0) + " * " + std::to_string(axesStrides[0]);
      for (size_t i = 1; i < fAxis; i++) {
         axesIndex += " + axis_" + std::to_string(i) + " * " + std::to_string(axesStrides[i]);
      }

      auto normalizedStrides = UTILITY::ComputeStrideFromShape(fNormalizedShape);
      std::string normalizedIndex = "axis_" + std::to_string(fAxis) + " * " + std::to_string(normalizedStrides[0]);
      for (size_t i = fAxis + 1; i < fSize; i++) {
         normalizedIndex += " + axis_" + std::to_string(i) + " * " + std::to_string(normalizedStrides[i - fAxis]);
      }

      if (!fNCastedX.empty()) {
         // Cast X to float
         out << SP << "for (size_t i = 0; i < " << fLength << "; i++) {\n";
         out << SP << SP << "tensor_" << fNCastedX << "[i] = " << "static_cast<float>(tensor_" << fNX;
         out << "[i]);\n";
         out << SP << "}\n";
      }

      out << SP << "// Compute the mean\n";
      // Loop over the normalized dimensions
      for (size_t i = 0; i < fAxis; i++) {
         std::string iIdx = "axis_" + std::to_string(i);
         out << SP << "for (size_t " << iIdx << " = 0; " << iIdx << " < " << inputShape;
         out << "[" << i << "]; " << iIdx << "++) {\n";
      }
      out << SP << SP << fType << " sum = 0.;\n";
      // loop over all the dims in [0, fAxis)
      for (size_t j = fAxis; j < fSize; j++) {
         std::string jIdx = "axis_" + std::to_string(j);
         out << SP << SP << "for (size_t " << jIdx << " = 0; " << jIdx << " < " << inputShape;
         out << "[" << j << "]; " << jIdx << "++) {\n";
      }
      out << SP << SP << SP << "sum += tensor_" << fNX << "[" << InputIndex << "];\n";
      for (size_t j = fAxis; j < fSize; j++) {
         out << SP << SP << "}\n";
      }
      out << SP << SP << "tensor_" << fNMean << "[" << axesIndex << "] = sum / " << fType << "(";
      out << fNormalizedLength << ");\n";
      for (size_t i = fAxis; i < fSize; i++) {
         out << SP << "}\n";
      }

      out << SP << "// Compute the inverse Standard Deviation\n";
      // Loop over the normalized dimensions
      for (size_t i = 0; i < fAxis; i++) {
         std::string iIdx = "axis_" + std::to_string(i);
         out << SP << "for (size_t " << iIdx << " = 0; " << iIdx << " < " << inputShape;
         out << "[" << i << "]; " << iIdx << "++){\n";
      }
      // Set sum = 0
      out << SP << SP << fType << " sum = 0.;\n";
      // loop over all the dims in [0, fAxis)
      for (size_t j = fAxis; j < fSize; j++) {
         std::string jIdx = "axis_" + std::to_string(j);
         out << SP << SP << "for (size_t " << jIdx << " = 0; " << jIdx << " < " << inputShape;
         out << "[" << j << "]; " << jIdx << "++){\n";
      }
      out << SP << SP << SP << "sum += std::pow(tensor_" << fNX << "[" << InputIndex << "] - tensor_";
      out << fNMean << "[" << axesIndex << "], static_cast<float>(2));\n";
      for (size_t j = fAxis; j < fSize; j++) {
         out << SP << SP << "}\n";
      }
      out << SP << SP << "tensor_" << fNInvStdDev << "[" << axesIndex << "] = 1 / (std::sqrt(";
      out << "sum / " << fType << "(" << fNormalizedLength << ")) + static_cast<float>(" << fAttrEpsilon << "));\n";
      for (size_t i = 0; i < fAxis; i++) {
         out << SP << "}\n";
      }

      if (!fNCastedX.empty()) {
         out << "// NormalizedX = InvStdDev * (CastedX - Mean)\n";
         for (size_t i = 0; i < fAxis; i++) {
            std::string iIdx = "axis_" + std::to_string(i);
            out << SP << "for (size_t " << iIdx << " = 0; " << iIdx << " < " << inputShape;
            out << "[" << i << "]; " << iIdx << "++){\n";
         }
         for (size_t j = fAxis; j < fSize; j++) {
            std::string jIdx = "axis_" + std::to_string(j);
            out << SP << SP << "for (size_t " << jIdx << " = 0; " << jIdx << " < " << inputShape;
            out << "[" << j << "]; " << jIdx << "++){\n";
         }
         out << SP << SP << SP << "tensor_" << fNNormalizedX << "[" << InputIndex << "] = tensor_";
         out << fNInvStdDev << "[" << axesIndex << "] * (tensor_" << fNCastedX << "[" << InputIndex;
         out << "] - tensor_" << fNMean << "[" << axesIndex << "])\n";
         for (size_t j = fAxis; j < fSize; j++) {
            out << SP << SP << "}\n";
         }
         for (size_t i = fAxis; i < fSize; i++) {
            out << SP << "}\n";
         }
         out << "// Y = Scale o NormalizedX";
         for (size_t i = 0; i < fAxis; i++) {
            std::string iIdx = "axis_" + std::to_string(i);
            out << SP << "for (size_t " << iIdx << " = 0; " << iIdx << " < " << inputShape;
            out << "[" << i << "]; " << iIdx << "++){\n";
         }
         for (size_t j = fAxis; j < fSize; j++) {
            std::string jIdx = "axis_" + std::to_string(j);
            out << SP << SP << "for (size_t " << jIdx << " = 0; " << jIdx << " < " << inputShape;
            out << "[" << j << "]; " << jIdx << "++){\n";
         }
         out << SP << SP << SP << "tensor_" << fNY << "[" << InputIndex << "] = tensor_" << fNScale;
         out << "[" << axesIndex << "] * static_cast<" << fType << ">(tensor_" << fNCastedX << "[" << InputIndex;
         out << "]);\n";
         for (size_t j = fAxis; j < fSize; j++) {
            out << SP << SP << "}\n";
         }
         for (size_t i = fAxis; i < fSize; i++) {
            out << SP << "}\n";
         }
      } else {
         out << SP << "// Y = Scale o InvStdDev (X - Mean)\n";
         for (size_t i = 0; i < fAxis; i++) {
            std::string iIdx = "axis_" + std::to_string(i);
            out << SP << "for (size_t " << iIdx << " = 0; " << iIdx << " < " << inputShape;
            out << "[" << i << "]; " << iIdx << "++){\n";
         }
         for (size_t j = fAxis; j < fSize; j++) {
            std::string jIdx = "axis_" + std::to_string(j);
            out << SP << SP << "for (size_t " << jIdx << " = 0; " << jIdx << " < " << inputShape;
            out << "[" << j << "]; " << jIdx << "++){\n";
         }
         out << SP << SP << SP << "tensor_" << fNY << "[" << InputIndex << "] = tensor_" << fNScale;
         out << "[" << normalizedIndex << "] * tensor_" << fNInvStdDev << "[" << axesIndex;
         out << "] * (tensor_" << fNX << "[" << InputIndex << "] - tensor_" << fNMean << "[";
         out << axesIndex << "]);\n";
         for (size_t j = fAxis; j < fSize; j++) {
            out << SP << SP << "}\n";
         }
         for (size_t i = fAxis; i < fSize; i++) {
            out << SP << "}\n";
         }
      }

      if (!fNB.empty()) {
         std::string Bias = "tensor_" + (fNBroadcastedB.empty() ? fNB : fNBroadcastedB);
         out << SP << "// Add the bias to Y\n";
         out << SP << "int " << OpName << "_n = " << fLength << ";\n";
         out << SP << "float " << OpName << "_alpha = 1.;\n";
         out << SP << "int " << OpName << "_inc = 1;\n";
         out << SP << "BLAS::saxpy_(&" << OpName << "_n, &" << OpName << "_alpha, " << Bias << ", &";
         out << OpName << "_inc, " << "tensor_" << fNY << ", &" << OpName << "_inc);\n";
      }

      return out.str();
   }

   std::string GenerateGPU(std::string OpName) override {
      OpName = "op_" + OpName;
      if (fShapeX.empty()) {
         throw std::runtime_error("TMVA::SOFIE LayerNormalization operator " + OpName +
                                  " called to generate without beging initialized first.");
      }
      if (fShapeX.size() > 5) {
         throw std::runtime_error("TMVA::SOFIE LayerNormalization operator not "
                                  "implemented for input tensor of size > 5.");
      }

      std::stringstream out;

      out << "\n" << SP*3 << "// Operator " << OpName << "\n";
      out << SP*3 << "std::array<size_t, " << fSize << "> " << OpName << "_InputShape ({";
      for (size_t i=0; i < fSize; i++) {
         out << fShapeX[i];
         if (i + 1 < fSize) {
            out << ", ";
         }
      }

      out << "});\n";
      std::string inputShape = OpName + "_InputShape";

      auto strides = UTILITY::ComputeStrideFromShape(fShapeX);
      std::string InputIndex = "axis_0 * " + std::to_string(strides[0]);
      for (size_t i = 1; i < fSize; i++) {
         InputIndex += " + axis_" + std::to_string(i) + " * " + std::to_string(strides[i]);
      }

      auto axesStrides = UTILITY::ComputeStrideFromShape(fAxesShape);
      std::string axesIndex = "axis_" + std::to_string(0) + " * " + std::to_string(axesStrides[0]);
      for (size_t i = 1; i < fAxis; i++) {
         axesIndex += " + axis_" + std::to_string(i) + " * " + std::to_string(axesStrides[i]);
      }

      auto normalizedStrides = UTILITY::ComputeStrideFromShape(fNormalizedShape);
      std::string normalizedIndex = "axis_" + std::to_string(fAxis) + " * " + std::to_string(normalizedStrides[0]);
      for (size_t i = fAxis + 1; i < fSize; i++) {
         normalizedIndex += " + axis_" + std::to_string(i) + " * " + std::to_string(normalizedStrides[i - fAxis]);
      }

      // cast X to float
      if (!fNCastedX.empty()) {
         out << SP*3 << "q.submit([&](cl::sycl::handler& cgh){\n";
         out << SP*4 << "auto acc_tensor_" << fNX << " = cl::sycl::accessor{buf_tensor_" << fNX;
         out << ", cgh, cl::sycl::read_only};\n";
         out << SP*4 << "auto acc_tensor_" << fNCastedX << " = cl::sycl::accessor{buf_tensor_" << fNCastedX;
         out << ", cgh, cl::sycl::write_only, cl::sycl::no_init};\n";
         out << SP*4 << "cgh.parallel_for<class " << OpName << "_0>(cl::sycl::range<1>(" << fLength;
         out << "), [=](cl::sycl::id<1> id){\n";
         out << SP*5 << "acc_tensor_" << fNCastedX << "[id] = static_cast<float>(acc_tensor_" << fNX << "[id]);\n";
         out << SP*4 << "});\n";
         out << SP*3 << "});\n";
      }

      /*
      out << SP*3 << "// Compute the mean\n";
      out << SP*3 << "size_t num_work_items_0 = 1";
      for (size_t i=0; i<fAxis; i++) {
         out << " * " << inputShape << "[" << i << "]";
      }
      out << ";\n";

      out << SP*3 << "size_t num_work_items_1 = num_work_items_0";
      for (size_t i=fAxis; i<fSize; i++) {
         out << " * " << inputShape << "[" << i << "]";
      }
      out << ";\n";
      */

      out << SP*3 << "auto num_work_items_0 = cl::sycl::range<" << fAxis << ">{";
      for (size_t i=0; i<fAxis; i++) {
         out << inputShape << "[" << i << "], ";
      }
      out.str().pop_back();
      out << "};\n";

      out << SP*3 << "auto num_work_items_1 = cl::sycl::range<" << fSize << ">{";
      for (size_t i=0; i<fSize; i++) {
         out << inputShape << "[" << i << "], ";
      }
      out.str().pop_back();
      out << "};\n";

      out << SP*3 << "q.submit([&](cl::sycl::handler& cgh){\n";
      out << SP*4 << "auto acc_tensor_" << fNX << " = cl::sycl::accessor{buf_tensor_" << fNX;
      out << ", cgh, cl::sycl::read_only};\n";
      out << SP*4 << "auto acc_tensor_" << fNMean << " = cl::sycl::accessor{buf_tensor_" << fNMean;
      out << ", cgh, cl::sycl::write_only, cl::sycl::no_init};\n";
      
      out << SP*4 << "cgh.parallel_for<class " << OpName << "_1>(cl::sycl::range<";
      out << fAxis << ">(num_work_items_0), [=](cl::sycl::id<" << fAxis << "> id){\n";
      out << SP*5 << "float sum = 0.0;\n";

      for (size_t i=0; i<fAxis; i++) {
         out << SP*5 << "size_t axis_" + std::to_string(i) "= num_work_items_0[" << std::to::string(i) << "];\n";
      }

      for (size_t j = fAxis; j < fSize; j++) {
         std::string jIdx = "axis_" + std::to_string(j);
         out << SP*(5 + (j - fAxis)) << "for (size_t " << jIdx << " = 0; " << jIdx << " < " << inputShape;
         out << "[" << j << "]; " << jIdx << "++) {\n";
      }

      out << SP*(5 + (fSize - fAxis + 2)) << "sum += acc_tensor_" << fNX << "[" << InputIndex << "];\n";

      for (size_t j = fSize-1; j >= fAxis; j--) {
         out << SP*(5 + (j - fAxis)) << "}\n";
      }

      out << SP*5 << "acc_tensor_" << fNMean << "[id] = sum / " << fType << "(" << fNormalizedLength << ");\n";
      out << SP*4 << "});\n";
      out << SP*3 << "});\n\n";

      out << SP*3 << "// Compute the Inverse Standard Deviation\n";

      out << SP*3 << "q.submit([&](cl::sycl::handler& cgh){\n";
      out << SP*4 << "auto acc_tensor_" << fNX << " = cl::sycl::accessor{buf_tensor_" << fNX;
      out << ", cgh, cl::sycl::read_only};\n";
      out << SP*4 << "auto acc_tensor_" << fNMean << " = cl::sycl::accessor{buf_tensor_" << fNMean;
      out << ", cgh, cl::sycl::read_only};\n";
      out << SP*4 << "auto acc_tensor_" << fNInvStdDev << "= cl::sycl::accessor{buf_tensor_" << fNInvStdDev;
      out << ", cgh, cl::sycl::write_only, cl::sycl::no_init};\n";
      
      out << SP*4 << "cgh.parallel_for<class " << OpName << "_2>(cl::sycl::range<";
      out << fAxis << ">(num_work_items_0), [=](cl::sycl::id<" << fAxis << "1> id){\n";

      out << SP*5 << fType << " sum = 0.0;\n";

      for (size_t i=0; i<fAxis; i++) {
         out << SP*5 << "size_t axis_" + std::to_string(i) "= num_work_items_0[" << std::to_string(i) << "];\n";
      }

      for (size_t j = fAxis; j < fSize; j++) {
         std::string jIdx = "axis_" + std::to_string(j);
         out << SP*(5 + (j - fAxis)) << "for (size_t " << jIdx << " = 0; " << jIdx << " < " << inputShape;
         out << "[" << j << "]; " << jIdx << "++) {\n";
      }

      out << SP*(5 + (fSize - fAxis + 2)) << "sum += cl::sycl::pow(acc_tensor_" << fNX << "[" << InputIndex << "] - acc_tensor_";
      out << fNMean << "[" << axesIndex << "], static_cast<float>(2));\n";

      for (size_t j = fSize-1; j >= fAxis; j--) {
         out << SP*(5 + (j - fAxis)) << "}\n";
      }
      
      out << SP*5 << "acc_tensor_" << fNInvStdDev << "[id] = cl::sycl::native::recip(cl::sycl::sqrt(sum / " << fType << "(" << fNormalizedLength << ")) + static_cast<float>(" << fAttrEpsilon << "));\n";  
      out << SP*4 << "});\n";
      out << SP*3 << "});\n\n";

      if (!fNCastedX.empty()) {


         out << SP*3 << "q.submit([&](cl::sycl::handler& cgh){\n";
         out << SP*4 << "auto acc_tensor_" << fNScale << " = cl::sycl::accessor{buf_tensor_" << fNScale;
         out << ", cgh, cl::sycl::read_only};\n";
         out << SP*4 << "auto acc_tensor_" << fNInvStdDev << " = cl::sycl::accessor{buf_tensor_" << fNInvStdDev;
         out << ", cgh, cl::sycl::read_only};\n";
         out << SP*4 << "auto acc_tensor_" << fNX << " = cl::sycl::accessor{buf_tensor_" << fNX;
         out << ", cgh, cl::sycl::read_only};\n";
         out << SP*4 << "auto acc_tensor_" << fNMean << " = cl::sycl::accessor{buf_tensor_" << fNMean;
         out << ", cgh, cl::sycl::read_only};\n";
         out << SP*4 << "auto acc_tensor_" << fNNormalizedX << " cl::sycl::accessor{buf_tensor_" << fNNormalizedX;
         out << ", cgh, cl::sycl::write_only, cl::sycl::no_init};\n";
         out << SP*4 << "auto acc_tensor_" << fNY << " = cl::sycl::accessor{buf_tensor_" << fNY;
         out << ", cgh, cl::sycl::write_only, cl::sycl::no_init};\n";

         out << SP*4 << "cgh.parallel_for<class " << OpName << "_3>(cl::sycl::range<";
         out << fSize << ">(num_work_items_1), [=](cl::sycl::id<" << fSize << ">id){\n";
      
         for (size_t i=0; i<fSize; i++) {
            out << SP*5 << "size_t axis_" + std::to_string(i) "= num_work_items_1[" << std::to_string(i) << "];\n";
         }

         out << "\n" << SP*5 << "// NormalizedX = InvStdDev * (CastedX - Mean)\n";
         out << SP*5 << "acc_tensor_" << fNNormalizedX << "[" << InputIndex << "] = acc_tensor_";
         out << fNInvStdDev << "[" << axesIndex << "] * (acc_tensor_" << fNCastedX << "[" << InputIndex;
         out << "] - acc_tensor_" << fNMean << "[" << axesIndex << "]);\n";

         out << "\n" << SP*5 << "// Y = Scale o NormalizedX\n";
         out << SP*5 << "acc_tensor_" << fNY << "[" << InputIndex << "] = acc_tensor_" << fNScale;
         out << "[" << axesIndex << "] * static_cast<" << fType << ">(acc_tensor_" << fNCastedX << "[" << InputIndex;
         out << "]);\n";

         out << SP*4 << "});\n";
         out << SP*3 << "});\n";
      }
      else {

         out << SP*3 << "q.submit([&](cl::sycl::handler& cgh){\n";
         out << SP*4 << "auto acc_tensor_" << fNScale << " = cl::sycl::accessor{buf_tensor_" << fNScale;
         out << ", cgh, cl::sycl::read_only};\n";
         out << SP*4 << "auto acc_tensor_" << fNInvStdDev << " = cl::sycl::accessor{buf_tensor_" << fNInvStdDev;
         out << ", cgh, cl::sycl::read_only};\n";
         out << SP*4 << "auto acc_tensor_" << fNX << " = cl::sycl::accessor{buf_tensor_" << fNX;
         out << ", cgh, cl::sycl::read_only};\n";
         out << SP*4 << "auto acc_tensor_" << fNMean << " = cl::sycl::accessor{buf_tensor_" << fNMean;
         out << ", cgh, cl::sycl::read_only};\n";
         out << SP*4 << "auto acc_tensor_" << fNY << " = cl::sycl::accessor{buf_tensor_" << fNY;
         out << ", cgh, cl::sycl::write_only, cl::sycl::no_init};\n";

         out << "\n" << SP*4 << "// Y = Scale o InvStdDev (X-Mean)\n";
         out << SP*4 << "cgh.parallel_for<class " << OpName << "_3>(cl::sycl::range<";
         out << fSize << ">(num_work_items_1), [=](cl::sycl::id<" << fSize << ">id){\n";

         for (size_t i=0; i<fSize; i++) {
            out << SP*5 << "size_t axis_" + std::to_string(i) "= num_work_items_1[" << std::to_string(i) << "];\n";
         }

         out << SP*5 << "acc_tensor_" << fNY << "[" << InputIndex << "] = acc_tensor_" << fNScale;
         out << "[" << normalizedIndex << "] * acc_tensor_" << fNInvStdDev << "[" << axesIndex;
         out << "] * (acc_tensor_" << fNX << "[" << InputIndex << "] - acc_tensor_" << fNMean << "[";
         out << axesIndex << "]);\n";  

         out << SP*4 << "});\n";
         out << SP*3 << "});\n";
      }

      if (!fNB.empty()) {
         std::string Bias = "buf_tensor_" + ((fNBroadcastedB.empty()) ? fNB : fNBroadcastedB);
         out << "\n" << SP*3 << "// Add the bias to Y\n";
         out << SP*3 << "int " << OpName << "_n = " << fLength << ";\n";
         out << SP*3 << "float " << OpName << "_alpha = 1.;\n";
         out << SP*3 << "int " << OpName << "_inc = 1;\n";
         out << SP*3 << "oneapi::mkl::blas::axpy(q, " << OpName << "_n, " << OpName << "_alpha, ";
         out << Bias << ", " << OpName << "_inc, buf_tensor_" << fNY << ", " << OpName << "_inc);\n";
      }

      return out.str();
   }

   std::vector<std::string> GetBlasRoutines() override { return { std::string("Axpy") }; }

   std::vector<std::string> GetStdLibs() override { return { std::string("cmath") }; }
};

} // namespace SOFIE
} // namespace Experimental
} // namespace TMVA

#endif
