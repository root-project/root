#ifndef TMVA_SOFIE_ROPERATOR_CONV
#define TMVA_SOFIE_ROPERATOR_CONV

#include "TMVA/SOFIE_common.hxx"
#include "TMVA/ROperator.hxx"
#include "TMVA/RModel.hxx"

#include <memory>
#include <sstream>
#include <algorithm>
#include <stdexcept>
#include <vector>

namespace TMVA {
namespace Experimental {
namespace SOFIE {

template<typename T>
class ROperator_Conv final : public ROperator
{
private:
   std::string fAttrAutopad;
   std::vector<size_t> fAttrDilations;
   size_t fAttrGroup;
   std::vector<size_t> fAttrKernelShape;
   std::vector<size_t> fAttrPads;
   std::vector<size_t> fAttrStrides;

   std::string fNX;
   std::string fNW;
   std::string fNB;
   std::string fNY;

   std::vector<size_t> fShapeX;
   std::vector<size_t> fShapeW;
   std::vector<size_t> fShapeB;
   std::vector<size_t> fShapeY;

   std::string fType;

public:

   ROperator_Conv() {}

   ROperator_Conv(std::string autopad, std::vector<size_t> dilations,
      size_t group, std::vector<size_t> kernelShape, std::vector<size_t> pads,
      std::vector<size_t> strides, std::string nameX, std::string nameW,
      std::string nameB, std::string nameY):
      fAttrAutopad(autopad), fAttrDilations(dilations), fAttrGroup(group), fAttrKernelShape(kernelShape),
      fAttrPads(pads), fAttrStrides(strides),
      fNX(UTILITY::Clean_name(nameX)), fNW(UTILITY::Clean_name(nameW)),
      fNB(UTILITY::Clean_name(nameB)), fNY(UTILITY::Clean_name(nameY))
   {
      if(std::is_same<T, float>::value) {
         fType = "float";
      } else {
         throw
            std::runtime_error("TMVA SOFIE Encountered unsupported type parsing a Conv operator");
      }
   }

   ROperator_Conv(std::string autopad, std::vector<size_t> dilations,
      size_t group, std::vector<size_t> kernelShape, std::vector<size_t> pads,
      std::vector<size_t> strides, std::string nameX, std::string nameW,
      std::string nameY):
      fAttrAutopad(autopad), fAttrDilations(dilations), fAttrGroup(group), fAttrKernelShape(kernelShape),
      fAttrPads(pads), fAttrStrides(strides),
      fNX(UTILITY::Clean_name(nameX)), fNW(UTILITY::Clean_name(nameW)), fNY(UTILITY::Clean_name(nameY))
   {
      if(std::is_same<T, float>::value) {
         fType = "float";
      } else {
         throw
            std::runtime_error("TMVA SOFIE Encountered unsupported type parsing a Conv operator");
      }
   }

   std::vector<ETensorType> TypeInference(std::vector<ETensorType> input) {
      ETensorType out = input[0];
      return {out};
   }

   std::vector<std::vector<size_t>> ShapeInference(std::vector<std::vector<size_t>> input) {
      if (input.size() > 3 ) {
         throw
            std::runtime_error("TMVA SOFIE Conv Op Shape inference need 2 or 3 input tensors");
      }
      for(size_t i = 0; i < input.size(); i++) {
         if (input[i].size() != 4) {
            throw
               std::runtime_error("TMVA SOFIE Conv Op Shape inference only accept tensor with 4 dimensions");
         }
      }

      if (fAttrGroup == 0) {
         fAttrGroup = input[0][1] / input[1][1];
      }

      size_t kHeight = ((fAttrKernelShape.empty())? input[1][2] : fAttrKernelShape[0]);
      size_t kWidth = ((fAttrKernelShape.empty())? input[1][3] : fAttrKernelShape[1]);

      if (fAttrDilations.empty()) {
         fAttrDilations = {1, 1};
      }
      // Shape of the kernel
      fAttrKernelShape = {kHeight + (fAttrDilations[0] - 1) * (kHeight - 1), kWidth + (fAttrDilations[1] - 1) * (kWidth - 1)};

      if (fAttrAutopad == "NOTSET") {
         if (fAttrPads.empty()) {
            fAttrPads = {1, 1, 1, 1};
         }
      } else if (fAttrAutopad == "SAME_UPPER" || fAttrAutopad == "SAME_LOWER") {
         fAttrPads = {fAttrKernelShape[0] / 2, fAttrKernelShape[1] / 2, fAttrKernelShape[0] / 2, fAttrKernelShape[1] / 2};
         if (fAttrKernelShape[0] % 2 == 1) {
            (fAttrAutopad == "SAME_UPPER") ? fAttrPads[0]++ : fAttrPads[2]++;
         }
         if (fAttrKernelShape[1] % 2 == 1) {
            (fAttrAutopad == "SAME_UPPER") ? fAttrPads[1]++ : fAttrPads[3]++;
         }
      } else if (fAttrAutopad != "VALID") {
         throw
            std::runtime_error("TMVA SOFIE Conv Op invalid fAutopad");
      }

      if (fAttrStrides.empty()) {
         fAttrStrides = {1, 1};
      }

      size_t outputHeight =
          (input[0][2] + fAttrPads[0] + fAttrPads[2] - fAttrKernelShape[0] + fAttrStrides[0]) /
          fAttrStrides[0];
      size_t outputWidth =
          (input[0][3] + fAttrPads[1] + fAttrPads[3] - fAttrKernelShape[1] + fAttrStrides[1]) /
          fAttrStrides[1];

      std::vector<std::vector<size_t>> ret({{input[1][0], input[0][0], outputHeight, outputWidth}});
      return ret;
   }

   void Initialize(RModel& model) {
      if (!model.CheckIfTensorAlreadyExist(fNX)) {
         throw
            std::runtime_error("TMVA SOFIE Conv op Input Tensor " + fNX + " is not found in model");
      }
      fShapeX = model.GetTensorShape(fNX);
      if (fShapeX.size() != 4) {
         throw
            std::runtime_error("TMVA SOFIE Conv Op input tensor" + fNX + " is not of 4 dimensions");
      }
      if (!model.CheckIfTensorAlreadyExist(fNW)) {
         throw
            std::runtime_error("TMVA SOFIE Conv op Input Tensor " + fNW + " is not found in model");
      }
      fShapeW = model.GetTensorShape(fNW);
      if (fShapeW.size() != 4) {
         throw
            std::runtime_error("TMVA SOFIE Conv Op input tensor" + fNW + " is not of 4 dimensions");
      }
      fShapeY = ShapeInference({fShapeX, fShapeW})[0];
      model.AddIntermediateTensor(fNY, model.GetTensorType(fNX), fShapeY);
      if (fNB != "") {
         if (!model.CheckIfTensorAlreadyExist(fNB)) {
            throw
               std::runtime_error("TMVA SOFIE Conv op Input Tensor " + fNB + " is not found in model");
         }
         fShapeB = model.GetTensorShape(fNB);
         bool broadcast_needed = (fShapeB.size() != fShapeY.size());
         if (broadcast_needed) {
            auto original_data = model.GetInitializedTensorData(fNB);
            if (fType == "float") {
               std::shared_ptr<void> new_data_ptr(UTILITY::Unidirectional_broadcast<float>(
                  static_cast<float*>(original_data.get()), fShapeB, fShapeY), std::default_delete<float[]>());
               model.UpdateInitializedTensor(fNB, model.GetTensorType(fNB), fShapeY, new_data_ptr);
               fShapeB = model.GetTensorShape(fNB);
            }
         }
      }
      
   }

   std::string Generate(std::string OpName) {
      OpName = "op_" + OpName;

      if (fShapeX.empty() || fShapeW.empty() || (fNB != "" && fShapeB.empty()) || fShapeY.empty()) {
         throw
            std::runtime_error("TMVA SOFIE Conv Op called to Generate without being initialized first");
      }

      std::stringstream out;

      if (fType == "float") {
         out << "\t" << "float " << OpName << "_xpad[" << fShapeX[0] * fShapeX[1] * (fShapeX[2] + fAttrPads[0] + fAttrPads[2])
          * (fShapeX[3] + fAttrPads[1] + fAttrPads[3]) << "] = {0};\n";
      }
      // Padding the input with zeros
      if (fShapeX[0] == 1) {
         out << "\t" << "for (size_t c = 0; c < " << fShapeX[1] << "; c++) {\n";
         out << "\t" << "\t" << "for (size_t h = 0; h < " << fShapeX[2] << "; h++) {\n";
         out << "\t" << "\t" << "\t" << "size_t xpad_offset = c * "
             << (fShapeX[2] + fAttrPads[0] + fAttrPads[2]) * (fShapeX[3] + fAttrPads[1] + fAttrPads[3]) << " + (h + " << fAttrPads[0]
             << ") * " << (fShapeX[3] + fAttrPads[1] + fAttrPads[3]) << " + " << fAttrPads[1] << ";\n";
         out << "\t" << "\t" << "\t" << "size_t x_offset = c * " << fShapeX[2] * fShapeX[3] << " + h * " << fShapeX[3] << ";\n";
         out << "\t" << "\t" << "\t" << "std::copy(tensor_" << fNX << " + x_offset, tensor_" << fNX
             << " + x_offset + " << fShapeX[3] << ", " << OpName << "_xpad + xpad_offset);\n";
         out << "\t" << "\t" << "}\n";
         out << "\t" << "}\n";
      } else {
         out << "\t" << "for (size_t n = 0; n < " << fShapeX[0] << "; n++) {\n";
         out << "\t" << "\t" << "for (size_t c = 0; c < " << fShapeX[1] << "; c++) {\n";
         out << "\t" << "\t" << "\t" << "for (size_t h = 0; h < " << fShapeX[2] << "; h++) {\n";
         out << "\t" << "\t" << "\t" << "\t" << "size_t xpad_offset = n * "
             << fShapeX[1] * (fShapeX[2] + fAttrPads[0] + fAttrPads[2]) * (fShapeX[3] + fAttrPads[1] + fAttrPads[3])
             << " + c * " << (fShapeX[2] + fAttrPads[0] + fAttrPads[2]) * (fShapeX[3] + fAttrPads[1] + fAttrPads[3])
             << " + (h + " << fAttrPads[0] <<  ") * " << (fShapeX[3] + fAttrPads[1] + fAttrPads[3]) << " + "
             << fAttrPads[1] << ";\n";
         out << "\t" << "\t" << "\t" << "\t" << "size_t x_offset = n * " << fShapeX[1] * fShapeX[2] * fShapeX[3] << " + c * "
             << fShapeX[2] * fShapeX[3] << " + h * " << fShapeX[3] << ";\n";
         out << "\t" << "\t" << "\t" << "std::copy(tensor_" << fNX << " + x_offset, tensor_" << fNX
             << " + x_offset + " << fShapeX[3] << ", " << OpName << "_xpad + xpad_offset);\n";
         out << "\t" << "\t" << "\t" << "}\n";
         out << "\t" << "\t" << "}\n";
         out << "\t" << "}\n";
      }

      // convolution kernels
      if (fType == "float") {
         out << "\t" << "float " << OpName << "_f[" << fShapeW[0] * fShapeW[1] * fAttrKernelShape[0] * fAttrKernelShape[1] << "] = {0};\n";
      }
      // vectorize the (dilated)convolution kernels into a matrix
      out << "\t" << "for (std::size_t k = 0; k < " << fShapeW[0] << "; k++) {\n";
      out << "\t" << "\t" << "for (std::size_t d = 0; d < " << fShapeW[1] << "; d++) {\n";
      out << "\t" << "\t" << "\t" << "for (std::size_t h = 0; h < " << fShapeW[2] << "; h++) {\n";
      out << "\t" << "\t" << "\t" << "\t" << "for (std::size_t w = 0; w < " << fShapeW[3] << "; w++) {\n";
      out << "\t" << "\t" << "\t" << "\t" << "\t" << OpName <<  "_f[k + " << "(d * "
          << fAttrKernelShape[0] * fAttrKernelShape[1] << " + h * " << fAttrDilations[0] * fAttrKernelShape[1]
          << " + w * " << fAttrDilations[1] << ") * " << fShapeW[0] << "] = tensor_" << fNW << "[k * "
          << fShapeW[1] * fShapeW[2] * fShapeW[3] << " + d * " << fShapeW[2] * fShapeW[3] << " + h * "
          << fShapeW[3] << " + w ];\n";
      out << "\t" << "\t" << "\t" << "\t" << "}\n";
      out << "\t" << "\t" << "\t" << "}\n";
      out << "\t" << "\t" << "}\n";
      out << "\t" << "}\n";

      if (fAttrGroup == 1) {
         if (fType == "float") {
         out << "\t" << "float " << OpName << "_xcol[" << fShapeX[1] * fAttrKernelShape[0] * fAttrKernelShape[1]
             * fShapeX[0] * fShapeY[2] * fShapeY[3] << "] = {0};\n";
         }
         // Unroll the input tensor
         out << "\t" << "size_t " << OpName << "_index = 0;\n";
         out << "\t" << "for (size_t n = 0; n < " << fShapeX[0] << "; n++) {\n";
         out << "\t" << "\t" << "for (size_t c = 0; c < " << fShapeW[1] << "; c++) {\n";
         out << "\t" << "\t" << "\t" << "for (size_t h = 0; h < " << fShapeX[2] + fAttrPads[0] + fAttrPads[2] - fAttrKernelShape[0] + 1
             << "; h += " << fAttrStrides[0] << ") {\n";
         out << "\t" << "\t" << "\t" << "\t" << "for (size_t w = 0; w < " << fShapeX[3] + fAttrPads[1] + fAttrPads[3] - fAttrKernelShape[1] + 1
             << ";w += " << fAttrStrides[1] << ") {\n";
         out << "\t" << "\t" << "\t" << "\t" << "\t" << "for (size_t x = 0; x < " << fAttrKernelShape[0] << "; x++) {\n";
         out << "\t" << "\t" << "\t" << "\t" << "\t" << "size_t offset = n * "
             << fShapeX[1] * (fShapeX[2] + fAttrPads[0] + fAttrPads[2]) * (fShapeX[3] + fAttrPads[1] + fAttrPads[3])
             << " + c * " << (fShapeX[2] + fAttrPads[0] + fAttrPads[2]) * (fShapeX[3] + fAttrPads[1] + fAttrPads[3])
             << "+ (h + x) * " << (fShapeX[3] + fAttrPads[1] + fAttrPads[3]) << " + w;\n";
         out << "\t" << "\t" << "\t" << "\t" << "\t" << "std::copy(" << OpName << "_xpad + offset, " << OpName
             << "_xpad + offset + " << fAttrKernelShape[1] << ", " << OpName << "_xcol + " << OpName << "_index);\n";
         out << "\t" << "\t" << "\t" << "\t" << "\t" << OpName << "_index += " << fAttrKernelShape[1] << ";\n";
         out << "\t" << "\t" << "\t" << "\t" << "\t" << "}\n";
         out << "\t" << "\t" << "\t" << "\t" << "}\n";
         out << "\t" << "\t" << "\t" << "}\n";
         out << "\t" << "\t" << "}\n";
         out << "\t" << "}\n";

         out << "\t" << "char " << OpName << "_transA = 'N';\n";
         out << "\t" << "char " << OpName << "_transB = 'N';\n";
         out << "\t" << "int " << OpName << "_m = " << fShapeW[0] << ";\n";
         out << "\t" << "int " << OpName << "_n = " << fShapeX[0] * fShapeY[2] * fShapeY[3] << ";\n";
         out << "\t" << "int " << OpName << "_k = " << fShapeW[1] * fAttrKernelShape[0] * fAttrKernelShape[1] << ";\n";
         out << "\t" << "float " << OpName << "_alpha = 1.0;\n";
         out << "\t" << "float " << OpName << "_beta = 0.0;\n";
         out << "\t" << "BLAS::sgemm_(&" << OpName << "_transA, &" << OpName << "_transB, &" << OpName << "_m, &"
             << OpName << "_n, &" << OpName << "_k, &" << OpName << "_alpha, " << OpName << "_f, &" << OpName << "_m,\n";
         out << "\t" << "\t" << OpName << "_xcol, &" << OpName << "_k, &" << OpName << "_beta, tensor_" << fNY
             << ", &" << OpName << "_m);\n";
      } else {
         if (fType == "float") {
         out << "\t" << "float " << OpName << "_xcol[" << fShapeX[1] * fAttrKernelShape[0] * fAttrKernelShape[1]
             * fShapeX[0] * fShapeY[2] * fShapeY[3] << "] = {0};\n";
         }
         // Unroll the input tensor
         out << "\t" << "for (size_t g = 0; g < " << fAttrGroup << "; g++) {\n";
         out << "\t" << "\t" << "size_t index = g * " << fShapeW[1] * fAttrKernelShape[0] * fAttrKernelShape[1] << ";\n";
         out << "\t" << "\t" << "for (size_t n = 0; n < " << fShapeX[0] << "; n++) {\n";
         out << "\t" << "\t" << "\t" << "for (size_t c = g * " << fShapeW[1] << "; c < (g + 1) * " << fShapeW[1] << "; c++) {\n";
         out << "\t" << "\t" << "\t" << "\t" << "for (size_t h = 0; h < " << fShapeX[2] + fAttrPads[0] + fAttrPads[2] - fAttrKernelShape[0] + 1
             << "; h += " << fAttrStrides[0] << ") {\n";
         out << "\t" << "\t" << "\t" << "\t" << "\t" << "for (size_t w = 0; w < "
             << fShapeX[3] + fAttrPads[1] + fAttrPads[3] - fAttrKernelShape[1] + 1 << ";w += " << fAttrStrides[1] << ") {\n";
         out << "\t" << "\t" << "\t" << "\t" << "\t" << "\t" << "for (size_t x = 0; x < " << fAttrKernelShape[0] << "; x++) {\n";
         out << "\t" << "\t" << "\t" << "\t" << "\t" << "\t" << "size_t offset = n * "
             << fShapeX[1] * (fShapeX[2] + fAttrPads[0] + fAttrPads[2]) * (fShapeX[3] + fAttrPads[1] + fAttrPads[3])
             << " + c * " << (fShapeX[2] + fAttrPads[0] + fAttrPads[2]) * (fShapeX[3] + fAttrPads[1] + fAttrPads[3])
             << "+ (h + x) * " << (fShapeX[3] + fAttrPads[1] + fAttrPads[3]) << " + w;\n";
         out << "\t" << "\t" << "\t" << "\t" << "\t" << "\t" << "std::copy(" << OpName << "_xpad + offset, " << OpName
             << "_xpad + offset + " << fAttrKernelShape[1] << ", " << OpName << "_xcol + index);\n";
         out << "\t" << "\t" << "\t" << "\t" << "\t" << "\t" << "index += " << fAttrKernelShape[1] << ";\n";
         out << "\t" << "\t" << "\t" << "\t" << "\t" << "\t" << "}\n";
         out << "\t" << "\t" << "\t" << "\t" << "\t" << "}\n";
         out << "\t" << "\t" << "\t" << "\t" << "}\n";
         out << "\t" << "\t" << "\t" << "}\n";
         out << "\t" << "\t" << "}\n";
         out << "\t" << "}\n";
         size_t fgHeight = fAttrKernelShape[0] / fAttrGroup;
         size_t fgWidth  = fAttrKernelShape[1] * fAttrKernelShape[2] * fAttrKernelShape[3];
         if (fType == "float") {
            out << "\t" << "float " << OpName << "_fg[" << fgHeight * fgWidth << "];\n";
         }
         size_t xgHeight = fShapeX[1] * fAttrKernelShape[2] * fAttrKernelShape[3] / fAttrGroup;
         size_t xgWidth  = fShapeX[0] * fShapeY[2] * fShapeY[3];
         if (fType == "float") {
            out << "\t" << "float " << OpName << "_xg[" << xgHeight * xgWidth << "];\n";
            out << "\t" << "float " << OpName << "_yg[" << fgHeight * xgWidth << "];\n";
         }
         out << "\t" << "char " << OpName << "_transA  = 'N';\n";
         out << "\t" << "char " << OpName << "_transB = 'N';\n";
         out << "\t" << "int " << OpName << "_m = " << fgHeight << ";\n";
         out << "\t" << "int " << OpName << "_n = " << xgWidth << ";\n";
         out << "\t" << "int " << OpName << "_k = " << fgWidth << ";\n";

         if (fType == "float") {
            out << "\t" << "float " << OpName << "_alpha = 1.0;\n";
            out << "\t" << "float " << OpName << "_beta  = 0.0;\n";
         }

         out << "\t" << "for (size_t g = 0; g < " << fAttrGroup << "; g++) {\n";
         out << "\t" << "\t" << "for (size_t h = 0; h < " << fgHeight << "; h++) {\n";
         out << "\t" << "\t" << "\t" << "for (size_t w = 0; w < " << fgWidth << "; w++) {\n";
         out << "\t" << "\t" << "\t" << "\t" << OpName << "_fg[h + w * " << fgHeight << "] = " << OpName
             << "_f[h + g * " << fgHeight << " + w * " << fAttrKernelShape[0] << "];\n";
         out << "\t" << "\t" << "\t" << "}\n";
         out << "\t" << "\t" << "}\n";
         out << "\t" << "\t" << "for (size_t h = 0; h < " << xgHeight << "; h++) {\n";
         out << "\t" << "\t" << "\t" << "for (size_t w = 0; w < " << xgWidth << "; w++) {\n";
         out << "\t" << "\t" << "\t" << "\t" << OpName << "_xg[h + w * " << xgHeight << "] = " << OpName
             << "_xcol[h + g * " << xgHeight << " + w * " << fShapeX[1] * fAttrKernelShape[2] * fAttrKernelShape[3] << "];\n";
         out << "\t" << "\t" << "\t" << "}\n";
         out << "\t" << "\t" << "}\n";
         out << "\t" << "\t" << "BLAS::sgemm_(&" << OpName << "_transA, &" << OpName << "_transB, &" << OpName
             << "_m, &" << OpName << "_n, &" << OpName << "_k, &" << OpName << "_alpha, " << OpName << "_fg, &"
             << OpName << "_m, " << OpName << "_xg, &" << OpName << "_k, &" << OpName << "_beta, " << OpName
             << "_yg, &" << OpName << "_m);\n";
         out << "\t" << "\t" << "for (size_t i = 0; i < " << fgHeight * xgWidth << "; i++) {\n";
         out << "\t" << "\t" << "\t" << "tensor_" << fNY << "[i + g * " << fgHeight * xgWidth << "] = "
             << OpName << "_yg[i];\n";
         out << "\t" << "\t" << "}\n";
         out << "\t" << "}\n";
      }

      if (fNB != "") {
         out << "\t" << "int " << OpName << "_size = " << fShapeY[0] * fShapeY[1] * fShapeY[2] * fShapeY[3] << ";\n";
         out << "\t" << "float " << OpName << "_gamma = 1.0;\n";
         out << "\t" << "int " << OpName << "_incx = 1;\n";
         out << "\t" << "int " << OpName << "_incy = 1;\n";

         out << "\t" << "BLAS::saxpy_(&" << OpName << "_size, &" << OpName << "_gamma, tensor_" << fNB << ", &"
             << OpName << "_incx, tensor_" << fNY << ", &" << OpName << "_incy);\n";
      }

      return out.str();
   }

};

} // namespace SOFIE
} // namespace Experimental
} // namespace TMVA

#endif
