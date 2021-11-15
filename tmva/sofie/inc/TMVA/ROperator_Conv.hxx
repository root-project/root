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
   std::string fNB2; // bias tensor name after broadcasting
   std::string fNY;

   std::vector<size_t> fShapeX;
   std::vector<size_t> fShapeW;
   std::vector<size_t> fShapeB;
   std::vector<size_t> fShapeY;

   std::string fType;

   bool fUseSession = false;

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

   // funciton returning output shape given input 
   std::vector<std::vector<size_t>> ShapeInference(std::vector<std::vector<size_t>> input) {
      // shape of convolution input has to be (according to ONNX): NxCxHxW  
      // Where N is batch size, C : input  channels, H : input height, W = input width
   
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

      // output is N x M x OH x OW 
      std::vector<std::vector<size_t>> ret({{input[0][0], input[1][0], outputHeight, outputWidth}});
      return ret;
   }

   void Initialize(RModel& model) {
      fUseSession = model.UseSession();
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
            // make bias shape equal to Y shape by adding 1
            if (fShapeB.size() < 1)
               throw std::runtime_error("TMVA SOFIE Conv op: Bias Tensor has empty shape");
            // we assume bias tensor dimension is equal to number of filters that is the second dimension in 
            // the output tensor
            if (fShapeB[0] != fShapeY[1])
               throw std::runtime_error("TMVA SOFIE Conv op: Bias Tensor has wrong shape: " +
                                           ConvertShapeToString(fShapeB));
            if (fType != "float")
               throw std::runtime_error("TMVA SOFIE Conv op: Broadcasting for non-float type tensors is not supported");
            
            // here the acual broadcasting
            if (!fUseSession) {

               fShapeB.resize(fShapeY.size(), 1.);

               std::shared_ptr<void> new_data_ptr(
                  UTILITY::Unidirectional_broadcast<float>(static_cast<float *>(original_data.get()), fShapeB, fShapeY),
                  std::default_delete<float[]>());
               model.UpdateInitializedTensor(fNB, model.GetTensorType(fNB), fShapeY, new_data_ptr);
               fShapeB = model.GetTensorShape(fNB);
               fNB2 = fNB;   // use same name
            }
            else {  
               // In case of session add broadcasting code in Session constructor and in GenerateInitCode
               // we need to add a new intermediate tensor for broadcasted bias tensor
               fNB2 = fNB + "bcast";
               model.AddIntermediateTensor(fNB2, model.GetTensorType(fNB), fShapeY);
            }
         }
      }
      
   }

   std::string GenerateInitCode() {
      std::stringstream out;
      // generate initialization code for broadcasting of bias tensor  
      if (fShapeB.size() != fShapeY.size() && !fNB2.empty() ) {
         // include a separate scope to avoid defining unique operator temp variables 
         out << "   {\n"; 
         out << "      std::vector<size_t> oldShape = " << ConvertShapeToString(fShapeB) << ";\n";
         out << "      std::vector<size_t> newShape = { " << fShapeY[1] << ", " << fShapeY[2] << ", " << fShapeY[3] << "};\n";
         out << "      oldShape.resize(newShape.size(), 1.);\n";
         std::string original_bias_tensor = "tensor_" + fNB;
         std::string new_bias_tensor = "tensor_" + fNB2;
         out << "      float * newData_ptr = TMVA::Experimental::SOFIE::UTILITY::Unidirectional_broadcast<float>("
             << original_bias_tensor << ", oldShape, newShape);\n";
         // extend the new broadcasted bias tensor for the batch dimension
         int length =  fShapeY[1]*fShapeY[2]*fShapeY[3]; // output nc*h*w
         out << "      for (int i = 0; i < " << fShapeY[0] << " ; i++)\n";
         out << "         std::copy(newData_ptr, newData_ptr + " << length << ", "
             <<  new_bias_tensor << " + i * " << length << ");\n";
         out << "      delete [] newData_ptr;\n";
         out << "   }\n";
      }
      return out.str();
   }
   
   // generate code for Session data members (e.g. internal vectors)
   virtual std::string GenerateSessionMembersCode(std::string opName) {
      opName = "op_" + opName;
      std::stringstream out;
      // matrix with convolution kernels
      out << "std::vector<" << fType << "> fVec_" << opName << "_f = std::vector<" << fType << ">("
          << fShapeW[0] * fShapeW[1] * fAttrKernelShape[0] * fAttrKernelShape[1] << ");\n";
      // pad input matrix with zero
      out << "std::vector<" << fType << "> fVec_" << opName << "_xpad = std::vector<" << fType << ">("
          << fShapeX[0] * fShapeX[1] * (fShapeX[2] + fAttrPads[0] + fAttrPads[2]) *
                (fShapeX[3] + fAttrPads[1] + fAttrPads[3])
          << ");\n";
      // output matrix of im2col
      out << "std::vector<" << fType << "> fVec_" << opName << "_xcol = std::vector<" << fType << ">(" 
          << fShapeW[1] * fAttrKernelShape[0] * fAttrKernelShape[1] * fShapeY[2] * fShapeY[3] << ");\n";
      out << "\n";

      return out.str(); 
   }

   std::string Generate(std::string OpName) {
      OpName = "op_" + OpName;
      const std::string SP = "   ";   // empty space to inline the code and avoid using tabs

      if (fShapeX.empty() || fShapeW.empty() || (fNB != "" && fShapeB.empty()) || fShapeY.empty()) {
         throw
            std::runtime_error("TMVA SOFIE Conv Op called to Generate without being initialized first");
      }

      std::stringstream out;
      size_t bsize = fShapeX[0];

      out << "\n//----  operator Conv " << OpName << "\n";

      // create first matrix with convolution kernels
      if (fUseSession)
         out << SP << fType << " * " << OpName << "_f = fVec_" << OpName << "_f.data();\n";
      else 
         out << SP << fType << " " << OpName << "_f[" << fShapeW[0] * fShapeW[1] * fAttrKernelShape[0] * fAttrKernelShape[1] << "] = {0};\n";

      // vectorize the (dilated)convolution kernels into a matrix
      // no need to transpose the matrix
      size_t hstride = fShapeW[3];
      size_t hstrideDil = fAttrDilations[0] * fAttrKernelShape[1];  // stride dilated in the height
      size_t wstrideDil = fAttrDilations[1];
      size_t dstride = fShapeW[2] * fShapeW[3];
      size_t dstrideDil = fAttrKernelShape[0] * fAttrKernelShape[1];
      size_t kstride = fShapeW[1] * fShapeW[2] * fShapeW[3];
      size_t kstrideDil = fShapeW[1] * dstrideDil;

      out << SP << "for (std::size_t k = 0; k < " << fShapeW[0] << "; k++) {\n";
      out << SP << SP << "for (std::size_t d = 0; d < " << fShapeW[1] << "; d++) {\n";
      out << SP << SP << SP << "for (std::size_t h = 0; h < " << fShapeW[2] << "; h++) {\n";
      out << SP << SP << SP << SP << "for (std::size_t w = 0; w < " << fShapeW[3] << "; w++) {\n";
      // out << SP << SP << SP << SP << SP << OpName <<  "_f[k + " << "(d * "
      //     << fAttrKernelShape[0] * fAttrKernelShape[1] << " + h * " << fAttrDilations[0] * fAttrKernelShape[1]
      //     << " + w * " << fAttrDilations[1] << ") * " << fShapeW[0] << "] = tensor_" << fNW << "[k * "
      //     << fShapeW[1] * fShapeW[2] * fShapeW[3] << " + d * " << fShapeW[2] * fShapeW[3] << " + h * "
      //     << fShapeW[3] << " + w ];\n";
      out << SP << SP << SP << SP << SP << OpName <<  "_f[k * "
          << kstrideDil << " + d * " << dstrideDil << " + h * " << hstrideDil << " + w * " << wstrideDil 
          << "  ] = tensor_" << fNW << "[k * " << kstride << " + d * " << dstride << " + h * "
          << hstride << " + w ];\n";

      out << SP << SP << SP << SP << "}\n";
      out << SP << SP << SP << "}\n";
      out << SP << SP << "}\n";
      out << SP << "}\n";

      // pad inputs with zero
      if (fUseSession)
         out << SP << fType << " * " << OpName << "_xpad = fVec_" << OpName << "_xpad.data();\n";
      else 
         out << SP << fType << " " << OpName << "_xpad[" <<  fShapeX[0] * fShapeX[1] * (fShapeX[2] + fAttrPads[0] + fAttrPads[2])
          * (fShapeX[3] + fAttrPads[1] + fAttrPads[3]) << "] = {0};\n";
      // Padding the input with zeros
      if (bsize == 1) {
         out << SP << "for (size_t c = 0; c < " << fShapeX[1] << "; c++) {\n";
         out << SP << SP << "for (size_t h = 0; h < " << fShapeX[2] << "; h++) {\n";
         out << SP << SP << SP << "size_t xpad_offset = c * "
             << (fShapeX[2] + fAttrPads[0] + fAttrPads[2]) * (fShapeX[3] + fAttrPads[1] + fAttrPads[3]) << " + (h + " << fAttrPads[0]
             << ") * " << (fShapeX[3] + fAttrPads[1] + fAttrPads[3]) << " + " << fAttrPads[1] << ";\n";
         out << SP << SP << SP << "size_t x_offset = c * " << fShapeX[2] * fShapeX[3] << " + h * " << fShapeX[3] << ";\n";
         out << SP << SP << SP << "std::copy(tensor_" << fNX << " + x_offset, tensor_" << fNX
             << " + x_offset + " << fShapeX[3] << ", " << OpName << "_xpad + xpad_offset);\n";
         out << SP << SP << "}\n";
         out << SP << "}\n";

      } else { 
         // case batch size is not 1 
         out << SP << "for (size_t n = 0; n < " << bsize << "; n++) {\n";
         out << SP << SP << "for (size_t c = 0; c < " << fShapeX[1] << "; c++) {\n";
         out << SP << SP << SP << "for (size_t h = 0; h < " << fShapeX[2] << "; h++) {\n";
         out << SP << SP << SP << SP << "size_t xpad_offset = n * "
             << fShapeX[1] * (fShapeX[2] + fAttrPads[0] + fAttrPads[2]) * (fShapeX[3] + fAttrPads[1] + fAttrPads[3])
             << " + c * " << (fShapeX[2] + fAttrPads[0] + fAttrPads[2]) * (fShapeX[3] + fAttrPads[1] + fAttrPads[3])
             << " + (h + " << fAttrPads[0] <<  ") * " << (fShapeX[3] + fAttrPads[1] + fAttrPads[3]) << " + "
             << fAttrPads[1] << ";\n";
         out << SP << SP << SP << SP << "size_t x_offset = n * " << fShapeX[1] * fShapeX[2] * fShapeX[3] << " + c * "
             << fShapeX[2] * fShapeX[3] << " + h * " << fShapeX[3] << ";\n";
         out << SP << SP << SP << "std::copy(tensor_" << fNX << " + x_offset, tensor_" << fNX
             << " + x_offset + " << fShapeX[3] << ", " << OpName << "_xpad + xpad_offset);\n";
         out << SP << SP << SP << "}\n";
         out << SP << SP << "}\n";
         out << SP << "}\n";
      }

      out << SP << "char " << OpName << "_transA = 'T';\n";
      out << SP << "char " << OpName << "_transB = 'N';\n";
      out << SP << "int " << OpName << "_m = " << fShapeY[2] * fShapeY[3] << ";\n"; // output h*w
      assert(fShapeY[1] == fShapeW[0]);
      assert(fShapeW[1] == fShapeX[1] / fAttrGroup);
      out << SP << "int " << OpName << "_n = " << fShapeW[0] << ";\n"; // output channels
      out << SP << "int " << OpName << "_k = " << fShapeW[1] * fAttrKernelShape[0] * fAttrKernelShape[1] << ";\n";
      out << SP << "float " << OpName << "_alpha = 1.0;\n";
      out << SP << "float " << OpName << "_beta = 0.0;\n";

      if (fUseSession) 
         out << SP << fType << " * " << OpName << "_xcol = fVec_" << OpName << "_xcol.data();\n";
      else
         out << SP << fType << " " << OpName << "_xcol[" << fShapeX[1] * fAttrKernelShape[0] *
               fAttrKernelShape[1] * fShapeY[2] * fShapeY[3] << "] = {0};\n";

      // Loop on batch size 
      std::string outOffset = "offset_tensor_" + fNY;
      out << SP << "size_t " << outOffset << " = 0;\n";
      out << SP << "for (size_t n = 0; n < " << bsize << "; n++) {\n";

      // Padding the input with zeros

      out << SP << SP << "for (size_t c = 0; c < " << fShapeX[1] << "; c++) {\n";
      out << SP << SP << SP << "for (size_t h = 0; h < " << fShapeX[2] << "; h++) {\n";
      out << SP << SP << SP << SP << "size_t xpad_offset = c * "
         << (fShapeX[2] + fAttrPads[0] + fAttrPads[2]) * (fShapeX[3] + fAttrPads[1] + fAttrPads[3]) << " + (h + "
         << fAttrPads[0] << ") * " << (fShapeX[3] + fAttrPads[1] + fAttrPads[3]) << " + " << fAttrPads[1] << ";\n";
      if (bsize == 1)
         out << SP << SP << SP << SP << "size_t x_offset = c * " << fShapeX[2] * fShapeX[3] << " + h * " << fShapeX[3] << ";\n";
      else
         out << SP << SP << SP << SP << "size_t x_offset = n * " << fShapeX[1] * fShapeX[2] * fShapeX[3] << " + c * "
            << fShapeX[2] * fShapeX[3] << " + h * " << fShapeX[3] << ";\n";

      out << SP << SP << SP << SP << "std::copy(tensor_" << fNX << " + x_offset, tensor_" << fNX << " + x_offset + "
         << fShapeX[3] << ", " << OpName << "_xpad + xpad_offset);\n";
      out << SP << SP << SP << "}\n";
      out << SP << SP << "}\n";

         // IM2COL: Unroll the input tensor
         // order input data as  (e.g. kernel 2x2)  and (xa,ya) is channel 1 and (xb,yb) is channel 2
         //   (xa1,..,xak,ya1,..yak)(xb1,...,xbk,yb1,..,ybk)
         //   (xa2,...xak+1,ya1,...yak)(......)
      if (fAttrGroup == 1) {
         // case of standard convolution

         out << SP << SP << "size_t " << OpName << "_index = 0;\n";
         out << SP << SP  << "for (size_t h = 0; h < " << fShapeX[2] + fAttrPads[0] + fAttrPads[2] - fAttrKernelShape[0] + 1
             << "; h += " << fAttrStrides[0] << ") {\n";
         out << SP << SP  << SP << "for (size_t w = 0; w < " << fShapeX[3] + fAttrPads[1] + fAttrPads[3] - fAttrKernelShape[1] + 1
             << ";w += " << fAttrStrides[1] << ") {\n";
         // loop on input channel must be done inside loop on input pixels
         out << SP << SP <<  SP << SP << "for (size_t c = 0; c < " << fShapeW[1] << "; c++) {\n";
         out << SP << SP << SP << SP << SP << "for (size_t x = 0; x < " << fAttrKernelShape[0] << "; x++) {\n";
         out << SP << SP << SP << SP << SP << "size_t offset = "
             << " c * " << (fShapeX[2] + fAttrPads[0] + fAttrPads[2]) * (fShapeX[3] + fAttrPads[1] + fAttrPads[3])
             << " + (h + x) * " << (fShapeX[3] + fAttrPads[1] + fAttrPads[3]) << " + w;\n";
         //out << "assert( offset + 5 <= " << fShapeX[1] * (fShapeX[2] + fAttrPads[0] + fAttrPads[2]) * (fShapeX[3] + fAttrPads[1] + fAttrPads[3]) << " );\n";
         //out << "assert( " << OpName << "_index + 5  <= " << fShapeW[1] * fAttrKernelShape[0] * fAttrKernelShape[1] * fShapeY[2] * fShapeY[3] << " );\n";
         out << SP << SP << SP << SP << SP << "std::copy(" << OpName << "_xpad + offset, " << OpName
             << "_xpad + offset + " << fAttrKernelShape[1] << ", " << OpName << "_xcol + " << OpName << "_index);\n";
         out << SP << SP << SP << SP << SP << OpName << "_index += " << fAttrKernelShape[1] << ";\n";
         out << SP << SP << SP << SP << SP << "}\n";
         out << SP << SP << SP << SP << "}\n";
         out << SP << SP << SP << "}\n";
         out << SP << SP << "}\n";
         
         out << SP <<  SP << "BLAS::sgemm_(&" << OpName << "_transA, &" << OpName << "_transB, &" << OpName << "_m, &"
             << OpName << "_n, &" << OpName << "_k, &" << OpName << "_alpha, " << OpName << "_xcol, &" << OpName << "_k,\n";
         out << SP << SP  << SP << OpName << "_f, &" << OpName << "_k, &" << OpName << "_beta, tensor_" << fNY
             << " + " << outOffset << ", &" << OpName << "_m);\n";

         out << SP << SP << outOffset << " += " << fShapeW[0]*fShapeY[2]*fShapeY[3]<< ";\n";

      } else {
         // case of group convolution

         // Unroll (IM2COL) the input tensor- make loop on groups and repeat operations (IM2COL + GEMM for each group)
         out << SP << SP << "for (size_t g = 0; g < " << fAttrGroup << "; g++) {\n";
         // increment index by group offset
         out << SP << SP << SP << "size_t index = 0;\n"; //g * " << fShapeW[1] * fAttrKernelShape[0] * fAttrKernelShape[1] << ";\n";
         //out << SP << SP << "for (size_t n = 0; n < " << bsize << "; n++) {\n";
         out << SP << SP << SP << "for (size_t h = 0; h < " << fShapeX[2] + fAttrPads[0] + fAttrPads[2] - fAttrKernelShape[0] + 1
             << "; h += " << fAttrStrides[0] << ") {\n";
         out << SP << SP << SP << SP << "for (size_t w = 0; w < "
             << fShapeX[3] + fAttrPads[1] + fAttrPads[3] - fAttrKernelShape[1] + 1 << ";w += " << fAttrStrides[1] << ") {\n";
         // loop on input equivalent channels = total input channels/ngroups = shapeW[1]
         out << SP << SP << SP << SP << SP << "for (size_t c = 0 ; c <  " << fShapeW[1] << "; c++) {\n";
         ///out << SP << SP << SP << "for (size_t c = g * " << fShapeW[1] << "; c < (g + 1) * " << fShapeW[1] << "; c++) {\n";
         out << SP << SP << SP << SP << SP << SP << "for (size_t x = 0; x < " << fAttrKernelShape[0] << "; x++) {\n";

         out << SP << SP << SP << SP << SP << SP << SP << "size_t offset = " << " g * " <<
                fShapeW[1] * (fShapeX[2] + fAttrPads[0] + fAttrPads[2]) * (fShapeX[3] + fAttrPads[1] + fAttrPads[3])
             << "+ c * " << (fShapeX[2] + fAttrPads[0] + fAttrPads[2]) * (fShapeX[3] + fAttrPads[1] + fAttrPads[3])
             << "+ (h + x) * " << (fShapeX[3] + fAttrPads[1] + fAttrPads[3]) << " + w;\n";
         out << SP << SP << SP << SP << SP << SP << SP << "std::copy(" << OpName << "_xpad + offset, " << OpName
             << "_xpad + offset + " << fAttrKernelShape[1] << ", " << OpName << "_xcol + index);\n";
         out << SP << SP << SP << SP << SP << SP << SP << "index += " << fAttrKernelShape[1] << ";\n";
         out << SP << SP << SP << SP << SP << SP << "}\n";
         out << SP << SP << SP << SP << SP << "}\n";
         out << SP << SP << SP << SP << "}\n";
         out << SP << SP << SP << "}\n";   // end im2col loop 

         // n must be divided by the number of groups 
         out << SP << SP << SP << OpName << "_n = " << fShapeW[0] / fAttrGroup << ";\n";
         // offset g must be  g * k * n
         out << SP << SP << SP << "size_t offset_f = g * " << fShapeW[0] * fShapeW[1] * fAttrKernelShape[0] * fAttrKernelShape[1] / fAttrGroup << ";\n";
         out << SP << SP << SP << "BLAS::sgemm_(&" << OpName << "_transA, &" << OpName << "_transB, &" << OpName
             << "_m, &" << OpName << "_n, &" << OpName << "_k, &" << OpName << "_alpha, " << OpName << "_xcol, &"
             << OpName << "_k, " << OpName << "_f + offset_f, &" << OpName << "_k, &" << OpName << "_beta, tensor_" << fNY << " + "
             << outOffset << ", &" << OpName << "_m);\n";
         // out << SP << SP << "for (size_t i = 0; i < " << fgHeight * xgWidth << "; i++) {\n";
         // out << SP << SP << SP << "tensor_" << fNY << "[i + g * " << fgHeight * xgWidth << "] = "
         //     << OpName << "_yg[i];\n";

         out << SP << SP << SP << outOffset << " += " << fShapeW[0]*fShapeY[2]*fShapeY[3] / fAttrGroup << ";\n";
         out << SP << SP << "}\n";    // end of group loop
        
         } // endif group convolution

          out << SP << "}\n"; // end of batch size loop

    
      if (fNB2 != "") {
         out << SP << "int " << OpName << "_size = " << fShapeY[0] * fShapeY[1] * fShapeY[2] * fShapeY[3] << ";\n";
         out << SP << "float " << OpName << "_gamma = 1.0;\n";
         out << SP << "int " << OpName << "_incx = 1;\n";
         out << SP << "int " << OpName << "_incy = 1;\n";

         out << SP << "BLAS::saxpy_(&" << OpName << "_size, &" << OpName << "_gamma, tensor_" << fNB2 << ", &"
             << OpName << "_incx, tensor_" << fNY << ", &" << OpName << "_incy);\n";

      }

      
      return out.str();
      }
};

} // namespace SOFIE
} // namespace Experimental
} // namespace TMVA

#endif
