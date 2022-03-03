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
#include <cassert>

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

   size_t fDim;   // dimension of the convolution


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
         if (input[i].size() -2 != fDim) {
            throw
               std::runtime_error("TMVA SOFIE Conv Op Shape inference - invalid inputs ");
         }
      }

      if (fAttrGroup == 0) {
         fAttrGroup = input[0][1] / input[1][1];
      }

      // kernel shape  
      size_t k1 = ((fAttrKernelShape.empty())? input[1][2] : fAttrKernelShape[0]);
      size_t k2 = (fDim > 1) ? ((fAttrKernelShape.empty()) ? input[1][3] : fAttrKernelShape[1]) : 1;
      size_t k3 = (fDim > 2) ? ((fAttrKernelShape.empty()) ? input[1][4] : fAttrKernelShape[2]) : 1;


      size_t i1 = (fDim > 1) ? ((fDim > 2) ? 3 : 2) : 1;
      size_t i2 = (fDim > 2) ? 4 : 3;
      size_t i3 = 5;

      if (fAttrDilations.empty()) {
         fAttrDilations = {1, 1, 1};
      }
      fAttrDilations.resize(3); 
      if (fDim < 3) {
         fAttrDilations.resize(3, 1);
      }
      // Shape of the kernel
      fAttrKernelShape = {k1 + (fAttrDilations[0] - 1) * (k1 - 1), 
                          k2 + (fAttrDilations[1] - 1) * (k2 - 1),
                          k3 + (fAttrDilations[2] - 1) * (k3 - 1)};

      if (fAttrAutopad == "NOTSET") {
         if (fAttrPads.empty()) {
            fAttrPads = {1, 1, 1, 1, 1, 1};
         }
      } else if (fAttrAutopad == "SAME_UPPER" || fAttrAutopad == "SAME_LOWER") {
         if (fDim == 1)
            fAttrPads = {fAttrKernelShape[0] / 2, fAttrKernelShape[0] / 2};
         else if (fDim == 2)
            fAttrPads = {fAttrKernelShape[0] / 2, fAttrKernelShape[1] / 2, fAttrKernelShape[0] / 2, fAttrKernelShape[1] / 2};
         else if (fDim == 3)
            fAttrPads = {fAttrKernelShape[0] / 2, fAttrKernelShape[1] / 2, fAttrKernelShape[2] / 2,
                         fAttrKernelShape[0] / 2, fAttrKernelShape[1] / 2, fAttrKernelShape[2] / 2};
         // add extra padding at beginnig or end (depending if SAME_UPPER or SAME_LOWER)
         // need to check this!
         if (fAttrKernelShape[0] % 2 == 1) {
            (fAttrAutopad == "SAME_UPPER") ? fAttrPads[0]++ : fAttrPads[i1]++;
         }
         if (fDim > 1 && fAttrKernelShape[1] % 2 == 1) {
            (fAttrAutopad == "SAME_UPPER") ? fAttrPads[1]++ : fAttrPads[i2]++;
         }
         if (fDim > 2 && fAttrKernelShape[2] % 2 == 1) {
            (fAttrAutopad == "SAME_UPPER") ? fAttrPads[2]++ : fAttrPads[i3]++;
         }
      } else if (fAttrAutopad != "VALID") {
         throw
            std::runtime_error("TMVA SOFIE Conv Op invalid fAutopad");
      }
      // to be sure pad is vector of size 6
      if (fDim < 3) fAttrPads.resize(6, 0);

      if (fAttrStrides.empty()) {
         fAttrStrides = {1, 1, 1};
      }
      if (fDim < 3)
         fAttrStrides.resize(3, 1);


      size_t input1 = input[0][2];
      size_t input2 = (fDim > 1) ? input[0][3] : 1;
      size_t input3 = (fDim > 2) ? input[0][4] : 1;

      size_t pad1 = fAttrPads[0] + fAttrPads[i1];
      size_t output1 = (input1 + pad1 - fAttrKernelShape[0]) / fAttrStrides[0] + 1;

      size_t batch_size = input[0][0];        // first element in input tensor
      size_t output_channels = input[1][0];   // first element in weight tensor

      std::vector<std::vector<size_t>> ret({{batch_size, output_channels, output1 }});

      if (fDim == 1) 
         return ret;

      size_t pad2 = fAttrPads[1] + fAttrPads[i2];
      size_t output2 = (input2 + pad2 - fAttrKernelShape[1]) / fAttrStrides[1] + 1;
      // output is N x M x OH x OW
      ret[0].push_back(output2);
      if (fDim == 2)
         return ret;

      size_t pad3 = fAttrPads[2] + fAttrPads[i3];
      size_t output3 = (input3 + pad3 - fAttrKernelShape[2] ) / fAttrStrides[2] + 1;

      // output is N x M x OH x OW x OD
      ret[0].push_back(output3);
      return ret;
   }

   void Initialize(RModel& model) {
      fUseSession = model.UseSession();
      if (!model.CheckIfTensorAlreadyExist(fNX)) {
         throw
            std::runtime_error("TMVA SOFIE Conv op Input Tensor " + fNX + " is not found in model");
      }
      fShapeX = model.GetTensorShape(fNX);
      if (fShapeX.size() < 3 || fShapeX.size()  > 5) {
         std::cout << fNX << " : " << ConvertShapeToString(fShapeX) << std::endl;
         throw
            std::runtime_error("TMVA SOFIE Conv Op input data tensor" + fNX + " is not of 3,4 or 5 dimensions");
      }
      fDim = fShapeX.size() - 2;
      if (!model.CheckIfTensorAlreadyExist(fNW)) {
         throw
            std::runtime_error("TMVA SOFIE Conv op Input weight Tensor " + fNW + " is not found in model");
      }
      fShapeW = model.GetTensorShape(fNW);
      if (fShapeW.size() < 3 || fShapeW.size()  > 5) {
         std::cout << fNW << " : " << ConvertShapeToString(fShapeW) << std::endl;
         throw std::runtime_error("TMVA SOFIE Conv Op input weight tensor" + fNW + " is not of 3,4 or 5 dimensions");
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

      size_t oDepth = (fDim > 2) ? fShapeY[2] : 1; // output depth
      size_t oHeight = (fDim > 1) ? fShapeY[fDim] : 1;  // ouput height
      size_t oWidth = fShapeY[fDim+1]; // output width

      std::stringstream out;
      // generate initialization code for broadcasting of bias tensor  
      if (fShapeB.size() != fShapeY.size() && !fNB2.empty() ) {
         // include a separate scope to avoid defining unique operator temp variables 
         out << "   {\n"; 
         out << "      std::vector<size_t> oldShape = " << ConvertShapeToString(fShapeB) << ";\n";
         out << "      std::vector<size_t> newShape = { " << fShapeY[1] << ", ";
         if (fDim > 2) out << oDepth << ", ";
         if (fDim > 1) out << oHeight << ", ";
         out << oWidth << "};\n";
         out << "      oldShape.resize(newShape.size(), 1.);\n";
         std::string original_bias_tensor = "tensor_" + fNB;
         std::string new_bias_tensor = "tensor_" + fNB2;
         out << "      float * newData_ptr = TMVA::Experimental::SOFIE::UTILITY::Unidirectional_broadcast<float>("
             << original_bias_tensor << ", oldShape, newShape);\n";
         // extend the new broadcasted bias tensor for the batch dimension
         int length =  fShapeY[1]*oDepth*oHeight*oWidth; // output nc*h*w
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

      size_t outputChannelSize = fShapeY[2];  // size/chanhel = D * H * W
      size_t kernelSize = fAttrKernelShape[0];
      for (size_t i = 1; i < fDim; i++) {
         outputChannelSize *= fShapeY[2 + i];
         kernelSize *= fAttrKernelShape[i];
      }

      opName = "op_" + opName;
      std::stringstream out;
      // matrix with convolution kernels
      out << "std::vector<" << fType << "> fVec_" << opName << "_f = std::vector<" << fType << ">("
          << fShapeW[0] * fShapeW[1] * kernelSize << ");\n";
      // output matrix of im2col
      out << "std::vector<" << fType << "> fVec_" << opName << "_xcol = std::vector<" << fType << ">(" 
          << fShapeW[1] * kernelSize * outputChannelSize << ");\n";
      out << "\n";

      return out.str(); 
   }

   std::string Generate(std::string OpName) {
      OpName = "op_" + OpName;

      if (fShapeX.empty() || fShapeW.empty() || (fNB != "" && fShapeB.empty()) || fShapeY.empty()) {
         throw
            std::runtime_error("TMVA SOFIE Conv Op called to Generate without being initialized first");
      }

      std::stringstream out;
      size_t bsize = fShapeX[0];
      size_t kDepth = (fDim > 2) ?  fShapeW[2] : 1;  // kernel depth
      size_t kHeight = (fDim > 1) ? fShapeW[fDim] : 1;  // kernel height
      size_t kWidth = fShapeW[fDim+1]; // kernel width
      size_t iDepth = (fDim > 2) ?  fShapeX[2] : 1;  // input depth
      size_t iHeight = (fDim > 1) ? fShapeX[fDim] : 1; // input height
      size_t iWidth = fShapeX[fDim+1]; // input width
      size_t oDepth = (fDim > 2) ? fShapeY[2] : 1; // output depth
      size_t oHeight = (fDim > 1) ? fShapeY[fDim] : 1;  // ouput height
      size_t oWidth = fShapeY[fDim+1]; // output width

      out << "\n//----  operator Conv " << OpName << "\n";

      // create first matrix with convolution kernels
      if (fUseSession)
         out << SP << fType << " * " << OpName << "_f = fVec_" << OpName << "_f.data();\n";
      else 
         out << SP << fType << " " << OpName << "_f[" << fShapeW[0] * fShapeW[1] * fAttrKernelShape[0] * fAttrKernelShape[1] << "] = {0};\n";

      // vectorize the (dilated)convolution kernels into a matrix
      // no need to transpose the matrix
      // to fix for 1d and 3d 

      size_t id = (fDim > 2) ? fDim-3 : 2;
      size_t ih = (fDim > 1) ? fDim-2 : 1;
      size_t iw = fDim-1;

      size_t wstrideDil = fAttrDilations[iw];
      size_t hstride = kWidth;
      size_t hstrideDil = fAttrDilations[ih] * fAttrKernelShape[iw];  // stride dilated in the height
      size_t dstride = kHeight * kWidth;
      size_t dstrideDil = fAttrDilations[id] * fAttrKernelShape[ih] * fAttrKernelShape[iw];
      size_t icstride = kHeight * kWidth * kDepth;
      size_t icstrideDil = fAttrKernelShape[id] * fAttrKernelShape[ih] * fAttrKernelShape[iw];
      size_t ocstride = fShapeW[1] * icstride;
      size_t ocstrideDil = fShapeW[1] * icstrideDil;

      out << SP << "for (std::size_t oc = 0; oc < " << fShapeW[0] << "; oc++) {\n";
      out << SP << SP << "for (std::size_t ic = 0; ic < " << fShapeW[1] << "; ic++) {\n";
      if (fDim > 2)
         out << SP << SP << SP << "for (std::size_t kd = 0; kd < " << kDepth << "; kd++) {\n";
      if (fDim > 1)
         out << SP << SP << SP << "for (std::size_t kh = 0; kh < " << kHeight << "; kh++) {\n";
      out << SP << SP << SP << SP << "for (std::size_t kw = 0; kw < " << kWidth << "; kw++) {\n";
     
      out << SP << SP << SP << SP << SP << OpName <<  "_f[oc * "
          << ocstrideDil << " + ic * " << icstrideDil; 
      if (fDim > 2) out << " + kd * " << dstrideDil;
      if (fDim > 1) out << " + kh * " << hstrideDil; 
      out << " + kw * " << wstrideDil  << "  ] = tensor_" << fNW << "[oc * " << ocstride << " + ic * " << icstride;
      if (fDim > 2) out << " + kd * " << dstride;
      if (fDim > 1) out << " + kh * " << hstride;
      out  << " + kw ];\n";

      out << SP << SP << SP << SP << "}\n";
      if (fDim > 1) out << SP << SP << SP << "}\n";
      if (fDim > 2) out << SP << SP << SP << "}\n";
      out << SP << SP << "}\n";
      out << SP << "}\n";

      //out << SP << "char " << OpName << "_transA = 'T';\n";
      out << SP << "char " << OpName << "_transA = 'N';\n";
      out << SP << "char " << OpName << "_transB = 'N';\n";
      out << SP << "int " << OpName << "_m = " << oHeight * oWidth * oDepth << ";\n"; // output h*w
      assert(fShapeY[1] == fShapeW[0]);
      assert(fShapeW[1] == fShapeX[1] / fAttrGroup);
      out << SP << "int " << OpName << "_n = " << fShapeW[0] << ";\n"; // output channels
      out << SP << "int " << OpName << "_k = " << fShapeW[1] * fAttrKernelShape[0] * fAttrKernelShape[1] * fAttrKernelShape[2] << ";\n";
      out << SP << "float " << OpName << "_alpha = 1.0;\n";
      out << SP << "float " << OpName << "_beta = 0.0;\n";

      if (fUseSession) {
         out << SP << fType << " * " << OpName << "_xcol = fVec_" << OpName << "_xcol.data();\n";
      }
      else {
         out << SP << fType << " " << OpName << "_xcol["
             << fShapeX[1] * fAttrKernelShape[0] * fAttrKernelShape[1] * fAttrKernelShape[2] * oDepth * oHeight * oWidth 
             << "] = {0};\n";
      }

      // Loop on batch size 
      out << SP << "for (size_t n = 0; n < " << bsize << "; n++) {\n";

      // IM2COL: Unroll the input tensor
      // order input data as  (e.g. kernel 2x2)  and (xa,ya) is channel 1 and (xb,yb) is channel 2
      //   (xa1,..,xak,ya1,..yak)(xb1,...,xbk,yb1,..,ybk)
      //   (xa2,...xak+1,ya1,...yak)(......)
      // trick for speed is using caffe im2col and output a matrix which contains filtered values as rows.
      // By doing this one has consecutive memory reads and writes
      // Resulting matrix op_xcol is (input channels * filter_h * filter_w , output_h * output_w)
      if (fDim ==1) {
         if (fAttrPads[0] != fAttrPads[1] ) {
            std::cout << "TMVA SOFIE Operator Conv:  asymmetric padding not supported. Assume an average padding "
                      << std::endl;
            fAttrPads[0] = (fAttrPads[0] + fAttrPads[1]) / 2;
         }
         fAttrPads[1] = 0;
         fAttrStrides[1] = 1;
      }
      if (fDim == 2) {
         if (fAttrPads[0] != fAttrPads[2] || fAttrPads[1] != fAttrPads[3]) {
            std::cout << "TMVA SOFIE Operator Conv:  asymmetric padding not supported. Assume an average padding " << std::endl;
            fAttrPads[0] = (fAttrPads[0] + fAttrPads[2]) / 2;
            fAttrPads[1] = (fAttrPads[1] + fAttrPads[3]) / 2;
         }
      } 
      if (fDim == 3) {
         if (fAttrPads[0] != fAttrPads[3] || fAttrPads[1] != fAttrPads[4] || fAttrPads[2] != fAttrPads[5]) {
            std::cout << "TMVA SOFIE Operator Conv:  asymmetric padding not supported. Assume an average padding " << std::endl;
            fAttrPads[0] = (fAttrPads[0] + fAttrPads[3]) / 2;
            fAttrPads[1] = (fAttrPads[1] + fAttrPads[4]) / 2;
            fAttrPads[2] = (fAttrPads[2] + fAttrPads[5]) / 2;
         }
      }   
      if (fAttrGroup == 1) {
         out << SP << SP << "size_t x_offset = n * " << fShapeX[1] * iHeight * iWidth << ";\n";
         out << SP << SP << "size_t out_offset = n * " << fShapeY[1] * oHeight * oWidth << ";\n";
         // when using im2col - resulting matrix is transposed, is (input_c * filter_h * filter_y,  output_h *
         // output_w)
         if (fDim < 3) {
            out << SP << SP << "TMVA::Experimental::SOFIE::UTILITY::Im2col<float>(tensor_" << fNX
                << " + x_offset,"
                //  channels, height, width, kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w, dilation_h,
                //  dilation_w,
                //
                << fShapeW[1] << "," << iHeight << "," << iWidth << ",";
            if (fDim == 1)
               out << "1, " << fAttrKernelShape[0] << ",0," << fAttrPads[0] << ",1," << fAttrStrides[0] << ",1,"
                   << fAttrDilations[0];
            else // dim ==2
               out << fAttrKernelShape[0] << "," << fAttrKernelShape[1] << "," << fAttrPads[0] << "," << fAttrPads[1]
                   << "," << fAttrStrides[0] << "," << fAttrStrides[1] << "," << fAttrDilations[0] << ","
                   << fAttrDilations[1];
            out << "," << OpName << "_xcol);\n\n ";
         } else { 
            // 3d im2col
            out << SP << SP << "TMVA::Experimental::SOFIE::UTILITY::Im2col_3d<float>(tensor_" << fNX
                << " + x_offset,"
                //  channels, d, h, w, k_d, k_h, k_w, pad_d, pad_h, pad_w, stride_d, stride_h, stride_w, 
                //  dilation_d, dilation_h, dilation_w,
                //
                << fShapeW[1] << "," << iDepth << "," << iHeight << "," << iWidth << "," 
                << fAttrKernelShape[0] << "," << fAttrKernelShape[1] << "," << fAttrKernelShape[2] << "," 
                << fAttrPads[0] << "," << fAttrPads[1] << "," << fAttrPads[2] << "," 
                << fAttrStrides[0] << "," << fAttrStrides[1] << "," << fAttrStrides[2] << "," 
                << fAttrDilations[0] << "," << fAttrDilations[1] << "," << fAttrDilations[2] << "," 
                << OpName << "_xcol);\n\n ";
         }
         // BLAS
         out << SP << SP << "BLAS::sgemm_(&" << OpName << "_transA, &" << OpName << "_transB, &" << OpName << "_m, &"
             << OpName << "_n, &" << OpName << "_k, &" << OpName << "_alpha, " << OpName << "_xcol, &" << OpName
             << "_m,\n"; // use m if op_xcol is not transpose , otherwise k
         out << SP << SP << SP << OpName << "_f, &" << OpName << "_k, &" << OpName << "_beta, tensor_" << fNY
             << " + out_offset, &" << OpName << "_m);\n";
      } else {
         // case of group convolution
         // Unroll (IM2COL) the input tensor- make loop on groups and repeat operations (IM2COL + GEMM for each
         // group)
         out << SP << SP << "size_t out_offset = n * " << fShapeY[1] * oDepth * oHeight * oWidth << ";\n";
         out << SP << SP << "for (size_t g = 0; g < " << fAttrGroup << "; g++) {\n";
         out << SP << SP << "size_t x_offset = n * " << fShapeX[1] * iDepth * iHeight * iWidth << " + g * "
             << fShapeW[1] * iDepth * iHeight * iWidth << ";\n ";
         out << SP << SP << "size_t out_offset = n * " << fShapeY[1] * oDepth * oHeight * oWidth << " + g * "
             << fShapeW[0] * oDepth * oHeight * oWidth / fAttrGroup << ";\n ";

         if (fDim < 3) {
            out << SP << SP << "TMVA::Experimental::SOFIE::UTILITY::Im2col<float>(tensor_" << fNX
                << " + x_offset,"
                //  channels, height, width, kernel_h, kernel_w, pad_h, pad_w, stride_h, stride_w, dilation_h,
                //  dilation_w,
                //
                << fShapeW[1] << "," << iHeight << "," << iWidth << ",";
            if (fDim == 1)
               out << "1, " << fAttrKernelShape[0] << ",0," << fAttrPads[0] << ",1," << fAttrStrides[0] << ",1,"
                   << fAttrDilations[0];
            else // dim ==2
               out << fAttrKernelShape[0] << "," << fAttrKernelShape[1] << "," << fAttrPads[0] << "," << fAttrPads[1]
                   << "," << fAttrStrides[0] << "," << fAttrStrides[1] << "," << fAttrDilations[0] << ","
                   << fAttrDilations[1];
            out << "," << OpName << "_xcol);\n\n ";
         } else {
            // 3d im2col
            out << SP << SP << "TMVA::Experimental::SOFIE::UTILITY::Im2col_3d<float>(tensor_" << fNX
                << " + x_offset,"
                //  channels, d, h, w, k_d, k_h, k_w, pad_d, pad_h, pad_w, stride_d, stride_h, stride_w,
                //  dilation_d, dilation_h, dilation_w,
                //
                << fShapeW[1] << "," << iDepth << "," << iHeight << "," << iWidth << "," << fAttrKernelShape[0] << ","
                << fAttrKernelShape[1] << "," << fAttrKernelShape[2] << "," << fAttrPads[0] << "," << fAttrPads[1]
                << "," << fAttrPads[2] << "," << fAttrStrides[0] << "," << fAttrStrides[1] << "," << fAttrStrides[2]
                << "," << fAttrDilations[0] << "," << fAttrDilations[1] << "," << fAttrDilations[2] << "," << OpName
                << "_xcol);\n\n ";
         } 

         // BLAS
         // n must be divided by the number of groups
         out << SP << SP << SP << OpName << "_n = " << fShapeW[0] / fAttrGroup << ";\n";
         // offset g must be  g * k * n
         out << SP << SP << SP << "size_t offset_f = g * "
             << fShapeW[0] * fShapeW[1] * fAttrKernelShape[0] * fAttrKernelShape[1] * fAttrKernelShape[2] / fAttrGroup 
             << ";\n";
         out << SP << SP << "BLAS::sgemm_(&" << OpName << "_transA, &" << OpName << "_transB, &" << OpName << "_m, &"
             << OpName << "_n, &" << OpName << "_k, &" << OpName << "_alpha, " << OpName << "_xcol, &" << OpName
             << "_m,\n"; // use m if op_xcol is not transpose , otherwise k
         out << SP << SP << SP << OpName << "_f + offset_f, &" << OpName << "_k, &" << OpName << "_beta, tensor_" << fNY
             << " + out_offset"
             << ", &" << OpName << "_m);\n";

         out << SP << SP << "}\n"; // end of group loop
      }

      out << SP << "}\n"; // end of batch size loop

    
      if (fNB2 != "") {
         out << SP << "int " << OpName << "_size = " << fShapeY[0] * fShapeY[1] * oDepth * oHeight * oWidth << ";\n";
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
