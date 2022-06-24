#ifndef TMVA_SOFIE_ROPERATOR_POOL
#define TMVA_SOFIE_ROPERATOR_POOL

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

struct RAttributes_Pool {
   // structure contain Pool attributes
   std::string auto_pad = "NOTSET";
   int ceil_mode = 0;
   int count_include_pad = 0;     // not for MaxPool
   int storage_order = 0;         // not for AveragePool
   std::vector<size_t> dilations; // not for AveragePool
   std::vector<size_t> kernel_shape;
   std::vector<size_t> pads;
   std::vector<size_t> strides;
};

enum PoolOpMode { InvalidPool, MaxPool, AveragePool, GlobalAveragePool };

template<typename T>
class ROperator_Pool final : public ROperator
{

private:

   PoolOpMode fPoolMode;

   size_t fAttrCeilMode;
   size_t fAttrCountIncludePad;
   size_t fAttrStorageOrder;
   std::string fAttrAutopad;
   std::vector<size_t> fAttrDilations;
   std::vector<size_t> fAttrKernelShape;
   std::vector<size_t> fAttrPads;
   std::vector<size_t> fAttrStrides;

   std::string fNX;
   std::string fNY;

   std::vector<size_t> fShapeX;
   std::vector<size_t> fShapeY;

   std::string fType;

   size_t fDim;   // dimension of the MaxPool
   bool fUseSession = false;

public:

   std::string Name() {
      if (fPoolMode == AveragePool)  return "AveragePool";
      if (fPoolMode == MaxPool)  return "MaxPool";
      return "Invalid";
   }

   ROperator_Pool() {}

   ROperator_Pool(PoolOpMode mode, RAttributes_Pool attr, std::string nameX, std::string nameY)
      : fPoolMode(mode), fAttrCeilMode(attr.ceil_mode), fAttrCountIncludePad(attr.count_include_pad),
        fAttrStorageOrder(attr.storage_order), fAttrAutopad(attr.auto_pad),
        fAttrDilations(attr.dilations), fAttrKernelShape(attr.kernel_shape), fAttrPads(attr.pads), fAttrStrides(attr.strides),
        fNX(UTILITY::Clean_name(nameX)), fNY(UTILITY::Clean_name(nameY))
   {
      if(std::is_same<T, float>::value) {
         fType = "float";
      } else {
         throw
            std::runtime_error("TMVA SOFIE Encountered unsupported type parsing a Pool operator");
      }
   }

   // return input type (defined abstract in ROperator class )
   std::vector<ETensorType> TypeInference(std::vector<ETensorType> input) {
      // only one input in Pool operators
      return input;
   }

   // function returning output shape given input
   std::vector<std::vector<size_t>> ShapeInference(std::vector<std::vector<size_t>> input) {
      // shape of pooling input has to be (according to ONNX): NxCxHxW
      // Where N is batch size, C : input  channels, H : input height, W = input width
      // or it can be [N, C, F1,F2,....FN] . Minimum dimension is 3
      if (input.size() != 1 ) {
         throw std::runtime_error("TMVA SOFIE" + Name() + "Op Shape inference need 1 input tensor");
      }
      if (input[0].size() < 3) {
         throw std::runtime_error("TMVA SOFIE" + Name() + "Op Shape inference only accept tensor with at least 3 dimensions");
      }
      // for the time being support only 3,4,5 dimens
      if (input[0].size() < 3 || input[0].size() >  5) {
         throw std::runtime_error("TMVA SOFIE" + Name() + "Op : tensors with dimension " + std::to_string(input[0].size()) + " are not yet supported");
      }

      if (input[0].size() -2 != fDim) {
         throw
            std::runtime_error("TMVA SOFIE Pool Op Shape inference - invalid inputs ");
      }
       // kernel shape
      size_t k1 = ((fAttrKernelShape.empty())? input[0][2] : fAttrKernelShape[0]);
      size_t k2 = (fDim > 1) ? ((fAttrKernelShape.empty()) ? input[0][3] : fAttrKernelShape[1]) : 1;
      size_t k3 = (fDim > 2) ? ((fAttrKernelShape.empty()) ? input[0][4] : fAttrKernelShape[2]) : 1;


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
         // add extra padding at beginning or end (depending if SAME_UPPER or SAME_LOWER)
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
            std::runtime_error("TMVA SOFIE" + Name() + "Op invalid Autopad value : " + fAttrAutopad);
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
      size_t output_channels = input[0][1];   // first element in output tensor

      std::vector<std::vector<size_t>> ret({{ batch_size, output_channels, output1 }});

      if (fDim == 1)
         return ret;

      size_t pad2 = fAttrPads[1] + fAttrPads[i2];
      size_t output2 = (input2 + pad2 - fAttrKernelShape[1]) / fAttrStrides[1] + 1;
      // output is N x C x OH x OW
      ret[0].push_back(output2);
      if (fDim == 2)
         return ret;

      size_t pad3 = fAttrPads[2] + fAttrPads[i3];
      size_t output3 = (input3 + pad3 - fAttrKernelShape[2] ) / fAttrStrides[2] + 1;

      // output is N x C x OH x OW x OD
      ret[0].push_back(output3);
      return ret;
   }

   void Initialize(RModel& model) {

      fUseSession = model.UseSession();

      if (!model.CheckIfTensorAlreadyExist(fNX)) {
         throw
            std::runtime_error("TMVA SOFIE Pool op Input Tensor " + fNX + " is not found in model");
      }
      fShapeX = model.GetTensorShape(fNX);
      if (fShapeX.size() < 3 || fShapeX.size()  > 5) {
         std::cout << fNX << " : " << ConvertShapeToString(fShapeX) << std::endl;
         throw
            std::runtime_error("TMVA SOFIE Pool Op input data tensor" + fNX + " is not of 3,4 or 5 dimensions");
      }
       fDim = fShapeX.size() - 2;
      // case of GlobalAveragePool. It is a pool case with kernel shape == image shape
      if (fPoolMode == GlobalAveragePool) {
         fPoolMode = AveragePool;
         fAttrKernelShape.resize(3);
         fAttrKernelShape[0] = fShapeX[2];
         if (fDim > 1)
            fAttrKernelShape[1] = fShapeX[3];
         if (fDim > 2)
            fAttrKernelShape[2] = fShapeX[4];
         fAttrAutopad = "VALID";
         fAttrPads = {0, 0, 0, 0, 0, 0 };
         assert(fAttrStrides.empty());
      }
      // find shape of Y and add it in the list of intermidiate tensors
      fShapeY = ShapeInference({fShapeX})[0];
      model.AddIntermediateTensor(fNY, model.GetTensorType(fNX), fShapeY);

      // need cmath for INFINITY when using MaxPool
      if (fPoolMode == MaxPool) model.AddNeededStdLib("cmath");

   }

   std::string GenerateInitCode() {
      std::stringstream out;
      return out.str();
   }

   // generate code for Session data members (e.g. internal vectors)
   virtual std::string GenerateSessionMembersCode(std::string opName)
   {
      opName = "op_" + opName;
      std::stringstream out;
      // input matrix padded with zero
      if(fDim == 1){
          out << "std::vector<" << fType << "> fVec_" << opName << "_xpad = std::vector<" << fType << ">("
          << fShapeX[1] * (fShapeX[2] + fAttrPads[0] + fAttrPads[2]) << ");\n";
      }
      else if(fDim == 2){
          out << "std::vector<" << fType << "> fVec_" << opName << "_xpad = std::vector<" << fType << ">("
          << fShapeX[1] * (fShapeX[2] + fAttrPads[0] + fAttrPads[2]) * (fShapeX[3] + fAttrPads[1] + fAttrPads[3])
          << ");\n";
      }
      else{ //dim is 3D
          out << "std::vector<" << fType << "> fVec_" << opName << "_xpad = std::vector<" << fType << ">("
          << fShapeX[1] * (fShapeX[2] + fAttrPads[0] + fAttrPads[2]) * (fShapeX[3] + fAttrPads[1] + fAttrPads[3]) *
          (fShapeX[4] + fAttrPads[2] + fAttrPads[4]) << ");\n";
      }

      return out.str();
   }

   std::string Generate(std::string OpName) {
      OpName = "op_" + OpName;

      if (fShapeX.empty() || fShapeY.empty()) {
         throw std::runtime_error("TMVA SOFIE Pool Op called to Generate without being initialized first");
      }

      std::stringstream out;

      out << "\n//----  operator " << Name() << "  " << OpName << "\n";
      out << "{\n"; // create a new scope to avoid name clash

      // vectorize the (dilated)convolution kernels into a matrix
      // no need to transpose the matrix
      // size_t hstride = fShapeW[3];
      // size_t hstrideDil = fAttrDilations[0] * fAttrKernelShape[1];  // stride dilated in the height
      // size_t wstrideDil = fAttrDilations[1];
      // size_t dstride = fShapeW[2] * fShapeW[3];
      // size_t dstrideDil = fAttrKernelShape[0] * fAttrKernelShape[1];
      // size_t kstride = fShapeW[1] * fShapeW[2] * fShapeW[3];
      // size_t kstrideDil = fShapeW[1] * dstrideDil;


      assert(fShapeX[0] == fShapeY[0]);
      assert(fShapeX[1] == fShapeY[1]);
      assert(fAttrPads.size() == 6);
      assert(fAttrKernelShape.size() == 3);
      // find lower bounds of filtered area
      int hmin = - fAttrPads[0];   // minimum lower bound value of filter area
      int hmax = fShapeX[2] + fAttrPads[1] - fAttrKernelShape[0] +1;  // maximum lower bound value + 1
      int wmin,wmax,dmin,dmax;

      if(fDim >= 2){
         wmin = - fAttrPads[2];   // minimum lower bound value of filter area
         wmax = fShapeX[3] + fAttrPads[3] - fAttrKernelShape[1] +1;  // maximum lower bound value + 1
      }
      else{
         wmin=1;
         wmax=1;
      }
      if(fDim == 3){
         dmin = - fAttrPads[4];   // minimum lower bound value of filter area
         dmax = fShapeX[4] + fAttrPads[5] - fAttrKernelShape[2] +1;  // maximum lower bound value + 1
      }
      else{
         dmin=1;
         dmax=1;
      }
      out << SP << "constexpr int hsize = " << fShapeX[2] << ";\n";
      out << SP << "constexpr int hmin = " << hmin << ";\n";
      out << SP << "constexpr int hmax = " << hmax << ";\n";
      out << SP << "constexpr int kh = " << fAttrKernelShape[0] << ";\n";
      if (fDim > 1) {
         size_t wsize = fShapeX[3];
         out << SP << "constexpr int wsize = " << wsize << ";\n";
         out << SP << "constexpr int wmin = " << wmin << ";\n";
         out << SP << "constexpr int wmax = " << wmax << ";\n";
         out << SP << "constexpr int kw = " << fAttrKernelShape[1] << ";\n";
      }
      if (fDim > 2) {
         size_t dsize = fShapeX[4];
         out << SP << "constexpr int dsize = " << dsize << ";\n";
         out << SP << "constexpr int dmin = " << dmin << ";\n";
         out << SP << "constexpr int dmax = " << dmax << ";\n";
         out << SP << "constexpr int kd = " << fAttrKernelShape[2] << ";\n";
      }


      bool doPadding = false;
      for ( auto & e : fAttrPads)
         doPadding |= (e > 0);


      if(fDim==1){
         // loop on batches and channels
         out << SP << "size_t outIndex = 0;\n";
         out << SP << "for (size_t n = 0; n < " << fShapeX[0]*fShapeX[1] << "; n++) {\n";
         out << SP << SP << "size_t inputOffset = n*" << fShapeX[2] << ";\n";
         out << SP << SP << "for (int i = hmin; i < hmax; i+=" << fAttrStrides[0] << ") {\n";
         // loop on elements of filter region to compute maximum
         if (fPoolMode == MaxPool)
            out << SP << SP << SP << SP << "float value = -INFINITY;\n";
         else if (fPoolMode == AveragePool) {
            out << SP << SP << SP << SP << "float value = 0;\n";
            if (fAttrCountIncludePad == 0 && doPadding)
               out << SP << SP << SP << SP << "int nsum = 0;\n";
            else // in case we count the pad values in average
               out << SP << SP << SP << SP << "constexpr int nsum = kh;\n";
         }
         // loop on rows of filtered region
         out << SP << SP << SP << SP  << "for (int l = i;  l < i + kh; l++) {\n";
         out << SP << SP << SP << SP  << SP << "if (l < 0 || l >= hsize) continue;\n";
         out << SP << SP << SP << SP  << SP << SP <<  "int index = inputOffset + l;\n";
         if (fPoolMode == MaxPool) {
         out << SP << SP << SP << SP << SP << SP << "auto xval = tensor_" << fNX << "[index];\n";
         out << SP << SP << SP << SP << SP << SP << "if (xval > value) value = xval;\n";
         }
         else if (fPoolMode == AveragePool) {
            // compute sum of values
            out << SP << SP << SP << SP << SP << SP << "value += tensor_" << fNX << "[index];\n";
            if (fAttrCountIncludePad == 0 && doPadding)
               // compute number of elements used for the average
               out << SP << SP << SP << SP << SP << SP << "nsum++;\n";
         }
          out << SP << SP << SP << SP << SP << "}\n"; // end loop on region elements
         if (fPoolMode == AveragePool) {
         // compute average
         out << SP << SP << SP << SP << "value /= float(nsum);\n";
         }

         out << SP << SP << SP << SP << "tensor_" << fNY << "[outIndex++] = value;\n";

         out << SP << SP << "}\n";   // end loop on i (image rows)
         out << SP << "}\n";  // end loop on c*b
      }
      else if(fDim==2){
         // loop on batches and channels
         out << SP << "size_t outIndex = 0;\n";
         out << SP << "for (size_t n = 0; n < " << fShapeX[0]*fShapeX[1] << "; n++) {\n";
         out << SP << SP << "size_t inputOffset = n*" << fShapeX[2]*fShapeX[3] << ";\n";
         out << SP << SP << "for (int i = hmin; i < hmax; i+=" << fAttrStrides[0] << ") {\n";
         out << SP << SP << SP << "for (int j = wmin; j < wmax; j+=" << fAttrStrides[1] << ") {\n";
         // loop on elements of filter region to compute maximum
         if (fPoolMode == MaxPool)
            out << SP << SP << SP << SP << "float value = -INFINITY;\n";
         else if (fPoolMode == AveragePool) {
            out << SP << SP << SP << SP << "float value = 0;\n";
            if (fAttrCountIncludePad == 0 && doPadding)
               out << SP << SP << SP << SP << "int nsum = 0;\n";
            else // in case we count the pad values in average
               out << SP << SP << SP << SP << "constexpr int nsum = kw*kh;\n";
         }
         // loop on rows of filtered region
         out << SP << SP << SP << SP << "for (int l = i;  l < i + kh; l++) {\n";
         out << SP << SP << SP << SP << SP << "if (l < 0 || l >= hsize) continue;\n";
         // loop on columns of filtered region
         out << SP << SP << SP << SP << SP << "for (int m = j; m < j + kw; m++) {\n";
         out << SP << SP << SP << SP << SP << SP << "if (m < 0 || m >= wsize) continue;\n";
         out << SP << SP << SP << SP << SP << SP << SP << SP << "int index = inputOffset + l*hsize + m;\n";
         if (fPoolMode == MaxPool) {
         out << SP << SP << SP << SP << SP << SP << "auto xval = tensor_" << fNX << "[index];\n";
         out << SP << SP << SP << SP << SP << SP << "if (xval > value) value = xval;\n";
         }
         else if (fPoolMode == AveragePool) {
            // compute sum of values
            out << SP << SP << SP << SP << SP << SP << "value += tensor_" << fNX << "[index];\n";
            if (fAttrCountIncludePad == 0 && doPadding)
               // compute number of elements used for the average
               out << SP << SP << SP << SP << SP << SP << "nsum++;\n";
         }
         out << SP << SP << SP << SP << SP << SP << "}\n";
         out << SP << SP << SP << SP << SP << "}\n"; // end loop on region elements
         if (fPoolMode == AveragePool) {
            // compute average
            out << SP << SP << SP << SP << "value /= float(nsum);\n";
         }
         out << SP << SP << SP << SP << "tensor_" << fNY << "[outIndex++] = value;\n";
         out << SP << SP << SP << "}\n";   // end loop on j (columns of image)
         out << SP << SP << "}\n";   // end loop on i (image rows)
         out << SP << "}\n";  // end loop on c*b
      }
      else if(fDim==3){
         // loop on batches and channels
         out << SP << "size_t outIndex = 0;\n";
         out << SP << "for (size_t n = 0; n < " << fShapeX[0]*fShapeX[1] << "; n++) {\n";
         out << SP << SP << "size_t inputOffset = n*" << fShapeX[2]*fShapeX[3]*fShapeX[4] << ";\n";
         out << SP << SP << "for (int i = hmin; i < hmax; i+=" << fAttrStrides[0] << ") {\n";
         out << SP << SP << SP << "for (int j = wmin; j < wmax; j+=" << fAttrStrides[1] << ") {\n";
         out << SP << SP << SP << SP << "for (int k = dmin; k < dmax; k+=" << fAttrStrides[2] << ") {\n";
         // loop on elements of filter region to compute maximum
         if (fPoolMode == MaxPool)
            out << SP << SP << SP << SP << "float value = -INFINITY;\n";
         else if (fPoolMode == AveragePool) {
            out << SP << SP << SP << SP << "float value = 0;\n";
            if (fAttrCountIncludePad == 0 && doPadding)
               out << SP << SP << SP << SP << "int nsum = 0;\n";
            else // in case we count the pad values in average
               out << SP << SP << SP << SP << "constexpr int nsum = kw*kh*kd;\n";
         }
         // loop on rows of filtered region
         out << SP << SP << SP << SP  << "for (int l = i;  l < i + kh; l++) {\n";
         out << SP << SP << SP << SP  << SP << "if (l < 0 || l >= hsize) continue;\n";
         // loop on columns of filtered region
         out << SP << SP << SP << SP << SP << "for (int m = j; m < j + kw; m++) {\n";
         out << SP << SP << SP << SP << SP << SP << "if (m < 0 || m >= wsize) continue;\n";
         // loop on layers of filtered region
         out << SP << SP << SP << SP << SP << SP << "for (int p = k; p < k + kd; p++) {\n";
         out << SP << SP << SP << SP << SP << SP << SP << "if (p < 0 || p >= dsize) continue;\n";
         out << SP << SP << SP << SP << SP << SP << SP << SP << "int index = inputOffset + l*hsize + m*wsize + p;\n";

         if (fPoolMode == MaxPool) {
            out << SP << SP << SP << SP << SP << SP << "auto xval = tensor_" << fNX << "[index];\n";
            out << SP << SP << SP << SP << SP << SP << "if (xval > value) value = xval;\n";
         }
         else if (fPoolMode == AveragePool) {
            // compute sum of values
            out << SP << SP << SP << SP << SP << SP << "value += tensor_" << fNX << "[index];\n";
            if (fAttrCountIncludePad == 0 && doPadding)
               // compute number of elements used for the average
               out << SP << SP << SP << SP << SP << SP << "nsum++;\n";
         }
         out << SP << SP << SP << SP << SP << SP << "}\n";
         out << SP << SP << SP << SP << SP << "}\n";
         out << SP << SP << SP << SP << "}\n"; // end loop on region elements
         if (fPoolMode == AveragePool) {
            // compute average
            out << SP << SP << SP << SP << "value /= float(nsum);\n";
         }

         out << SP << SP << SP << SP << "tensor_" << fNY << "[outIndex++] = value;\n";
         out << SP << SP << SP << SP << "}\n" ; // end loop on k (layers of image)
         out << SP << SP << SP << "}\n";   // end loop on j (columns of image)
         out << SP << SP << "}\n";   // end loop on i (image rows)
         out << SP << "}\n";  // end loop on c*b
      }
      // end scope
      out << SP << "}\n";


      return out.str();
   }
};

} // namespace SOFIE
} // namespace Experimental
} // namespace TMVA


#endif
