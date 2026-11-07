/// \file SOFIE_common_helpers.cxx
/// Standalone definitions of the SOFIE inference helpers (Im2col, Gemm_Call, ...).
/// RModel records which helpers a model needs (RModel::AddNeededHelperFunction)
/// and dumps only those into the generated namespace, so the emitted header is
/// self-contained (no TMVA/SOFIE_common.hxx include). These are dependency-free
/// copies of the SOFIE_common.hxx originals: keep them in sync when those change.

#include "TMVA/SOFIE_common.hxx"

namespace TMVA {
namespace Experimental {
namespace SOFIE {

namespace {

// Standalone helper snippets, each emitted verbatim inside the generated model
// namespace. They depend only on the C++ stdlib (headers collected in
// GenerateHelperFunctionsCode) and, for Gemm, on the local BLAS::sgemm_ below.

// extern "C" declaration of the BLAS routine Gemm_Call needs, in a nested BLAS
// namespace. Skipped when the caller already declared sgemm_ (see the
// sgemmAlreadyDeclared parameter of GenerateHelperFunctionsCode).
constexpr const char *kBlasSgemm = R"SOFIE(
namespace BLAS {
extern "C" void sgemm_(const char *transa, const char *transb, const int *m, const int *n, const int *k,
                       const float *alpha, const float *A, const int *lda, const float *B, const int *ldb,
                       const float *beta, float *C, const int *ldc);
} // namespace BLAS
)SOFIE";

constexpr const char *kConvertShapeToLength = R"SOFIE(
inline std::size_t ConvertShapeToLength(const std::vector<std::size_t> &shape)
{
   std::size_t length = 1;
   for (auto &dim : shape)
      length *= dim;
   return length;
}
)SOFIE";

constexpr const char *kConvertShapeToString = R"SOFIE(
inline std::string ConvertShapeToString(const std::vector<std::size_t> &shape)
{
   std::stringstream out;
   out << "{ ";
   for (std::size_t i = 0; i < shape.size(); i++) {
      out << shape[i];
      if (i < shape.size() - 1)
         out << " , ";
   }
   out << " }";
   return out.str();
}
)SOFIE";

// Branchless bounds check `0 <= a < b`: casting to unsigned collapses it to a
// single comparison (a negative `a` wraps to a large value that fails `< b`).
constexpr const char *kIsAGeZero = R"SOFIE(
inline bool is_a_ge_zero_and_a_lt_b(int a, int b)
{
   return static_cast<unsigned>(a) < static_cast<unsigned>(b);
}
)SOFIE";

// im2col: re-arrange convolution input into a matrix usable directly by BLAS.
// It loops over each element of the filtered region first, following the input
// layout, so reads/writes stay consecutive in memory; the result is already
// transposed -- a (channels*kernel_h*kernel_w , output_h*output_w) matrix.
// Example: input   a1 a2 a3
//                  b1 b2 b3   with a 2x2 kernel (k1,k2,k3,k4) and padding 1
//                  c1 c2 c3
//   gives a 4x16 matrix, output-ordered (all elements for k1, then k2, ...):
//     ( 0  0  0  0  0  a1 a2 a3 0  b1 b2 b3  0 c1 c2 c3 )   k1
//     ( 0  0  0  0  a1 a2 a3  0 b1 b2 b3  0 c1 c2 c3  0 )   k2
//     ( 0  a1 a2 a3 0  b1 b2 b3 0  c1 c2 c3  0  0  0  0 )   k3
//     ( a1 a2 a3 0  b1 b2 b3  0 c1 c2 c3  0  0  0  0  0 )   k4
// Per-axis begin/end padding can differ (ONNX "pads" attribute, and odd total
// padding from the SAME_UPPER / SAME_LOWER autopad modes).
constexpr const char *kIm2col = R"SOFIE(
template <typename T>
void Im2col(const T *data_im, const int channels, const int height, const int width, const int kernel_h,
            const int kernel_w, const int pad_h_begin, const int pad_h_end, const int pad_w_begin,
            const int pad_w_end, const int stride_h, const int stride_w,
            const int dilation_h, const int dilation_w, T *data_col)
{
   const int output_h = (height + pad_h_begin + pad_h_end - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
   const int output_w = (width + pad_w_begin + pad_w_end - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
   const int channel_size = height * width;
   for (int channel = channels; channel--; data_im += channel_size) {
      for (int kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
         for (int kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
            int input_row = -pad_h_begin + kernel_row * dilation_h;
            for (int output_rows = output_h; output_rows; output_rows--) {
               if (!is_a_ge_zero_and_a_lt_b(input_row, height)) {
                  for (int output_cols = output_w; output_cols; output_cols--) {
                     *(data_col++) = 0;
                  }
               } else {
                  int input_col = -pad_w_begin + kernel_col * dilation_w;
                  for (int output_col = output_w; output_col; output_col--) {
                     if (is_a_ge_zero_and_a_lt_b(input_col, width)) {
                        *(data_col++) = data_im[input_row * width + input_col];
                     } else {
                        *(data_col++) = 0;
                     }
                     input_col += stride_w;
                  }
               }
               input_row += stride_h;
            }
         }
      }
   }
}
)SOFIE";

constexpr const char *kIm2col3d = R"SOFIE(
template <typename T>
void Im2col_3d(const T *data_im, const int channels,
               const int depth, const int height, const int width,
               const int kernel_d, const int kernel_h, const int kernel_w,
               const int pad_d_begin, const int pad_d_end, const int pad_h_begin, const int pad_h_end,
               const int pad_w_begin, const int pad_w_end,
               const int stride_d, const int stride_h, const int stride_w,
               const int dilation_d, const int dilation_h, const int dilation_w, T *data_col)
{
   const int output_h = (height + pad_h_begin + pad_h_end - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
   const int output_w = (width + pad_w_begin + pad_w_end - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
   const int output_d = (depth + pad_d_begin + pad_d_end - (dilation_d * (kernel_d - 1) + 1)) / stride_d + 1;
   const int channel_size = height * width * depth;
   for (int channel = channels; channel--; data_im += channel_size) {
      for (int kernel_depth = 0; kernel_depth < kernel_d; kernel_depth++) {
         for (int kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
            for (int kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
               int input_dep = -pad_d_begin + kernel_depth * dilation_d;
               for (int output_dep = output_d; output_dep; output_dep--) {
                  if (!is_a_ge_zero_and_a_lt_b(input_dep, depth)) {
                     for (int output_rows = output_h; output_rows; output_rows--) {
                        for (int output_cols = output_w; output_cols; output_cols--) {
                           *(data_col++) = 0;
                        }
                     }
                  } else {
                     int input_row = -pad_h_begin + kernel_row * dilation_h;
                     for (int output_rows = output_h; output_rows; output_rows--) {
                        if (!is_a_ge_zero_and_a_lt_b(input_row, height)) {
                           for (int output_cols = output_w; output_cols; output_cols--) {
                              *(data_col++) = 0;
                           }
                        } else {
                           int input_col = -pad_w_begin + kernel_col * dilation_w;
                           for (int output_col = output_w; output_col; output_col--) {
                              if (is_a_ge_zero_and_a_lt_b(input_col, width)) {
                                 *(data_col++) = data_im[input_dep * width * height + input_row * width + input_col];
                              } else {
                                 *(data_col++) = 0;
                              }
                              input_col += stride_w;
                           }
                        }
                        input_row += stride_h;
                     }
                  }
                  input_dep += stride_d;
               }
            }
         }
      }
   }
}
)SOFIE";

constexpr const char *kCol2im = R"SOFIE(
template <typename Dtype>
void col2im(const Dtype *data_col, const int channels,
            const int height, const int width, const int kernel_h, const int kernel_w,
            const int pad_h_begin, const int pad_h_end, const int pad_w_begin, const int pad_w_end,
            const int stride_h, const int stride_w,
            const int dilation_h, const int dilation_w,
            Dtype *data_im)
{
   // output must start zeroed: col2im scatters with += so overlapping columns accumulate
   std::fill(data_im, data_im + height * width * channels, 0.);
   const int output_h = (height + pad_h_begin + pad_h_end - (dilation_h * (kernel_h - 1) + 1)) / stride_h + 1;
   const int output_w = (width + pad_w_begin + pad_w_end - (dilation_w * (kernel_w - 1) + 1)) / stride_w + 1;
   const int channel_size = height * width;
   for (int channel = channels; channel--; data_im += channel_size) {
      for (int kernel_row = 0; kernel_row < kernel_h; kernel_row++) {
         for (int kernel_col = 0; kernel_col < kernel_w; kernel_col++) {
            int input_row = -pad_h_begin + kernel_row * dilation_h;
            for (int output_rows = output_h; output_rows; output_rows--) {
               if (!is_a_ge_zero_and_a_lt_b(input_row, height)) {
                  data_col += output_w;
               } else {
                  int input_col = -pad_w_begin + kernel_col * dilation_w;
                  for (int output_col = output_w; output_col; output_col--) {
                     if (is_a_ge_zero_and_a_lt_b(input_col, width)) {
                        data_im[input_row * width + input_col] += *data_col;
                     }
                     data_col++;
                     input_col += stride_w;
                  }
               }
               input_row += stride_h;
            }
         }
      }
   }
}
)SOFIE";

// Broadcast helpers implementing numpy broadcasting rules (see
// https://numpy.org/doc/stable/user/basics.broadcasting.html and
// https://github.com/onnx/onnx/blob/main/docs/Broadcasting.md). Unidirectional
// broadcast: only the input shape is stretched to targetShape, not vice versa.
// These are rewritten with respect to TMVA/SOFIE_common.hxx to avoid the
// dependency on std::span (which would require C++20 or ROOT's RSpan.hxx in the
// generated code): the input is passed as a raw pointer plus a length instead.
constexpr const char *kBroadcastTensor = R"SOFIE(
template <typename T>
void BroadcastTensor(const T *data, std::size_t curLength, const std::vector<std::size_t> &shape,
                     const std::vector<std::size_t> &targetShape, T *broadcastedData)
{
   std::size_t size = shape.size();
   if (size > 1 && shape.front() == targetShape.front() && shape.back() == 1) {
      std::size_t bsize = targetShape.back();
      for (int k = int(size) - 2; k >= 0; k--) {
         if (shape[k] != 1)
            break;
         bsize *= targetShape[k];
      }
      for (std::size_t i = 0; i < curLength; i++) {
         std::fill(broadcastedData + i * bsize, broadcastedData + (i + 1) * bsize, data[i]);
      }
      return;
   }

   std::copy(data, data + curLength, broadcastedData);
   std::size_t arrayNum = 1;
   std::vector<T> newData(ConvertShapeToLength(targetShape));

   for (std::size_t idx = 0; idx < size; idx++) {
      std::size_t dim = shape[idx];
      std::size_t targetDim = targetShape[idx];
      if (dim == 1 && targetDim > 1) {
         std::size_t newLength = curLength * targetDim;
         std::size_t arrayLength = curLength / arrayNum;
         if (arrayLength > 1) {
            for (std::size_t arrayIdx = 0; arrayIdx < arrayNum; arrayIdx++) {
               for (std::size_t targetIdx = 0; targetIdx < targetDim; targetIdx++) {
                  std::size_t offset = arrayIdx * arrayLength * targetDim + targetIdx * arrayLength;
                  std::copy(broadcastedData + arrayIdx * arrayLength,
                            broadcastedData + (arrayIdx + 1) * arrayLength,
                            newData.begin() + offset);
               }
            }
         } else {
            for (std::size_t arrayIdx = 0; arrayIdx < arrayNum; arrayIdx++) {
               std::fill(newData.begin() + arrayIdx * targetDim,
                         newData.begin() + (arrayIdx + 1) * targetDim, broadcastedData[arrayIdx]);
            }
         }
         curLength = newLength;
         std::copy(newData.begin(), newData.begin() + newLength, broadcastedData);
      }
      arrayNum *= targetDim;
   }
}

template <typename T>
T *CreateBroadcastTensor(const T *data, const std::vector<std::size_t> &shape,
                         const std::vector<std::size_t> &targetShape, std::size_t targetLength)
{
   T *broadcastedData = new T[targetLength];
   std::size_t curLength = ConvertShapeToLength(shape);
   BroadcastTensor<T>(data, curLength, shape, targetShape, broadcastedData);
   return broadcastedData;
}

template <typename T>
T *UnidirectionalBroadcast(const T *data, const std::vector<std::size_t> &shape,
                           const std::vector<std::size_t> &targetShape)
{
   if (shape.size() < targetShape.size()) {
      std::size_t targetSize = targetShape.size();
      std::vector<std::size_t> newShape(targetSize, 1);
      std::size_t offset = targetSize - shape.size();
      std::copy(shape.begin(), shape.end(), newShape.begin() + offset);
      return CreateBroadcastTensor(data, newShape, targetShape, ConvertShapeToLength(targetShape));
   }
   return CreateBroadcastTensor(data, shape, targetShape, ConvertShapeToLength(targetShape));
}

template <typename T>
void UnidirectionalBroadcast(const T *data, const std::vector<std::size_t> &shape,
                             const std::vector<std::size_t> &targetShape, T *broadcastedData)
{
   std::size_t curLength = ConvertShapeToLength(shape);
   if (shape.size() < targetShape.size()) {
      std::size_t targetSize = targetShape.size();
      std::vector<std::size_t> newShape(targetSize, 1);
      std::size_t offset = targetSize - shape.size();
      std::copy(shape.begin(), shape.end(), newShape.begin() + offset);
      BroadcastTensor(data, curLength, newShape, targetShape, broadcastedData);
      return;
   }
   BroadcastTensor(data, curLength, shape, targetShape, broadcastedData);
}
)SOFIE";

constexpr const char *kBroadcastConvBias = R"SOFIE(
template <typename T>
T *BroadcastConvBias(const T *data, const std::size_t channel, const std::vector<std::size_t> &targetShape)
{
   std::size_t size = targetShape.size();
   if (targetShape[1] != channel) {
      std::stringstream ss;
      ss << "TMVA::SOFIE - Error broadcasting Conv Bias of shape {";
      ss << std::to_string(channel);
      ss << "} to ";
      ss << ConvertShapeToString(targetShape);
      throw std::runtime_error(ss.str());
   }

   std::size_t targetLength = ConvertShapeToLength(targetShape);
   T *newData = new T[targetLength];

   if (targetLength == channel) {
      std::copy(data, data + channel, newData);
      return newData;
   }

   std::size_t cStride = 1;
   for (std::size_t i = 2; i < size; i++)
      cStride *= targetShape[i];
   for (std::size_t i = 0; i < channel; i++) {
      std::fill(newData + i * cStride, newData + (i + 1) * cStride, data[i]);
   }
   std::size_t batch = targetShape[0];
   std::size_t bStride = channel * cStride;
   for (std::size_t i = 1; i < batch; i++) {
      std::copy(newData, newData + bStride, newData + i * bStride);
   }
   return newData;
}
)SOFIE";

constexpr const char *kGemmCall = R"SOFIE(
inline void Gemm_Call(float *output, bool transa, bool transb, int m, int n, int k, float alpha, const float *A,
                      const float *B, float beta, const float *C)
{
   char ct = 't';
   char cn = 'n';
   const int *lda = transa ? &k : &m;
   const int *ldb = transb ? &n : &k;
   const int *ldc = &m;
   if (C != nullptr) {
      std::copy(C, C + m * n, output);
   }
   BLAS::sgemm_(transa ? &ct : &cn, transb ? &ct : &cn, &m, &n, &k, &alpha, A, lda, B, ldb, &beta, output, ldc);
}
)SOFIE";

// Custom Clad reverse-mode pullbacks for the helpers (they used to live in
// Math/CladDerivator.h). Gemm_Call and Copy need a hand-written pullback because
// their bodies call bodyless routines (sgemm_, std::copy -> memmove) that Clad
// cannot differentiate; Fill and Relu are included for completeness so a
// differentiated model needs neither SOFIE_common.hxx nor CladDerivator.h. How
// they are placed and found is explained at the emission site
// (GenerateHelperFunctionsCode).
constexpr const char *kGemmCallPullback = R"SOFIE(
inline void Gemm_Call_pullback(float *output, bool transa, bool transb, int m, int n, int k, float alpha,
                               const float *A, const float *B, float beta, const float *C, float *_d_output, bool *,
                               bool *, int *, int *, int *, float *_d_alpha, float *_d_A, float *_d_B, float *_d_beta,
                               float *_d_C)
{
   // TODO:
   //    - fix and test the implementation for alpha != 1.0
   if (alpha != 1.0f) {
      return;
   }

   // beta needs to be one because we want to add to _d_A and _d_B instead of
   // overwriting it.
   float one = 1.;

   // ---- dA ----
   if (!transa) {
      // dA += dY * op(B)^T
      Gemm_Call(_d_A, false, !transb, m, k, n, one, _d_output, B, one, _d_A);
   } else {
      // dA += op(B) * dY^T
      Gemm_Call(_d_A, transb, true, k, m, n, one, B, _d_output, one, _d_A);
   }

   // ---- dB ----
   if (!transb) {
      // dB += op(A)^T * dY
      Gemm_Call(_d_B, !transa, false, k, n, m, one, A, _d_output, one, _d_B);
   } else {
      // dB += dY^T * op(A)
      Gemm_Call(_d_B, true, transa, n, k, m, one, _d_output, A, one, _d_B);
   }

   int sizeC = n * m;

   for (int i = 0; i < sizeC; ++i) {
      if (C) {
         *_d_alpha += _d_output[i] * (output[i] - beta * C[i]);
         *_d_beta += _d_output[i] * C[i];
      } else {
         *_d_alpha += _d_output[i] * output[i];
      }
      if (_d_C)
         _d_C[i] += _d_output[i] * beta;
   }
}
)SOFIE";

// Pullback for the Copy helper. Copy's body uses std::copy, which lowers to the
// bodyless __builtin_memmove that Clad cannot differentiate, so a hand-written
// pullback is required. The generated Copy is a template; this float overload
// matches its float instantiation (the only one used by inference code).
constexpr const char *kCopyPullback = R"SOFIE(
inline void Copy_pullback(float *output, const float *input, int size, float *_d_output, float *_d_input, int *)
{
   for (int i = 0; i < size; i++) {
      output[i] = input[i];
      _d_input[i] += _d_output[i];
      _d_output[i] = 0.F;
   }
}
)SOFIE";

// Pullback for the Fill helper (std::fill -> bodyless builtin).
constexpr const char *kFillPullback = R"SOFIE(
inline void Fill_pullback(float *output, float value, int size, float *_d_output, float *_d_value, int *)
{
   for (int i = 0; i < size; i++) {
      output[i] = value;
      *_d_value += _d_output[i];
      _d_output[i] = 0.F;
   }
}
)SOFIE";

// Pullback for the Relu helper. Relu's body is differentiable by Clad on its
// own, but providing the pullback keeps the derivative identical to the one
// previously supplied by Math/CladDerivator.h.
constexpr const char *kReluPullback = R"SOFIE(
inline void Relu_pullback(float *output, const float *input, int size, float *_d_output, float *_d_input, int *)
{
   for (int i = 0; i < size; i++) {
      output[i] = input[i] > 0.F ? input[i] : 0.F;
      float _r_d0 = _d_output[i];
      _d_output[i] = 0.F;
      if (input[i] > 0.F)
         _d_input[i] += _r_d0;
   }
}
)SOFIE";

// Custom Clad forward-mode pushforwards for the same helpers, following
// clad's convention for void functions: the original parameters followed by
// same-type tangent clones of every parameter. Clad computes Hessians as
// reverse-mode derivatives of forward-mode code, so the bodies below are
// themselves reverse-differentiated and must consist of plain loops and of
// calls with custom pullbacks only (for the namespace alias on the
// Gemm_Call calls, see the comment at the emission site). Note that
// Gemm_Call_pullback silently bails out for alpha != 1 (see the TODO there),
// so Hessians inherit that restriction.
constexpr const char *kGemmCallPushforward = R"SOFIE(
inline void Gemm_Call_pushforward(float *output, bool transa, bool transb, int m, int n, int k, float alpha,
                                  const float *A, const float *B, float beta, const float *C, float *_d_output, bool,
                                  bool, int, int, int, float _d_alpha, const float *_d_A, const float *_d_B,
                                  float _d_beta, const float *_d_C)
{
   // Primal: output = alpha * op(A) op(B) + beta * C, where a null C means
   // "accumulate onto the existing output" (see Gemm_Call).
   // Tangent:
   //   d_output = alpha * (op(dA) op(B) + op(A) op(dB)) + beta * dC
   //            + d_alpha * op(A) op(B) + d_beta * C
   // with (C, dC) read as (output, d_output) in accumulate mode. The tangent
   // is therefore computed first, while output and d_output still hold the
   // values the primal call overwrites. The pointer null-checks are hoisted
   // into bools: with a pointer-typed ternary condition, Clad's reverse pass
   // (as of v2.4) hoists and tapes the condition with the const qualifier
   // dropped, generating code that does not compile.
   const bool hasC = C != nullptr;
   const bool hasdC = _d_C != nullptr;
   SOFIE_MODEL_NS::Gemm_Call(_d_output, transa, transb, m, n, k, alpha, _d_A, B, (hasC && !hasdC) ? 0.0f : beta, _d_C);
   SOFIE_MODEL_NS::Gemm_Call(_d_output, transa, transb, m, n, k, alpha, A, _d_B, 1.0f, nullptr);
   if (_d_alpha != 0.0f) {
      SOFIE_MODEL_NS::Gemm_Call(_d_output, transa, transb, m, n, k, _d_alpha, A, B, 1.0f, nullptr);
   }
   if (_d_beta != 0.0f) {
      if (hasC) {
         for (int i = 0; i < m * n; ++i) {
            _d_output[i] += _d_beta * C[i];
         }
      } else {
         for (int i = 0; i < m * n; ++i) {
            _d_output[i] += _d_beta * output[i];
         }
      }
   }
   SOFIE_MODEL_NS::Gemm_Call(output, transa, transb, m, n, k, alpha, A, B, beta, C);
}
)SOFIE";

constexpr const char *kCopyPushforward = R"SOFIE(
inline void Copy_pushforward(float *output, const float *input, int size, float *_d_output, const float *_d_input,
                             int)
{
   for (int i = 0; i < size; i++) {
      output[i] = input[i];
      _d_output[i] = _d_input[i];
   }
}
)SOFIE";

constexpr const char *kFillPushforward = R"SOFIE(
inline void Fill_pushforward(float *output, float value, int size, float *_d_output, float _d_value, int)
{
   for (int i = 0; i < size; i++) {
      output[i] = value;
      _d_output[i] = _d_value;
   }
}
)SOFIE";

constexpr const char *kReluPushforward = R"SOFIE(
inline void Relu_pushforward(float *output, float const *input, int size, float *_d_output, float const *_d_input,
                             int)
{
   // Tangent first: the generated code applies Relu in place, so input/output
   // (and their tangents) may alias.
   for (int i = 0; i < size; i++) {
      _d_output[i] = (input[i] > 0.0f) ? _d_input[i] : 0.0f;
      output[i] = (input[i] > 0.0f) ? input[i] : 0.0f;
   }
}
)SOFIE";

constexpr const char *kRelu = R"SOFIE(
inline void Relu(float *output, float const *input, int size)
{
   for (int i = 0; i < size; i++) {
      output[i] = (input[i] > 0.0f) ? input[i] : 0.0f;
   }
}
)SOFIE";

constexpr const char *kFill = R"SOFIE(
inline void Fill(float *output, float value, int size)
{
   std::fill(output, output + size, value);
}
)SOFIE";

constexpr const char *kCopy = R"SOFIE(
template <class T>
inline void Copy(T *output, T const *input, int size)
{
   std::copy(input, input + size, output);
}
)SOFIE";

constexpr const char *kReadTensorFromStream = R"SOFIE(
inline float ParseFloatToken(const std::string &s)
{
   if (s == "inf")
      return std::numeric_limits<float>::infinity();
   if (s == "-inf")
      return -std::numeric_limits<float>::infinity();
   if (s == "nan")
      return std::numeric_limits<float>::quiet_NaN();
   return std::stof(s);
}

template <class T>
void ReadTensorFromStream(std::istream &is, T &target, std::string const &expectedName, std::size_t expectedLength)
{
   std::string name;
   std::size_t length;
   is >> name >> length;
   if (name != expectedName) {
      std::string err_msg =
         "TMVA-SOFIE failed to read the correct tensor name; expected name is " + expectedName + " , read " + name;
      throw std::runtime_error(err_msg);
   }
   if (length != expectedLength) {
      std::string err_msg = "TMVA-SOFIE failed to read the correct tensor size; expected size is " +
                            std::to_string(expectedLength) + " , read " + std::to_string(length);
      throw std::runtime_error(err_msg);
   }
   std::string token;
   for (std::size_t i = 0; i < length; ++i) {
      is >> token;
      target[i] = ParseFloatToken(token);
   }
   if (is.fail()) {
      throw std::runtime_error("TMVA-SOFIE failed to read the values for tensor " + expectedName);
   }
}
)SOFIE";

// Writer and reader support for the safetensors weight format
// (https://huggingface.co/docs/safetensors): an 8-byte little-endian JSON
// header size, the JSON header mapping tensor names to
// {"dtype", "shape", "data_offsets"}, then the concatenated raw little-endian
// payloads. data_offsets are relative to the start of the payload area, i.e.
// just after the JSON header.
// The header is scanned with a small self-contained streaming parser (no JSON
// value tree is built), keeping the generated header free of any third-party
// dependency.

// Non-owning view over a safetensors weight payload, either loaded from a
// file or provided directly from memory.
constexpr const char *kSafetensorsBlob = R"SOFIE(
// Non-owning view over a safetensors weight payload, either loaded from a
// file or provided directly from memory. The payload is little-endian per the
// safetensors specification.
struct SafetensorsBlob {
   const char *data = nullptr;
   std::size_t size = 0;
   constexpr SafetensorsBlob() = default;
   constexpr SafetensorsBlob(const char *d, std::size_t n) : data(d), size(n) {}
   SafetensorsBlob(const std::string &s) : data(s.data()), size(s.size()) {}
};
)SOFIE";

constexpr const char *kSafetensorsReader = R"SOFIE(
// Bookkeeping of one tensor inside a safetensors blob
struct SafetensorsTensorInfo {
   std::uint64_t fBegin = 0; // payload offset of the tensor in the blob
   std::uint64_t fEnd = 0;
   std::string fDtype;
};

// Scanner for the JSON header of a safetensors blob. The header structure (a
// flat object of {"dtype", "shape", "data_offsets"} tensor entries plus an
// optional "__metadata__" member) is fixed, so the header is streamed straight
// into the tensor index instead of being parsed into a JSON value tree.
// Content that is not needed ("shape", "__metadata__", unknown members) is
// still scanned completely, so malformed headers are rejected rather than
// silently skipped.
class SafetensorsHeaderScanner {
public:
   SafetensorsHeaderScanner(std::string_view text, std::uint64_t dataBegin, std::uint64_t payloadBytes)
      : fText(text), fDataBegin(dataBegin), fPayloadBytes(payloadBytes)
   {
   }

   void Scan(std::unordered_map<std::string, SafetensorsTensorInfo> &tensors)
   {
      SkipWhitespace();
      Expect('{');
      SkipWhitespace();
      if (!TryConsume('}')) {
         while (true) {
            SkipWhitespace();
            if (Peek() != '"')
               Fail("expected tensor name");
            const std::string name = ParseString(true);
            SkipWhitespace();
            Expect(':');
            SkipWhitespace();
            if (name == "__metadata__") {
               SkipValue();
            } else {
               ParseTensorEntry(name, tensors);
            }
            SkipWhitespace();
            if (TryConsume('}'))
               break;
            Expect(',');
         }
      }
      SkipWhitespace();
      if (fPos != fText.size())
         Fail("unexpected trailing content");
   }

private:
   std::string_view fText;
   std::size_t fPos = 0;
   int fDepth = 0; // recursion budget for skipped content
   std::uint64_t fDataBegin;    // absolute offset of the payload area in the blob
   std::uint64_t fPayloadBytes; // size of the payload area

   [[noreturn]] void Fail(const std::string &message)
   {
      throw std::runtime_error("tmva-sofie: malformed JSON in safetensors header at offset " +
                               std::to_string(fPos) + " : " + message);
   }

   void SkipWhitespace()
   {
      while (fPos < fText.size()) {
         const char c = fText[fPos];
         if (c == ' ' || c == '\t' || c == '\n' || c == '\r')
            ++fPos;
         else
            break;
      }
   }

   char Peek()
   {
      if (fPos >= fText.size())
         Fail("unexpected end of input");
      return fText[fPos];
   }

   void Expect(char c)
   {
      if (Peek() != c) {
         std::string msg = "expected '";
         msg += c;
         msg += "'";
         Fail(msg);
      }
      ++fPos;
   }

   bool TryConsume(char c)
   {
      if (fPos < fText.size() && fText[fPos] == c) {
         ++fPos;
         return true;
      }
      return false;
   }

   void ExpectLiteral(const char *literal)
   {
      const std::size_t len = std::string_view(literal).size();
      if (fText.compare(fPos, len, literal) != 0)
         Fail(std::string("expected '") + literal + "'");
      fPos += len;
   }

   // JSON string. Decoding only matters for the strings that are kept (tensor
   // names and dtypes); those are ASCII in practice, so \u escapes above 0x7F
   // are rejected rather than decoded (such a name could never match the
   // tensor names the generated code looks up). In skip mode only the scan
   // position matters.
   std::string ParseString(bool keep)
   {
      Expect('"');
      std::string out;
      while (true) {
         if (fPos >= fText.size())
            Fail("unterminated string");
         const char c = fText[fPos++];
         if (c == '"')
            return out;
         if (c != '\\') {
            if (static_cast<unsigned char>(c) < 0x20)
               Fail("unescaped control character in string");
            if (keep)
               out.push_back(c);
            continue;
         }
         if (fPos >= fText.size())
            Fail("truncated escape sequence");
         const char esc = fText[fPos++];
         std::uint32_t cp;
         switch (esc) {
         case '"': cp = '"'; break;
         case '\\': cp = '\\'; break;
         case '/': cp = '/'; break;
         case 'b': cp = '\b'; break;
         case 'f': cp = '\f'; break;
         case 'n': cp = '\n'; break;
         case 'r': cp = '\r'; break;
         case 't': cp = '\t'; break;
         case 'u': cp = ParseHexEscape(); break;
         default: Fail("invalid escape sequence");
         }
         if (keep) {
            if (cp >= 0x80)
               Fail("unsupported non-ASCII escape in tensor name");
            out.push_back(static_cast<char>(cp));
         }
      }
   }

   std::uint32_t ParseHexEscape()
   {
      if (fPos + 4 > fText.size())
         Fail("truncated \\u escape");
      std::uint32_t cp = 0;
      for (int i = 0; i < 4; ++i) {
         const char c = fText[fPos++];
         cp <<= 4;
         if (c >= '0' && c <= '9')
            cp |= static_cast<std::uint32_t>(c - '0');
         else if (c >= 'a' && c <= 'f')
            cp |= static_cast<std::uint32_t>(c - 'a' + 10);
         else if (c >= 'A' && c <= 'F')
            cp |= static_cast<std::uint32_t>(c - 'A' + 10);
         else
            Fail("invalid hex digit in \\u escape");
      }
      return cp;
   }

   // Non-negative decimal integer; the safetensors specification fixes the
   // data_offsets grammar to plain integers.
   std::uint64_t ParseUInt()
   {
      if (fPos >= fText.size() || fText[fPos] < '0' || fText[fPos] > '9')
         Fail("expected a non-negative integer");
      std::uint64_t v = 0;
      while (fPos < fText.size() && fText[fPos] >= '0' && fText[fPos] <= '9') {
         const std::uint64_t d = std::uint64_t(fText[fPos] - '0');
         if (v > (~std::uint64_t(0) - d) / 10)
            Fail("integer out of range");
         v = 10 * v + d;
         ++fPos;
      }
      if (fPos < fText.size() && (fText[fPos] == '.' || fText[fPos] == 'e' || fText[fPos] == 'E'))
         Fail("expected a non-negative integer");
      return v;
   }

   void ParseTensorEntry(const std::string &name, std::unordered_map<std::string, SafetensorsTensorInfo> &tensors)
   {
      Expect('{');
      SkipWhitespace();
      std::string dtype;
      std::uint64_t begin = 0;
      std::uint64_t end = 0;
      bool haveDtype = false;
      bool haveOffsets = false;
      if (!TryConsume('}')) {
         while (true) {
            SkipWhitespace();
            if (Peek() != '"')
               Fail("expected member name in tensor entry");
            const std::string member = ParseString(true);
            SkipWhitespace();
            Expect(':');
            SkipWhitespace();
            if (member == "dtype") {
               dtype = ParseString(true);
               haveDtype = true;
            } else if (member == "data_offsets") {
               Expect('[');
               SkipWhitespace();
               begin = ParseUInt();
               SkipWhitespace();
               Expect(',');
               SkipWhitespace();
               end = ParseUInt();
               SkipWhitespace();
               Expect(']');
               haveOffsets = true;
            } else {
               SkipValue();
            }
            SkipWhitespace();
            if (TryConsume('}'))
               break;
            Expect(',');
         }
      }
      if (!haveDtype || !haveOffsets)
         throw std::runtime_error("tmva-sofie: invalid entry for tensor " + name + " in safetensors blob");
      // Validate against the payload area before computing blob offsets, so
      // that the additions below cannot wrap around.
      if (begin > end || end > fPayloadBytes)
         throw std::runtime_error("tmva-sofie: invalid data offsets for tensor " + name + " in safetensors blob");
      tensors.emplace(name, SafetensorsTensorInfo{fDataBegin + begin, fDataBegin + end, dtype});
   }

   void SkipNumber()
   {
      // the value of skipped content does not matter, only the scan position
      const std::size_t begin = fPos;
      while (fPos < fText.size()) {
         const char c = fText[fPos];
         if ((c >= '0' && c <= '9') || c == '-' || c == '+' || c == '.' || c == 'e' || c == 'E')
            ++fPos;
         else
            break;
      }
      if (fPos == begin)
         Fail("unexpected character");
   }

   void SkipValue()
   {
      // A real safetensors header nests at most ~4 levels deep; the depth
      // budget stops deeply nested malformed headers from exhausting the stack
      if (++fDepth > 64)
         Fail("JSON nested too deeply");
      SkipWhitespace();
      const char c = Peek();
      if (c == '{') {
         ++fPos;
         SkipWhitespace();
         if (!TryConsume('}')) {
            while (true) {
               SkipWhitespace();
               if (Peek() != '"')
                  Fail("expected object key");
               ParseString(false);
               SkipWhitespace();
               Expect(':');
               SkipValue();
               SkipWhitespace();
               if (TryConsume('}'))
                  break;
               Expect(',');
            }
         }
      } else if (c == '[') {
         ++fPos;
         SkipWhitespace();
         if (!TryConsume(']')) {
            while (true) {
               SkipValue();
               SkipWhitespace();
               if (TryConsume(']'))
                  break;
               Expect(',');
            }
         }
      } else if (c == '"') {
         ParseString(false);
      } else if (c == 't') {
         ExpectLiteral("true");
      } else if (c == 'f') {
         ExpectLiteral("false");
      } else if (c == 'n') {
         ExpectLiteral("null");
      } else {
         SkipNumber();
      }
      --fDepth;
   }
};

class SafetensorsReader {
public:
   explicit SafetensorsReader(SafetensorsBlob blob) : fData(blob.data), fSize(blob.size)
   {
      // safetensors payloads are little-endian by specification
      const std::uint16_t one = 1;
      if (!*reinterpret_cast<const std::uint8_t *>(&one))
         throw std::runtime_error("tmva-sofie: safetensors weights can only be read on a little-endian host");

      if (fSize < 8)
         throw std::runtime_error("tmva-sofie: truncated safetensors blob: missing JSON header size");
      std::uint64_t headerSize = 0;
      for (int i = 0; i < 8; ++i)
         headerSize |= std::uint64_t(static_cast<unsigned char>(fData[i])) << (8 * i);
      // safeguard against absurd allocations on a corrupted payload
      if (headerSize > (std::uint64_t(1) << 30))
         throw std::runtime_error("tmva-sofie: invalid JSON header size in safetensors blob");
      if (headerSize > fSize - 8)
         throw std::runtime_error("tmva-sofie: truncated JSON header in safetensors blob");
      const std::uint64_t dataBegin = 8 + headerSize;
      SafetensorsHeaderScanner(std::string_view(fData + 8, headerSize), dataBegin, fSize - dataBegin).Scan(fTensors);
   }

   // Copy the payload of tensor `name` into `target`, checking data type and
   // size against what the generated code expects
   template <class T>
   void Read(const std::string &name, std::vector<T> &target, std::size_t expectedLength,
             const std::string &expectedDtype)
   {
      auto it = fTensors.find(name);
      if (it == fTensors.end())
         throw std::runtime_error("tmva-sofie: tensor " + name + " not found in safetensors blob");
      const SafetensorsTensorInfo &info = it->second;
      if (info.fDtype != expectedDtype)
         throw std::runtime_error("tmva-sofie: tensor " + name + " in safetensors blob has dtype " + info.fDtype +
                                  " , expected " + expectedDtype);
      const std::uint64_t nbytes = info.fEnd - info.fBegin;
      if (nbytes != expectedLength * sizeof(T))
         throw std::runtime_error("tmva-sofie: tensor " + name + " in safetensors blob has " +
                                  std::to_string(nbytes) + " bytes , expected " +
                                  std::to_string(expectedLength * sizeof(T)));
      target.resize(expectedLength);
      std::memcpy(target.data(), fData + info.fBegin, nbytes);
   }

private:
   const char *fData;
   std::size_t fSize;
   std::unordered_map<std::string, SafetensorsTensorInfo> fTensors;
};
)SOFIE";

// Constexpr helpers carrying static/symbolic shape metadata for the model's
// input tensors into the emitted code.
constexpr const char *kInputTensorDims = R"SOFIE(
struct SingleDim {
   enum class Kind { Static, Symbolic };
   Kind kind;
   std::size_t dim;
   std::string_view name;
   constexpr SingleDim(std::size_t v) : kind(Kind::Static), dim(v), name() {}
   constexpr SingleDim(const char *v) : kind(Kind::Symbolic), dim(0), name(v) {}
};

struct TensorDims {
   const SingleDim *data;
   std::size_t size;
   constexpr std::size_t total_size() const
   {
      std::size_t result = 1;
      for (std::size_t i = 0; i < size; ++i) {
         result *= data[i].dim;
      }
      return result;
   }
};

template <class Arr>
constexpr TensorDims makeDims(Arr const &arr)
{
   return TensorDims{arr.data(), arr.size()};
}
)SOFIE";

constexpr const char *kDynamicMemory = R"SOFIE(
struct TensorLifeInfo {
   int begin;        // start time (operator index) of the tensor's lifetime
   int end;          // end time (operator index)
   std::size_t size; // size in bytes
};

struct MemoryResult {
   std::size_t total_bytes = 0;      // total memory needed
   std::vector<std::size_t> offsets; // resulting offset for each tensor
};

namespace memory_detail {
struct FreeBlock {
   std::size_t offset;
   std::size_t size;
   // order by offset for deterministic coalescing
   bool operator<(const FreeBlock &other) const { return offset < other.offset; }
};
struct MemoryEvent {
   int t;    // time (operator index)
   int type; // 0 = END, 1 = START
   int idx;  // tensor index
   bool operator<(const MemoryEvent &o) const
   {
      if (t != o.t)
         return t < o.t;
      return type < o.type; // END before START at the same time
   }
};
} // namespace memory_detail

// Greedy best-fit planner with a coalescing free list.
inline MemoryResult OrganizeMemory(const std::vector<TensorLifeInfo> &tensorsInfo)
{
   using memory_detail::FreeBlock;
   using memory_detail::MemoryEvent;
   for (const auto &t : tensorsInfo) {
      if (!(t.end > t.begin)) {
         throw std::runtime_error("Each tensor must have end > begin.");
      }
   }

   std::vector<MemoryEvent> events;
   events.reserve(tensorsInfo.size() * 2);
   for (int i = 0; i < (int)tensorsInfo.size(); ++i) {
      events.push_back({tensorsInfo[i].end, 0, i});
      events.push_back({tensorsInfo[i].begin, 1, i});
   }
   std::sort(events.begin(), events.end());

   std::vector<std::size_t> tensorsOffset(tensorsInfo.size());
   std::set<FreeBlock> free_list;
   std::unordered_map<int, std::size_t> live_size;
   std::unordered_map<int, std::size_t> live_offset;
   std::size_t total_bytes = 0;

   auto allocate_best_fit = [&](std::size_t need) -> std::size_t {
      // Smallest free block with size >= need. free_list is ordered by offset, so
      // this scans linearly; for very large tensor sets a size-keyed multimap
      // would avoid the O(n) scan.
      auto best = free_list.end();
      for (auto it = free_list.begin(); it != free_list.end(); ++it) {
         if (it->size >= need) {
            if (best == free_list.end() || it->size < best->size)
               best = it;
         }
      }
      if (best != free_list.end()) {
         std::size_t off = best->offset;
         if (best->size == need) {
            free_list.erase(best);
         } else {
            FreeBlock updated{best->offset + need, best->size - need};
            free_list.erase(best);
            free_list.insert(updated);
         }
         return off;
      }
      std::size_t off = total_bytes;
      total_bytes += need;
      return off;
   };

   auto try_coalesce = [&](std::set<FreeBlock>::iterator it) {
      if (it != free_list.begin()) {
         auto prev = std::prev(it);
         if (prev->offset + prev->size == it->offset) {
            FreeBlock merged{prev->offset, prev->size + it->size};
            free_list.erase(prev);
            it = free_list.erase(it);
            it = free_list.insert(merged).first;
         }
      }
      auto next = std::next(it);
      if (next != free_list.end() && it->offset + it->size == next->offset) {
         FreeBlock merged{it->offset, it->size + next->size};
         free_list.erase(next);
         it = free_list.erase(it);
         free_list.insert(merged);
      }
   };

   for (const auto &e : events) {
      if (e.type == 0) {
         auto it_sz = live_size.find(e.idx);
         auto it_off = live_offset.find(e.idx);
         if (it_sz != live_size.end() && it_off != live_offset.end()) {
            FreeBlock fb{it_off->second, it_sz->second};
            auto it = free_list.insert(fb).first;
            try_coalesce(it);
            live_size.erase(it_sz);
            live_offset.erase(it_off);
         }
      } else {
         auto &t = tensorsInfo[e.idx];
         std::size_t off = allocate_best_fit(t.size);
         tensorsOffset[e.idx] = off;
         live_size[e.idx] = t.size;
         live_offset[e.idx] = off;
      }
   }

   return MemoryResult{total_bytes, std::move(tensorsOffset)};
}
)SOFIE";

} // anonymous namespace

HelperFunctionsCode GenerateHelperFunctionsCode(const std::set<std::string> &neededHelpers,
                                                const std::string &modelNamespace, bool sgemmAlreadyDeclared)
{
   auto need = [&](const char *key) { return neededHelpers.count(key) > 0; };

   const bool im2col = need("Im2col");
   const bool im2col3d = need("Im2col_3d");
   const bool col2im = need("col2im");
   const bool uniBroadcast = need("UnidirectionalBroadcast");
   const bool convBias = need("BroadcastConvBias");
   const bool gemm = need("Gemm_Call");
   const bool relu = need("Relu");
   const bool fill = need("Fill");
   const bool copy = need("Copy");
   const bool readTensor = need("ReadTensorFromStream");
   const bool safetensorsBlob = need("SafetensorsBlob");
   const bool readSafetensors = need("SafetensorsReader");
   const bool inputDims = need("InputTensorDims");
   const bool dynMemory = need("DynamicMemory");

   const bool im2colFamily = im2col || im2col3d || col2im;
   const bool needConvertLength = uniBroadcast || convBias;
   const bool needConvertString = convBias;

   // ---- collect the required standard headers -----------------------------
   std::set<std::string> stdHeaders;
   auto addStd = [&](std::initializer_list<const char *> hs) {
      for (auto h : hs)
         stdHeaders.insert(h);
   };

   if (im2colFamily || uniBroadcast || convBias || gemm || fill || copy || dynMemory)
      addStd({"algorithm"});
   if (needConvertLength || uniBroadcast || dynMemory)
      addStd({"vector", "cstddef"});
   if (needConvertString || convBias)
      addStd({"sstream", "string", "stdexcept"});
   if (readTensor)
      addStd({"string", "istream", "stdexcept", "limits"});
   if (safetensorsBlob)
      addStd({"cstddef", "string"});
   if (readSafetensors)
      addStd({"cstdint", "cstring", "string", "string_view", "unordered_map", "stdexcept", "vector"});
   if (inputDims)
      addStd({"array", "string_view", "cstddef"});
   if (dynMemory)
      addStd({"set", "unordered_map", "stdexcept", "iterator"});

   std::string includes;
   for (auto const &h : stdHeaders)
      includes += "#include <" + h + ">\n";

   // ---- assemble the definitions ------------------------------------------
   // The order matters: a definition must precede any non-dependent use of it.
   std::string defs;
   defs += "\n// --- Standalone SOFIE inference helper functions ---\n";

   // sgemm_ declaration for Gemm_Call, unless the caller already emitted one.
   if (gemm && !sgemmAlreadyDeclared)
      defs += kBlasSgemm;

   if (needConvertLength)
      defs += kConvertShapeToLength;
   if (needConvertString)
      defs += kConvertShapeToString;

   if (im2colFamily || uniBroadcast || convBias) {
      defs += "\nnamespace UTILITY {\n";
      if (im2colFamily)
         defs += kIsAGeZero;
      if (im2col)
         defs += kIm2col;
      if (im2col3d)
         defs += kIm2col3d;
      if (col2im)
         defs += kCol2im;
      if (uniBroadcast)
         defs += kBroadcastTensor;
      if (convBias)
         defs += kBroadcastConvBias;
      defs += "} // namespace UTILITY\n";
   }

   if (gemm)
      defs += kGemmCall;
   if (relu)
      defs += kRelu;
   if (fill)
      defs += kFill;
   if (copy)
      defs += kCopy;
   if (readTensor)
      defs += kReadTensorFromStream;
   if (safetensorsBlob)
      defs += kSafetensorsBlob;
   if (readSafetensors)
      defs += kSafetensorsReader;
   if (inputDims)
      defs += kInputTensorDims;
   if (dynMemory)
      defs += kDynamicMemory;

   defs += "// --- End of SOFIE inference helper functions ---\n\n";

   // ---- Clad custom derivatives (pullbacks and pushforwards) --------------
   // Some helpers cannot be differentiated automatically by Clad (Gemm_Call
   // calls the bodyless BLAS routine sgemm_). For those we emit a hand-written
   // pullback (reverse mode) and pushforward (forward mode). Clad looks up
   // custom derivatives in
   // clad::custom_derivatives::<function-namespace>, so they are placed
   // there (mirroring the generated model namespace) rather than next to the
   // function. The definitions reference the model's own helpers via a using
   // declaration / namespace alias, so no Clad header is pulled in and a user
   // who never differentiates the model simply carries unused inline
   // functions.
   std::string cladDefs;
   if (gemm || copy || fill || relu) {
      cladDefs += "\nnamespace clad {\nnamespace custom_derivatives {\nnamespace " + modelNamespace + " {\n";
      if (gemm) {
         // Gemm_Call_pullback calls Gemm_Call, so bring it into scope. The
         // pushforward uses the SOFIE_MODEL_NS alias instead: its body gets
         // reverse-differentiated by Clad for Hessians, and Clad resolves the
         // custom Gemm_Call_pullback for the inner calls only when they do
         // not go through a using-declaration.
         cladDefs += "using ::" + modelNamespace + "::Gemm_Call;\n";
         cladDefs += "namespace SOFIE_MODEL_NS = ::" + modelNamespace + ";\n";
         cladDefs += kGemmCallPullback;
         cladDefs += kGemmCallPushforward;
      }
      if (copy) {
         cladDefs += kCopyPullback;
         cladDefs += kCopyPushforward;
      }
      if (fill) {
         cladDefs += kFillPullback;
         cladDefs += kFillPushforward;
      }
      if (relu) {
         cladDefs += kReluPullback;
         cladDefs += kReluPushforward;
      }
      cladDefs += "} // namespace " + modelNamespace + "\n} // namespace custom_derivatives\n} // namespace clad\n";
   }

   return HelperFunctionsCode{std::move(includes), std::move(defs), std::move(cladDefs)};
}

} // namespace SOFIE
} // namespace Experimental
} // namespace TMVA
