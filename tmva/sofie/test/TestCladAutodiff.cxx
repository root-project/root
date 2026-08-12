constexpr auto modelHeaderSuffix = "_FromONNX_unoptimized.hxx";
constexpr auto modelDataSuffix = "_FromONNX_unoptimized.dat";
#include "test_helpers.h"

#include "gtest/gtest.h"

// Test differentiating a fully-connected neural network with Clad.
// Extension of the ONNX.Linear16 test in TestCustomModelsFromONNX.cxx
TEST(ONNXClad, Linear16)
{
   constexpr float TOLERANCE = DEFAULT_TOLERANCE;

   SofieReference ref = readReference("Linear_16");
   // Mutable copy: the numeric differentiation below perturbs the input values
   std::vector<float> input = ref.f32("input0");

   ASSERT_INCLUDE_AND_RUN(std::vector<float>, "Linear_16", input);

   gInterpreter->Declare(R"(
#include <Math/CladDerivator.h>

float Linear_16_wrapper(TMVA_SOFIE_Linear_16::Session const &session, float const *input)
{
   float out[160]{};
   float output_sum = 0.0;

   TMVA_SOFIE_Linear_16::doInfer(session, input, out);

   for (std::size_t i = 0; i < std::size(out); ++i) {
      output_sum += out[i];
   }
   return output_sum;
}

float Linear_16_outer_wrapper(TMVA_SOFIE_Linear_16::Session const &session, float const *input)
{
   return Linear_16_wrapper(session, input);
}

// Separate evaluation function for the numeric differentiation that
// accumulates in double: with a float accumulator, the rounding error of the
// output sum is comparable to the eps-induced change that the central
// difference measures, so the numeric derivative would be dominated by
// cancellation noise.
double Linear_16_num_eval(TMVA_SOFIE_Linear_16::Session const &session, float const *input)
{
   float out[160]{};
   double output_sum = 0.0;

   TMVA_SOFIE_Linear_16::doInfer(session, input, out);

   for (std::size_t i = 0; i < std::size(out); ++i) {
      output_sum += out[i];
   }
   return output_sum;
}

float Linear_16_wrapper_num_diff(TMVA_SOFIE_Linear_16::Session const &session, float *input, std::size_t i)
{
   const float origVal = input[i];

   const float eps = 1e-3;
   input[i] = origVal - eps;
   double funcValDown = Linear_16_num_eval(session, input);
   input[i] = origVal + eps;
   double funcValUp = Linear_16_num_eval(session, input);
   input[i] = origVal;

   return (funcValUp - funcValDown) / (2 * eps);
}
   )");

   auto inputInterp = toInterpreter(input, "std::vector<float>", true);

   // Why do we have two wrappers, the <>_wrapper and the <>_outer_wrapper?
   // This is because we are not interested in the created gradient function.
   // We are interested in the more low-level *pullback* function, which takes
   // also the data structures for the reverse pass as function arguments. Like
   // this, we can initialize the session for the backward pass once and re-use
   // it. The trick to get the wrapper pullback is to create another wrapper
   // around the wrapper, and creating the gradient for the outer wrapper
   // implicitly creates the pullback for the inner wrapper.
   gInterpreter->ProcessLine("clad::gradient(Linear_16_outer_wrapper, \"input\");");

   // Create two session data structures: one for the forward, and one for the backward pass
   gInterpreter->ProcessLine("TMVA_SOFIE_Linear_16::Session session_linear_16{\"Linear_16_FromONNX.dat\"};");
   gInterpreter->ProcessLine("TMVA_SOFIE_Linear_16::Session _d_session_linear_16{\"Linear_16_FromONNX.dat\"};");

   gInterpreter->ProcessLine("float grad_output[1600]{};");
   gInterpreter->ProcessLine(
      ("Linear_16_wrapper_pullback(session_linear_16, " + inputInterp + ", 1, &_d_session_linear_16, grad_output)")
         .c_str());

   // If you want to see the gradient code:
   // clang-format off
   // gInterpreter->ProcessLine("static_cast<void (*)(TMVA_SOFIE_Linear_16::Session const &, float const *, float *)>(Linear_16_outer_wrapper_grad_1)");
   // gInterpreter->ProcessLine("Linear_16_wrapper_pullback");
   // gInterpreter->ProcessLine("TMVA_SOFIE_Linear_16::doInfer_reverse_forw");
   // gInterpreter->ProcessLine("TMVA_SOFIE_Linear_16::doInfer_pullback");
   // clang-format on

   gInterpreter->ProcessLine((R"(
   float numeric_output[1600]{};
   for (std::size_t i = 0; i < std::size(grad_output); ++i) {
      numeric_output[i] = Linear_16_wrapper_num_diff(session_linear_16, )" +
                              inputInterp + R"(, i);
   }
   )")
                                .c_str());

   double tol = 0.0025;

   auto arr_size = static_cast<std::size_t>(gInterpreter->ProcessLine("std::size(grad_output);"));
   auto grad_arr = reinterpret_cast<float *>(gInterpreter->ProcessLine("grad_output;"));
   auto numeric_arr = reinterpret_cast<float *>(gInterpreter->ProcessLine("numeric_output;"));

   expectNearCapped(grad_arr, numeric_arr, arr_size, tol);

   expectNear(output, ref.f32("output0"), TOLERANCE);
}

// Test forward-mode differentiation and Hessian-vector products of a
// Gemm+Sigmoid network with Clad (a ReLU network like Linear_16 is
// piecewise-linear in its inputs, so its input Hessian vanishes and would not
// validate the second derivatives).
//
// The same inner/outer wrapper trick as in the gradient test above is used:
// differentiating the outer wrapper generates the inner wrapper's
// *pushforward*, which takes the session tangent as a function argument. The
// top-level derivative functions that clad generates are not usable directly,
// because they default-construct the Session tangent instead of zeroing its
// weights (the weights are constants): hence Session::SetWeightsToZero.
TEST(ONNXClad, LinearWithSigmoidForwardAndHessian)
{
   constexpr float TOLERANCE = DEFAULT_TOLERANCE;
   constexpr int nInputs = 48; // input tensor shape { 2, 24 }; the output shape is { 2, 12 }

   SofieReference ref = readReference("LinearWithSigmoid");
   std::vector<float> input = ref.f32("input0");
   ASSERT_EQ(input.size(), nInputs);

   ASSERT_INCLUDE_AND_RUN(std::vector<float>, "LinearWithSigmoid", input);
   expectNear(output, ref.f32("output0"), TOLERANCE);

   gInterpreter->Declare(R"(
#include <Math/CladDerivator.h>

float sig_w(TMVA_SOFIE_LinearWithSigmoid::Session const &session, float const *input)
{
   float out[24]{};
   float output_sum = 0.0;

   TMVA_SOFIE_LinearWithSigmoid::doInfer(session, input, out);

   for (std::size_t i = 0; i < std::size(out); ++i) {
      output_sum += out[i];
   }
   return output_sum;
}

float sig_outer(TMVA_SOFIE_LinearWithSigmoid::Session const &session, float const *input)
{
   return sig_w(session, input);
}
   )");

   // Generate sig_w_pullback (reverse mode) and sig_w_pushforward (forward mode).
   gInterpreter->ProcessLine("clad::gradient(sig_outer, \"input\");");
   gInterpreter->ProcessLine("clad::differentiate(sig_outer, \"input[0]\");");

   gInterpreter->Declare(R"(
// Directional derivative of sig_w via the generated pushforward. Reverse
// differentiation of this function gives the exact Hessian-vector product
// H * dinput (in the input adjoint) and the gradient (in the dinput adjoint).
// Both pointer arguments must be requested as active: for a non-varied
// argument clad would look up the custom Gemm_Call pullback with a
// reduced signature, not find it, and silently fall back to differentiating
// the BLAS call, which yields zero.
float sig_dir(TMVA_SOFIE_LinearWithSigmoid::Session const &session,
              TMVA_SOFIE_LinearWithSigmoid::Session const &zeroSession, float const *input, float const *dinput)
{
   return sig_w_pushforward(session, input, zeroSession, dinput).pushforward;
}

float sig_dir_outer(TMVA_SOFIE_LinearWithSigmoid::Session const &session,
                    TMVA_SOFIE_LinearWithSigmoid::Session const &zeroSession, float const *input, float const *dinput)
{
   return sig_dir(session, zeroSession, input, dinput);
}
   )");

   gInterpreter->ProcessLine("clad::gradient(sig_dir_outer, \"input, dinput\");");

   auto inputInterp = toInterpreter(input, "std::vector<float>", true);

   // Sessions for the forward pass, the first-order reverse-pass adjoint
   // (its weight values are irrelevant: adjoints are accumulated and
   // discarded), and the zero-weight tangent session.
   gInterpreter->Declare(R"(
TMVA_SOFIE_LinearWithSigmoid::Session sig_session{"LinearWithSigmoid_FromONNX_unoptimized.dat"};
TMVA_SOFIE_LinearWithSigmoid::Session sig_d_session{"LinearWithSigmoid_FromONNX_unoptimized.dat"};
TMVA_SOFIE_LinearWithSigmoid::Session sig_zero_session{"LinearWithSigmoid_FromONNX_unoptimized.dat"};

std::vector<float> sig_grad(48, 0.0f);
std::vector<float> sig_fwd(48, 0.0f);
std::vector<float> sig_dinput(48, 0.0f);
std::vector<float> sig_hvp_grad(48, 0.0f);
std::vector<float> sig_hess(48 * 48, 0.0f);
std::vector<float> sig_hess_fd(48 * 48, 0.0f);
float sig_hvp_dev = 0.0f;
   )");
   gInterpreter->ProcessLine("sig_zero_session.SetWeightsToZero();");

   // Reverse-mode gradient (the reference for the forward mode below).
   gInterpreter->ProcessLine(
      ("sig_w_pullback(sig_session, " + inputInterp + ", 1, &sig_d_session, sig_grad.data());").c_str());

   // Forward mode: directional derivatives along all unit directions.
   gInterpreter->ProcessLine((R"(
   for (int j = 0; j < 48; ++j) {
      std::fill(sig_dinput.begin(), sig_dinput.end(), 0.0f);
      sig_dinput[j] = 1.0f;
      sig_fwd[j] = sig_w_pushforward(sig_session, )" +
                              inputInterp + R"(, sig_zero_session, sig_dinput.data()).pushforward;
   }
   )")
                                .c_str());

   auto *gradArr = reinterpret_cast<float *>(gInterpreter->ProcessLine("sig_grad.data();"));
   auto *fwdArr = reinterpret_cast<float *>(gInterpreter->ProcessLine("sig_fwd.data();"));
   for (int i = 0; i < nInputs; ++i) {
      EXPECT_NEAR(fwdArr[i], gradArr[i], 1e-5) << "forward vs reverse derivative at input index " << i;
   }

   // Hessian, column by column, as Hessian-vector products with unit
   // directions. The adjoint sessions must be freshly constructed for every
   // call: the clad-generated second-order pullback does not restore the
   // intermediate adjoint state inside the adjoint sessions to zero (see the
   // "clad referenced '_tracker...' before its declaration" warnings it
   // prints during generation), so reusing them leaks state from one call
   // into the next. They are constructed from the weight file each time
   // because the generated Session is not copy-safe: the raw tensor_*
   // pointer members of a copy would still point into the original's
   // buffers.
   //
   // The adjoint of the tangent direction is the gradient again, for every
   // direction, so the loop also tracks the largest deviation from the
   // reverse-mode gradient as a cheap consistency check of the second-order
   // code path.
   gInterpreter->ProcessLine((R"(
   for (int j = 0; j < 48; ++j) {
      TMVA_SOFIE_LinearWithSigmoid::Session dSession{"LinearWithSigmoid_FromONNX_unoptimized.dat"};
      TMVA_SOFIE_LinearWithSigmoid::Session dSession2{"LinearWithSigmoid_FromONNX_unoptimized.dat"};
      std::fill(sig_dinput.begin(), sig_dinput.end(), 0.0f);
      std::fill(sig_hvp_grad.begin(), sig_hvp_grad.end(), 0.0f);
      sig_dinput[j] = 1.0f;
      std::vector<float> hvp(48, 0.0f);
      sig_dir_pullback(sig_session, sig_zero_session, )" +
                              inputInterp + R"(, sig_dinput.data(), 1, &dSession, &dSession2, hvp.data(),
                       sig_hvp_grad.data());
      for (int i = 0; i < 48; ++i) {
         sig_hess[i * 48 + j] = hvp[i];
         sig_hvp_dev = std::max(sig_hvp_dev, std::abs(sig_hvp_grad[i] - sig_grad[i]));
      }
   }
   )")
                                .c_str());

   auto *hvpDev = reinterpret_cast<float *>(gInterpreter->ProcessLine("&sig_hvp_dev;"));
   EXPECT_LT(*hvpDev, 1e-5) << "dinput adjoint vs gradient";

   // Reference Hessian: central finite differences of the exact reverse-mode
   // gradient.
   gInterpreter->Declare(R"(
void sig_fd_hessian(float *input)
{
   const float eps = 1e-2f;
   std::vector<float> gp(48);
   std::vector<float> gm(48);
   for (int j = 0; j < 48; ++j) {
      // Fresh adjoint sessions, so the FD reference does not rely on the
      // first-order pullback consuming-and-zeroing the adjoint state left
      // behind by previous calls (see the Hessian loop above for why copies
      // are not an option).
      TMVA_SOFIE_LinearWithSigmoid::Session dp{"LinearWithSigmoid_FromONNX_unoptimized.dat"};
      TMVA_SOFIE_LinearWithSigmoid::Session dm{"LinearWithSigmoid_FromONNX_unoptimized.dat"};
      const float orig = input[j];
      std::fill(gp.begin(), gp.end(), 0.0f);
      std::fill(gm.begin(), gm.end(), 0.0f);
      input[j] = orig + eps;
      sig_w_pullback(sig_session, input, 1, &dp, gp.data());
      input[j] = orig - eps;
      sig_w_pullback(sig_session, input, 1, &dm, gm.data());
      input[j] = orig;
      for (int i = 0; i < 48; ++i) {
         sig_hess_fd[i * 48 + j] = (gp[i] - gm[i]) / (2 * eps);
      }
   }
}
   )");
   gInterpreter->ProcessLine(("sig_fd_hessian(" + inputInterp + ");").c_str());

   auto *hessArr = reinterpret_cast<float *>(gInterpreter->ProcessLine("sig_hess.data();"));
   auto *hessFdArr = reinterpret_cast<float *>(gInterpreter->ProcessLine("sig_hess_fd.data();"));

   expectNearCapped(hessArr, hessFdArr, nInputs * nInputs, 1e-3);

   // Guard against a trivially-zero Hessian, which would make the check above
   // meaningless.
   double maxAbsHess = 0.;
   for (int i = 0; i < nInputs * nInputs; ++i) {
      maxAbsHess = std::max(maxAbsHess, std::abs(static_cast<double>(hessArr[i])));
   }
   EXPECT_GT(maxAbsHess, 1e-4);

   // The Hessian-vector products must assemble into a symmetric matrix.
   for (int i = 0; i < nInputs; ++i) {
      for (int j = 0; j < i; ++j) {
         EXPECT_NEAR(hessArr[i * 48 + j], hessArr[j * 48 + i], 1e-5)
            << "Hessian asymmetry at (" << i << ", " << j << ")";
      }
   }
}
