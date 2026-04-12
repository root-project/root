constexpr auto modelHeaderSuffix = "_FromONNX_unoptimized.hxx";
constexpr auto modelDataSuffix = "_FromONNX_unoptimized.dat";
#include "test_helpers.h"

#include "input_models/references/Linear_16.ref.hxx"

#include "gtest/gtest.h"

// Test differentiating a fully-connected neural network with Clad.
// Extension of the ONNX.Linear16 test in TestCustomModelsFromONNX.cxx
TEST(ONNXClad, Linear16)
{
   constexpr float TOLERANCE = DEFAULT_TOLERANCE;

   // Preparing the standard all-ones input
   std::vector<float> input(1600);
   std::fill_n(input.data(), input.size(), 1.0f);

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

float Linear_16_wrapper_num_diff(TMVA_SOFIE_Linear_16::Session const &session, float *input, std::size_t i)
{
   const float origVal = input[i];

   const float eps = 1e-3;
   input[i] = origVal - eps;
   float funcValDown = Linear_16_wrapper(session, input);
   input[i] = origVal + eps;
   float funcValUp = Linear_16_wrapper(session, input);
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

   constexpr std::size_t kMaxPrint = 10;
   std::size_t mismatchCount = 0;

   for (std::size_t i = 0; i < arr_size; ++i) {
      double diff = std::abs(grad_arr[i] - numeric_arr[i]);

      if (diff > tol) {
         if (mismatchCount < kMaxPrint) {
            ADD_FAILURE() << "Mismatch at index " << i << " analytic=" << grad_arr[i] << " numeric=" << numeric_arr[i]
                          << " diff=" << diff;
         }
         ++mismatchCount;
      }
   }

   if (mismatchCount > kMaxPrint) {
      ADD_FAILURE() << "Further mismatches suppressed (total mismatches: " << mismatchCount << ")";
   }

   // Checking output size
   EXPECT_EQ(output.size(), sizeof(Linear_16_ExpectedOutput::all_ones) / sizeof(float));

   float *correct = Linear_16_ExpectedOutput::all_ones;

   // Checking every output value, one by one
   for (size_t i = 0; i < output.size(); ++i) {
      EXPECT_LE(std::abs(output[i] - correct[i]), TOLERANCE);
   }
}
