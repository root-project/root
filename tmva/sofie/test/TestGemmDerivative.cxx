#include <TInterpreter.h>

#include "gtest/gtest.h"

#include <stdexcept>

class GemmTest : public testing::TestWithParam<std::tuple<int, int, int>> {
public:
   void SetUp() override
   {
      m = std::get<0>(GetParam());
      n = std::get<1>(GetParam());
      k = std::get<2>(GetParam());
   }

protected:
   int m = 0;
   int n = 0;
   int k = 0;
};

void validator(float *variables, int m, int n, int k, float *target)
{
   // Offset iteration by number of parameters before matA
   int offset = 2;

   // Check derivative with respect to alpha
   for (int _m = 1; _m < m + 1; ++_m) {
      for (int _n = 1; _n < n + 1; ++_n) {
         for (int _k = 1; _k < k + 1; ++_k) {
            int _a_idx = (_k - 1) * m + (_m - 1) % m;
            int _b_idx = (_n - 1) * k + (_k - 1) % k;
            // int _c_idx = (_n - 1) * m + (_m - 1) % m;

            target[0] += variables[0] * variables[_a_idx + offset] * variables[_b_idx + m * k + offset];
         }
      }
   }

   // Check derivative with respect to beta
   // target[1] = 0;
   for (int i = offset + m * k + k * n; i < offset + m * k + k * n + m * n; ++i) {
      target[1] += variables[i];
   }

   // Check derivatives with respect to matA, matB, matC
   for (int i = offset; i < m * k + k * n + m * n + offset; ++i) {
      target[i] = 0;
      if ((i - offset) < m * k) {
         // Check derivative with respect to matA
         int _k = ((i - offset) - (i - offset) % m + 1) / m + 1;
         for (int _n = 1; _n < n + 1; ++_n) {
            int index = m * k + (_k - 1) % k + (_n - 1) * k;
            target[i] += variables[index + offset];
         }
      } else if ((i - offset) < m * k + k * n) {
         // Check derivative with respect to matB
         int _k = ((i - offset) - m * k) % k + 1;
         for (int _m = 1; _m < m + 1; ++_m) {
            int index = (_m - 1) % m + (_k - 1) * m;
            target[i] += variables[index + offset];
         }
      } else {
         // Check derivative with respect to matC
         target[i] = variables[1];
      }
   }
}

TEST_P(GemmTest, GemmTestDerivative)
{
   static bool declared = false;
   if (declared == false) {
      gInterpreter->Declare(R"cpp(
            #include <Math/CladDerivator.h>

            float gemm_function(float *variables, int m, int n, int k) {
                // variable is assumed to pack Amk and Bkn,
                // so it's of length m*k + k*n

                float alpha = variables[0];
                float beta  = variables[1];
                float *matA = variables + 2;
                float *matB = variables + m*k + 2;
                float *matC = variables + m*k + k*n + 2;

                constexpr int n_out = 24; // Allocate a maximum length of arrays here
                assert (m*n+1 <= n_out);

                float output[n_out];

                TMVA::Experimental::SOFIE::Gemm_Call(output, false, false, m, n, k, alpha, matA, matB, beta, matC);

                float ret = 0;
                for (int i = 0; i < m*n; ++i) {
                    ret += output[i];
                }
                return ret;
            }

            #pragma clad ON
            void clad_request() {
               clad::gradient(gemm_function, "variables");
            }
            #pragma clad OFF
        )cpp");

      declared = true;
   }

   auto *gradient_ptr = reinterpret_cast<void (*)(float *, int, int, int, float *)>(
      gInterpreter->ProcessLine("static_cast<void (*)(float*, int, int, int, float*)>(gemm_function_grad_0);"));

   float variables[] = {1., 1., 4., 2., 2., 4., 0., 6., 1., 2., 5., 7., 3., 1.,
                        4., 2., 3., 1., 1., 2., 0., 9., 3., 4., 1., 2., 6., 3.};
   float grad_output[] = {0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                          0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.};
   float target[] = {0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                     0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.};

   gradient_ptr(variables, m, n, k, grad_output);
   validator(variables, m, n, k, target);

   for (int i = 0; i < 2 + m * k + k * n; ++i) {
      EXPECT_EQ(grad_output[i], target[i]);
   }
};

INSTANTIATE_TEST_SUITE_P(CladDerivator, GemmTest,
                         testing::Values(std::make_tuple(1, 1, 1), std::make_tuple(1, 2, 1), std::make_tuple(2, 1, 1),
                                         std::make_tuple(2, 1, 2), std::make_tuple(2, 3, 2), std::make_tuple(4, 3, 2)));
