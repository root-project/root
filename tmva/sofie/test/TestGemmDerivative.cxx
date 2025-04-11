#include <TInterpreter.h>
#include <TInterpreterValue.h>

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
   int m;
   int n;
   int k;
};

void validator(float *variables, int m, int n, int k, float *target)
{
   for (int i = 0; i < m * k + k * n; ++i) {
      target[i] = 0;
      if (i == 0) {
         // Check derivative with alpha
         int _a;
         int _b;
         //  float _prod;
         for (int _m = 1; _m < m + 1; ++_m) {
            for (int _n = 1; _n < n + 1; ++n) {
               for (int _k = 1; _k < k + 1; ++k) {
                  _a = (_m - 1) * m + (_k - 1) % k;
                  _b = (_k - 1) * k + (_n - 1) % n;
                  //   _prod = variables[_a + 1] * variables[_b + m * k + 1];
                  target[i] += variables[_a + 1] * variables[_b + m * k + 1];
               }
            }
         }
      } else if ((i - 1) < m * k) {
         // Get current position in m, k
         int _m = (i - 1) % m + 1;
         int _k = ((i - 1) - _m + 1) / m + 1;

         for (int _n = 1; _n < n + 1; ++_n) {
            int index = m * k + (_k - 1) % k + (_n - 1) * k;
            target[i] += variables[index + 1];
         }
      } else {
         int _k = ((i - 1) - m * k) % k + 1;

         for (int _m = 1; _m < m + 1; ++_m) {
            int index = (_m - 1) % m + (_k - 1) * m;
            target[i] += variables[index + 1];
         }
      }
   }
}

TEST_P(GemmTest, GemmTestDerivative)
{
   float variables[] = {3., 1., 4., 2., 2., 4., 0., 6., 1., 2., 5., 7., 3., 1.};
   float grad_output[] = {0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.};
   float target[] = {0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.};

   static bool declared = false;
   if (declared == false) {
      gInterpreter->Declare(R"cpp(
            #include <Math/CladDerivator.h>

            float gemm_function(float *variables, int m, int n, int k) {
                // variable is assumed to pack Amk and Bkn,
                // so it's of length m*k + k*n

                float *matA = &variables[0];
                float *matB = &variables[m*k];

                constexpr int n_out = 12; // Allocate a maximum length of arrays here
                assert (m*n <= n_out);

                float output[n_out];

                TMVA::Experimental::SOFIE::Gemm_Call(output, false, false, m, n, k, 1, matA, matB, 0, nullptr);

                float ret = 0;
                for (int i = 0; i < m*n; ++i) {
                    ret += output[i];
                }
                return ret;
            }

            #pragma clad ON
            void clad_request () {
                clad::gradient(gemm_function, "variables");
            }
            #pragma clad OFF
        )cpp");

      declared = true;
   }

   auto *gradient_ptr = reinterpret_cast<void (*)(float *, int, int, int, float *)>(
      gInterpreter->ProcessLine("static_cast<void (*)(float*, int, int, int, float*)>(gemm_function_grad_0);"));

   gradient_ptr(variables, m, n, k, grad_output);
   validator(variables, m, n, k, target);

   for (int i = 0; i < m * k + k * n; ++i) {
      EXPECT_EQ(grad_output[i], target[i]);
   }
};

INSTANTIATE_TEST_SUITE_P(CladDerivator, GemmTest,
                         testing::Values(std::make_tuple(1, 1, 1), std::make_tuple(1, 2, 1), std::make_tuple(2, 1, 1),
                                         std::make_tuple(2, 1, 2), std::make_tuple(2, 3, 2), std::make_tuple(4, 3, 2)));
