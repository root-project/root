// Tests for the matrix routines in the code SOFIE generates. By default, Gemm
// and Axpy are built-in reference implementations of the BLAS routines sgemm_
// and saxpy_; with RModel::SetUseBLAS(true), they call these BLAS routines
// instead. The emitted helper code is compiled with cling here and compared
// with a naive implementation of the BLAS contract: the built-in variant
// always, and the BLAS variant if the test is linked to BLAS
// (SOFIE_TEST_WITH_BLAS).

#include "TMVA/SOFIE_common.hxx"

#include <TInterpreter.h>

#include "gtest/gtest.h"

#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#ifdef SOFIE_TEST_WITH_BLAS
// The BLAS routines. Calling them directly checks the naive implementations
// below, and it makes sure that the BLAS library stays linked to this test, so
// that cling finds them when it compiles the code that uses BLAS.
extern "C" void sgemm_(const char *transa, const char *transb, const int *m, const int *n, const int *k,
                       const float *alpha, const float *A, const int *lda, const float *B, const int *ldb,
                       const float *beta, float *C, const int *ldc);
extern "C" void saxpy_(const int *n, const float *alpha, const float *x, const int *incx, float *y, const int *incy);
#endif

namespace {

using GemmFunc = void (*)(const char *, const char *, const int *, const int *, const int *, const float *,
                          const float *, const int *, const float *, const int *, const float *, float *, const int *);
using AxpyFunc = void (*)(const int *, const float *, const float *, const int *, float *, const int *);

constexpr const char *kGemmType = "void (*)(const char *, const char *, const int *, const int *, const int *, "
                                  "const float *, const float *, const int *, const float *, const int *, "
                                  "const float *, float *, const int *)";
constexpr const char *kAxpyType = "void (*)(const int *, const float *, const float *, const int *, float *, "
                                  "const int *)";

bool IsTransposed(char trans)
{
   return trans == 'T' || trans == 't';
}

// Naive implementation of the BLAS sgemm contract, with A, B and C in
// column-major storage: C := alpha * op(A) * op(B) + beta * C, where C is not
// read if beta == 0.
void NaiveGemm(const char *transa, const char *transb, const int *m, const int *n, const int *k, const float *alpha,
               const float *A, const int *lda, const float *B, const int *ldb, const float *beta, float *C,
               const int *ldc)
{
   for (int j = 0; j < *n; ++j) {
      for (int i = 0; i < *m; ++i) {
         double sum = 0.;
         for (int l = 0; l < *k; ++l) {
            const float a = IsTransposed(*transa) ? A[l + i * *lda] : A[i + l * *lda];
            const float b = IsTransposed(*transb) ? B[j + l * *ldb] : B[l + j * *ldb];
            sum += static_cast<double>(a) * b;
         }
         float &c = C[i + j * *ldc];
         c = static_cast<float>(*alpha * sum + (*beta == 0.f ? 0. : static_cast<double>(*beta) * c));
      }
   }
}

// Naive implementation of the BLAS saxpy contract, for positive strides.
void NaiveAxpy(const int *n, const float *alpha, const float *x, const int *incx, float *y, const int *incy)
{
   for (int i = 0; i < *n; ++i)
      y[i * *incy] += *alpha * x[i * *incx];
}

struct HelperFunctions {
   GemmFunc gemm = nullptr;
   AxpyFunc axpy = nullptr;
};

// Compile the helper code that SOFIE emits into generated models, with or
// without BLAS, and return its Gemm and Axpy.
HelperFunctions DeclareHelpers(bool useBLAS)
{
   using TMVA::Experimental::SOFIE::GenerateHelperFunctionsCode;
   const std::string ns = useBLAS ? "TMVA_SOFIE_BLASTest" : "TMVA_SOFIE_BuiltinTest";
   const auto code = GenerateHelperFunctionsCode({"Gemm", "Axpy"}, ns, useBLAS);
   const std::string src = code.includes + "namespace " + ns + " {\n" + code.definitions + "}\n";
   if (!gInterpreter->Declare(src.c_str()))
      throw std::runtime_error("TestSofieBLAS: failed to declare the emitted SOFIE helper code");

   auto address = [&](const char *type, const char *func) {
      const std::string expr = std::string("static_cast<") + type + ">(&" + ns + "::" + func + ");";
      return gInterpreter->ProcessLine(expr.c_str());
   };
   HelperFunctions helpers;
   helpers.gemm = reinterpret_cast<GemmFunc>(address(kGemmType, "Gemm"));
   helpers.axpy = reinterpret_cast<AxpyFunc>(address(kAxpyType, "Axpy"));
   if (!helpers.gemm || !helpers.axpy)
      throw std::runtime_error("TestSofieBLAS: failed to get the emitted SOFIE helper functions");
   return helpers;
}

const HelperFunctions &BuiltinHelpers()
{
   static const HelperFunctions helpers = DeclareHelpers(false);
   return helpers;
}

#ifdef SOFIE_TEST_WITH_BLAS
const HelperFunctions &BLASHelpers()
{
   static const HelperFunctions helpers = DeclareHelpers(true);
   return helpers;
}
#endif

std::vector<float> RandomVector(std::size_t size, std::mt19937 &rng)
{
   std::uniform_real_distribution<float> dist(-1.f, 1.f);
   std::vector<float> v(size);
   for (auto &x : v)
      x = dist(rng);
   return v;
}

void ExpectClose(const std::vector<float> &actual, const std::vector<float> &expected)
{
   ASSERT_EQ(actual.size(), expected.size());
   for (std::size_t i = 0; i < actual.size(); ++i)
      EXPECT_NEAR(actual[i], expected[i], 1e-4f * (1.f + std::abs(expected[i]))) << "at index " << i;
}

} // namespace

// transA, transB, {m, n, k}, padding of the leading dimensions, alpha, beta
using GemmParams = std::tuple<char, char, std::tuple<int, int, int>, int, float, float>;

class SofieGemm : public testing::TestWithParam<GemmParams> {};

TEST_P(SofieGemm, AgreesWithNaiveImplementation)
{
   // No structured bindings: lambdas can only capture them since C++20.
   const char transA = std::get<0>(GetParam());
   const char transB = std::get<1>(GetParam());
   const int m = std::get<0>(std::get<2>(GetParam()));
   const int n = std::get<1>(std::get<2>(GetParam()));
   const int k = std::get<2>(std::get<2>(GetParam()));
   const int pad = std::get<3>(GetParam());
   const float alpha = std::get<4>(GetParam());
   const float beta = std::get<5>(GetParam());

   // Column-major storage: op(A) is m x k and op(B) is k x n.
   const int lda = (IsTransposed(transA) ? k : m) + pad;
   const int ldb = (IsTransposed(transB) ? n : k) + pad;
   const int ldc = m + pad;

   std::mt19937 rng(m * 10000 + n * 100 + k);
   const auto a = RandomVector(lda * (IsTransposed(transA) ? m : k), rng);
   const auto b = RandomVector(ldb * (IsTransposed(transB) ? k : n), rng);
   auto c = RandomVector(ldc * n, rng);
   // With beta == 0, C has to be overwritten without being read, so whatever
   // it contains must not matter.
   if (beta == 0.f) {
      for (int j = 0; j < n; ++j)
         for (int i = 0; i < m; ++i)
            c[i + j * ldc] = std::numeric_limits<float>::quiet_NaN();
   }

   auto run = [&](GemmFunc gemm) {
      std::vector<float> out = c;
      gemm(&transA, &transB, &m, &n, &k, &alpha, a.data(), &lda, b.data(), &ldb, &beta, out.data(), &ldc);
      return out;
   };

   const auto expected = run(&NaiveGemm);
   ExpectClose(run(BuiltinHelpers().gemm), expected);
#ifdef SOFIE_TEST_WITH_BLAS
   ExpectClose(run(&sgemm_), expected);
   ExpectClose(run(BLASHelpers().gemm), expected);
#endif
}

// Gemm_Call passes the transposition flags in lower case.
INSTANTIATE_TEST_SUITE_P(SofieBLAS, SofieGemm,
                         testing::Combine(testing::Values('N', 'T', 'n', 't'), testing::Values('N', 'T', 'n', 't'),
                                          testing::Values(std::make_tuple(1, 1, 1), std::make_tuple(3, 5, 7),
                                                          std::make_tuple(17, 4, 33), std::make_tuple(64, 1, 128)),
                                          testing::Values(0, 3), testing::Values(1.f, 0.5f),
                                          testing::Values(0.f, 1.f, -0.75f)));

// n, incx, incy, alpha
using AxpyParams = std::tuple<int, int, int, float>;

class SofieAxpy : public testing::TestWithParam<AxpyParams> {};

TEST_P(SofieAxpy, AgreesWithNaiveImplementation)
{
   const int n = std::get<0>(GetParam());
   const int incx = std::get<1>(GetParam());
   const int incy = std::get<2>(GetParam());
   const float alpha = std::get<3>(GetParam());

   std::mt19937 rng(n * 100 + incx * 10 + incy);
   const auto x = RandomVector(n * incx, rng);
   const auto y = RandomVector(n * incy, rng);

   auto run = [&](AxpyFunc axpy) {
      std::vector<float> out = y;
      axpy(&n, &alpha, x.data(), &incx, out.data(), &incy);
      return out;
   };

   const auto expected = run(&NaiveAxpy);
   ExpectClose(run(BuiltinHelpers().axpy), expected);
#ifdef SOFIE_TEST_WITH_BLAS
   ExpectClose(run(&saxpy_), expected);
   ExpectClose(run(BLASHelpers().axpy), expected);
#endif
}

INSTANTIATE_TEST_SUITE_P(SofieBLAS, SofieAxpy,
                         testing::Combine(testing::Values(1, 7, 33), testing::Values(1, 2), testing::Values(1, 3),
                                          testing::Values(1.f, -0.5f)));
