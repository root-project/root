// test finding Roots of polynomials

#include "gtest/gtest.h"

#include "Math/Polynomial.h"
#include <vector>
#include <complex>
#include <algorithm>

using namespace std;
using namespace std::complex_literals;

namespace {

enum EModeType { kAnalytical = 1, kNumerical = 2 };

void checkResult(std::vector<complex<double>> &result, std::vector<complex<double>> const &expectedResult,
                 double tolerance)
{
   // sort first result
   auto smallerComplex = [&](complex<double> c1, complex<double> c2) {
      double diff = std::abs(c1.real() - c2.real());
      // in case of numerical roots difference can be small
      // if (c1.real() != c2.real())
      if (diff > std::max(tolerance, 4 * std::numeric_limits<double>::epsilon()))
         return c1.real() < c2.real();
      else
         return c1.imag() < c2.imag();
   };
   std::sort(result.begin(), result.end(), smallerComplex);

   ASSERT_EQ(result.size(), expectedResult.size());
   ASSERT_EQ(result.size(), 4);

   // unrolll loop to have test printing component when failed
   if (tolerance <= 0) {
      // in this case check with 4 ulp
      EXPECT_DOUBLE_EQ(result[0].real(), expectedResult[0].real());
      EXPECT_DOUBLE_EQ(result[0].imag(), expectedResult[0].imag());

      EXPECT_DOUBLE_EQ(result[1].real(), expectedResult[1].real());
      EXPECT_DOUBLE_EQ(result[1].imag(), expectedResult[1].imag());

      EXPECT_DOUBLE_EQ(result[2].real(), expectedResult[2].real());
      EXPECT_DOUBLE_EQ(result[2].imag(), expectedResult[2].imag());

      EXPECT_DOUBLE_EQ(result[3].real(), expectedResult[3].real());
      EXPECT_DOUBLE_EQ(result[3].imag(), expectedResult[3].imag());
   } else {
      // check within a given absolute tolerance
      EXPECT_NEAR(result[0].real(), expectedResult[0].real(), tolerance);
      EXPECT_NEAR(result[0].imag(), expectedResult[0].imag(), tolerance);

      EXPECT_NEAR(result[1].real(), expectedResult[1].real(), tolerance);
      EXPECT_NEAR(result[1].imag(), expectedResult[1].imag(), tolerance);

      EXPECT_NEAR(result[2].real(), expectedResult[2].real(), tolerance);
      EXPECT_NEAR(result[2].imag(), expectedResult[2].imag(), tolerance);

      EXPECT_NEAR(result[3].real(), expectedResult[3].real(), tolerance);
      EXPECT_NEAR(result[3].imag(), expectedResult[3].imag(), tolerance);
   }
}

void print(std::vector<double> const &coeff, const vector<complex<double>> &result, EModeType type)
{
   std::string algoName = (type == kNumerical) ? "Numerical" : "Analytical";
   std::cout << algoName << " solution for " << coeff[0] << "*x^4 + " << coeff[1] << "*x^3 + " << coeff[2] << "*x^2 + "
             << coeff[3] << "*x + " << coeff[4] << " = 0" << std::endl;
   for (unsigned int i = 0; i < result.size(); i++) {
      std::cout << " root " << i << " : " << result[i] << std::endl;
   }
}

void runTest(std::vector<double> const &coeff, std::vector<complex<double>> const &expectedResult, EModeType type,
             double tol = 0, bool debug = false)
{
   ROOT::Math::Polynomial pol(coeff[0], coeff[1], coeff[2], coeff[3], coeff[4]);

   auto result = (type != kNumerical) ? pol.FindRoots() : pol.FindNumRoots();

   checkResult(result, expectedResult, tol);

   if (debug) {
      print(coeff, result, type);
   }
}

} // namespace

// test x^4 - 16 = 0
TEST(QuarticPolynomial, FindRoots_DegPol1)
{
   std::vector<double> coeff{1, 0, 0, 0, -16};

   std::vector<complex<double>> expectedResult{-2., -2.i, 2.i, 2.};

   runTest(coeff, expectedResult, kAnalytical);
   runTest(coeff, expectedResult, kNumerical, 1.E-13);
}

// test (x+1)^4 = 0
TEST(QuarticPolynomial, FindRoots_DegPol2)
{
   std::vector<double> coeff{1, 4, 6, 4, 1};

   std::vector<complex<double>> expectedResult{-1., -1., -1., -1.};

   runTest(coeff, expectedResult, kAnalytical);
   // bad tolerance for this test
   // numerical method has a large error for this case
   runTest(coeff, expectedResult, kNumerical, 1.E-3);
}

// test x4 + 5x^2 + 4 = 0
// 4 imaginary roots (x-i)(x+i)(x-2i)(x+2i)=0
TEST(QuarticPolynomial, FindRoots_4ImagRoots)
{
   std::vector<double> coeff{1, 0, 5, 0, 4};

   std::vector<complex<double>> expectedResult{-2i, -1i, 1i, 2i};

   runTest(coeff, expectedResult, kAnalytical, 0);
   runTest(coeff, expectedResult, kNumerical, 1.E-12);
}

//
// four full complex roots, (x-1-i)(x-1+i)(x-1-2i)(x-1+2i)=0")
TEST(QuarticPolynomial, FindRoots_4CompRoots)
{
   std::vector<double> coeff{1.0, -4.0, 11.0, -14.0, 10.0};

   std::vector<complex<double>> expectedResult{1. - 2i, 1. - 1i, 1. + 1i, 1. + 2i};

   runTest(coeff, expectedResult, kAnalytical, 0);
   runTest(coeff, expectedResult, kNumerical, 1.E-12);
}

// four real roots (x+1)(x+2)(x+3)(x+4)=0
TEST(QuarticPolynomial, FindRoots_4RealRoots)
{

   std::vector<double> coeff{1.0, 10.0, 35.0, 50.0, 24.0};

   std::vector<complex<double>> expectedResult{-4., -3., -2., -1.};

   runTest(coeff, expectedResult, kAnalytical, 0);
   runTest(coeff, expectedResult, kNumerical, 1.E-12);
}

// test   x^4-8x^3+12x^2+16x+4=0
// 4 real roots where 2 are degeenrates

TEST(QuarticPolynomial, FindRoots_4RealDegRoots)
{

   std::vector<double> coeff{1.0, -8.0, 12.0, 16.0, 4.0};

   // 3rd and 4th values are 4 - r0
   std::vector<complex<double>> expectedResult{-0.44948974278317788, -0.44948974278317788, 4.44948974278317788,
                                               4.44948974278317788};

   runTest(coeff, expectedResult, kAnalytical, 0);
   runTest(coeff, expectedResult, kNumerical, 1.E-7);
}

// test 5x^4 + 4x^3 + 3x^2 + 2x + 1

TEST(QuarticPolynomial, FindRoots_54321)
{
   std::vector<double> coeff{5, 4, 3, 2, 1};

   std::vector<complex<double>> expectedResult{
      -0.5378322749029899 - 0.358284686345128i, -0.5378322749029899 + 0.358284686345128i,
      +0.13783227490298988 - 0.6781543891053364i, +0.13783227490298988 + 0.6781543891053364i};

   runTest(coeff, expectedResult, kAnalytical);
   runTest(coeff, expectedResult, kNumerical, 1.E-13);
}

// special case reported in issue #6900 by S. Binet
TEST(QuarticPolynomial, DISABLED_FindRoots_4RealDegRootsR0)
{
   std::vector<double> coeff{2.2206846808021337, 7.643281053997895, 8.831759446092846, 3.880673545129404,
                             0.5724551380144077};

   std::vector<complex<double>> expectedResult{-1.3429253, -1.3427327, -0.3781962, -0.3780036};

   runTest(coeff, expectedResult, kAnalytical, 1.E-6);
   runTest(coeff, expectedResult, kNumerical, 1.E-4);
}
