// Tests for the RooJohnson
// Author: Stephan Hageboeck, CERN  05/2019

#include "RooExponential.h"

#include "RooRealVar.h"
#include "RooNumIntConfig.h"
#include "RooDataSet.h"
#include "RooFitResult.h"
#include "RooFit/Detail/MathFuncs.h"

#include "TFile.h"

#include <cmath>
#include <limits>
#include <numeric>
#include <vector>

#include "gtest/gtest.h"

double simpleIntegration(RooRealVar& intVar, const RooAbsPdf& pdf) {
  const std::size_t nPoints = 200000;
  const double rangeMin = intVar.getMin();
  const double rangeMax = intVar.getMax();
  const double width = rangeMax - rangeMin;

  std::vector<double> values(nPoints, 0.);
  for (unsigned int i=0; i < nPoints; ++i) {
    intVar.setVal(rangeMin + width/nPoints * i);
    values[i] = pdf.getVal();
  }

  const double manualInt = std::accumulate(values.begin(), values.end(), 0.) / nPoints * width;

  return manualInt;
}



TEST(RooExponential, Integral)
{
  RooRealVar x("x", "x", 0.001, 30);
  x.setRange("integrationRange", 0.001, 30);

  RooRealVar c("c", "c", -0.2, -50., -0.001);
  RooExponential ex("exp", "exp", x, c);

  RooExponential expNumInt(ex);

  RooNumIntConfig intConfig(*RooAbsReal::defaultIntegratorConfig());
  intConfig.setEpsAbs(1.E-15);
  intConfig.setEpsRel(1.E-12);

  intConfig.getConfigSection("RooIntegrator1D").setRealValue("maxSteps", 100);
  expNumInt.setIntegratorConfig(intConfig);
  expNumInt.forceNumInt(true);

  auto integral = ex.createIntegral(x);
  auto integralRanged = ex.createIntegral(x, "integrationRange");
  auto numInt   = expNumInt.createIntegral(x);
  auto numIntRanged = expNumInt.createIntegral(x, "integrationRange");



  for (double theC : {-0.001, -0.01, -0.1, -1., -20., -50.}) {
    c = theC;

    auto result = simpleIntegration(x, ex);

    constexpr double accSumVsNum = 3.;
    EXPECT_NEAR(numIntRanged->getVal(), result, accSumVsNum/100.*result)
    << "Simple sum vs numerical integration within " << accSumVsNum << "%.";

    double accAnaVsNum = 1.;
    EXPECT_NEAR(integralRanged->getVal(), numIntRanged->getVal(),
        accAnaVsNum/100.*numIntRanged->getVal())
    << "Analytical vs numerical integral"
    << " within " << accAnaVsNum << "%. With "
    << theC;

    if (integral->getVal() > 1.E-9) { //Numerical integral cannot do this
      accAnaVsNum = 3.;
      EXPECT_NEAR(integral->getVal(), numInt->getVal(),
          accAnaVsNum/100.*numIntRanged->getVal())
      << "Analytical vs numerical integral (full range) "
      << " within " << accAnaVsNum << "%. With "
      << theC;
    }
  }
}


// Integral and slope derivative against 80-digit references, in particular
// near zero slope where the naive closed form cancels
// (https://github.com/root-project/root/issues/23724). Positive and negative
// slopes cover both branches of the implementation.
TEST(RooExponential, IntegralNearZeroSlope)
{
   using RooFit::Detail::MathFuncs::exponentialIntegral;
   using RooFit::Detail::MathFuncs::exponentialIntegralDConstant;

   struct Reference {
      double constant;
      double integral;
      double derivative;
   };

   struct Range {
      double xMin;
      double xMax;
      std::vector<Reference> refs;
   };

   // clang-format off
   const std::vector<Range> ranges{
      {100., 180., {
         {-1.0, 3.7200759760208360e-44, 3.7572767357810443e-42},
         {-0.1, 0.00045384699782740139, 0.049910985777218383},
         {-0.01, 20.258055294985578, 2729.2199532244234},
         {-0.001, 69.567206624687552, 9702.3103742545453},
         {-1e-06, 79.988800805293761, 11198.389452047368},
         {-1.497412393733355e-11, 79.999999832289812, 11199.999975881678},
         {-1e-15, 79.999999999988800, 11199.999999998389},
         {0.0, 80, 11200},
         {1e-15, 80.000000000011200, 11200.000000001611},
         {1e-12, 80.000000011200000, 11200.000001610667},
         {1e-06, 80.011200805372908, 11201.610785392632},
         {0.001, 92.046445046162540, 12935.588508198527},
         {0.01, 333.13656359539008, 48397.179715303569},
         {0.1, 656379426.71535704, 111602123714.24654},
         {1.0, 1.4893842007818384e+78, 2.6659977193994907e+80}}},
      {-50., 30., {
         {-1.0, 5.1847055285870725e+21, -2.5405057090076655e+23},
         {-0.1, 1483.6337203420874, -59385.178468377787},
         {-0.01, 90.790305001841028, -1387.0305153616915},
         {-0.001, 80.825562827515863, -851.35799774058453},
         {-1e-06, 80.000800025333560, -800.05066734667789},
         {-1.497412393733355e-11, 80.000000011979299, -800.00000075868895},
         {-1e-15, 80.000000000000800, -800.00000000005067},
         {0.0, 80, -800},
         {1e-15, 79.999999999999200, -799.99999999994933},
         {1e-12, 79.999999999200000, -799.99999994933333},
         {1e-06, 79.999200025333107, -799.94933401332211},
         {0.001, 79.225109452802847, -750.00220916164040},
         {0.01, 74.332814786336968, -351.05175734252037},
         {0.1, 200.78798976188582, 4021.1501528369848},
         {1.0, 10686474581524.462, 309907762864209.40}}}};
   // clang-format on

   constexpr double relTol = 1e-13;

   for (auto const &range : ranges) {
      for (auto const &ref : range.refs) {
         const double integral = exponentialIntegral(range.xMin, range.xMax, ref.constant);
         const double derivative = exponentialIntegralDConstant(range.xMin, range.xMax, ref.constant);
         EXPECT_NEAR(integral, ref.integral, relTol * std::abs(ref.integral))
            << "integral from " << range.xMin << " to " << range.xMax << " with constant " << ref.constant;
         EXPECT_NEAR(derivative, ref.derivative, relTol * std::abs(ref.derivative))
            << "derivative of integral from " << range.xMin << " to " << range.xMax << " with constant "
            << ref.constant;
         // Swapping the limits flips the sign.
         EXPECT_DOUBLE_EQ(exponentialIntegral(range.xMax, range.xMin, ref.constant), -integral);
         EXPECT_DOUBLE_EQ(exponentialIntegralDConstant(range.xMax, range.xMin, ref.constant), -derivative);
      }
   }

   // Infinite ranges, as for unbounded variables.
   constexpr double inf = std::numeric_limits<double>::infinity();
   EXPECT_NEAR(exponentialIntegral(-inf, 180., 0.01), std::exp(1.8) / 0.01, relTol * std::exp(1.8) / 0.01);
   EXPECT_NEAR(exponentialIntegral(100., inf, -0.01), std::exp(-1.) / 0.01, relTol * std::exp(-1.) / 0.01);
   EXPECT_EQ(exponentialIntegral(-inf, 180., -0.01), inf);
   EXPECT_EQ(exponentialIntegral(100., inf, 0.01), inf);
   EXPECT_EQ(exponentialIntegral(-inf, inf, 0.), inf);

   // No spurious over- or underflow if the integral itself is representable.
   EXPECT_NEAR(exponentialIntegral(-800., 0., 1.), 1., relTol);
   EXPECT_NEAR(exponentialIntegral(0., 800., -1.), 1., relTol);
}

TEST(RooExponential, IO) {
  TFile file("./exponentialPdf.root");
  ASSERT_TRUE(file.IsOpen());

  RooExponential* expo;
  file.GetObject("expo", expo);
  ASSERT_NE(expo, nullptr);

  // This depends on proper (de-)serialisation of proxies:
  std::unique_ptr<RooArgSet> vars(expo->getVariables());
  RooRealVar& x  = dynamic_cast<RooRealVar&>((*vars)["x"]);
  RooRealVar& ex = dynamic_cast<RooRealVar&>((*vars)["ex"]);

  x = 4.;
  ex = -5.;

  EXPECT_NEAR(expo->getVal(), std::exp(4.*-5.), 1.E-14);
}
