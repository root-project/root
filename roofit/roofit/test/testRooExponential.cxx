// Tests for the RooJohnson
// Author: Stephan Hageboeck, CERN  05/2019

#include "RooExponential.h"

#include "RooRealVar.h"
#include "RooNumIntConfig.h"
#include "RooDataSet.h"
#include "RooFitResult.h"

#include "TFile.h"

#include <numeric>

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
