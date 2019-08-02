// Tests for the RooJohnson
// Author: Stephan Hageboeck, CERN  05/2019

#include "RooExponential.h"

#include "RooRealVar.h"
#include "RooNumIntConfig.h"
#include "RooDataSet.h"
#include "RooFitResult.h"

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

//TEST(RooJohnson, Generator)
//{
//  MAKE_JOHNSON_AND_VARS
//
//  ASSERT_FALSE(mu.isConstant());
//
//  mu = 120.;
//  sigma = 30.;
//  gamma = 3.;
//  delta = 3.;
//
//  auto frame = mass.frame(RooFit::Title("Johnson"));
//  auto data = johnson.generate(mass, RooFit::NumEvents(10000));
//  data->plotOn(frame);
//  johnson.plotOn(frame, RooFit::LineColor(kRed), RooFit::LineColor(kDotted));
//  johnson.fitTo(*data, RooFit::PrintLevel(-1));
//  johnson.plotOn(frame, RooFit::LineColor(kBlue), RooFit::LineColor(kDashed));
//  johnson.paramOn(frame);
//
//  EXPECT_LT(frame->chiSquare(), 1.);
//
////  TCanvas canv;
////  frame->Draw();
////  canv.SaveAs("/tmp/Johnson_gen1.png");
//
//
//
//
//  mu = -100.;
//  sigma = 50.;
//  gamma = 2.;
//  delta = 1.;
//
//  frame = mass.frame(RooFit::Title("Johnson"));
//  auto data2 = johnson.generate(mass, 10000.);
//  data2->plotOn(frame);
//  johnson.plotOn(frame, RooFit::LineColor(kBlue));
//  EXPECT_LT(frame->chiSquare(), 1.);
//
//
//  auto res = johnson.fitTo(*data2, RooFit::Save(), RooFit::PrintLevel(-1));
//  johnson.plotOn(frame, RooFit::LineColor(kRed), RooFit::LineStyle(kDashed));
//  johnson.paramOn(frame);
//  EXPECT_LT(frame->chiSquare(res->floatParsInit().size()), 1.);
//
////  TCanvas canv2;
////  frame->Draw();
////  canv2.SaveAs("/tmp/Johnson_gen2.png");
//}

