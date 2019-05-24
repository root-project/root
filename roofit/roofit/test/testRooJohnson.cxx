// Tests for the RooJohnson
// Author: Stephan Hageboeck, CERN  05/2019

#include "RooJohnson.h"

#include "RooRealVar.h"
#include "RooGaussian.h"
#include "RooLinearVar.h"
#include "RooRealConstant.h"
#include "RooFirstMoment.h"
#include "RooGenericPdf.h"
#include "RooNumIntConfig.h"
#include "RooDataSet.h"
#include "RooFitResult.h"

#include "TCanvas.h"
#include "RooPlot.h"

#include <numeric>

#include "gtest/gtest.h"

#define MAKE_JOHNSON_AND_VARS RooRealVar mass("mass", "mass", 0., -200., 200.);\
RooRealVar mu("mu", "Location parameter of normal distribution", 100., -200., 200.);\
RooRealVar sigma("sigma", "Two sigma of normal distribution", 2., 0., 100.);\
RooRealVar gamma("gamma", "gamma", -10., -100., 100.);\
RooRealVar delta("delta", "delta", 3., 0., 100.);\
RooJohnson johnson("johnson", "johnson", mass, mu, sigma, gamma, delta, -1.E300);

const char* originalFormula = "delta/(sigma*TMath::Sqrt(TMath::Pi()))"
    "*TMath::Exp(-0.5*(gamma+delta*TMath::ASinH((mass-mu)/sigma))"
                    "*(gamma+delta*TMath::ASinH((mass-mu)/sigma)))"
    "/TMath::Sqrt(1+(mass-mu)*(mass-mu)/(sigma*sigma))";
const char* fixedFormula = "delta/(sigma*TMath::Sqrt(TMath::TwoPi()))"
    "*TMath::Exp(-0.5*(gamma+delta*TMath::ASinH((mass-mu)/sigma))"
                    "*(gamma+delta*TMath::ASinH((mass-mu)/sigma)))"
    "/TMath::Sqrt(1+(mass-mu)*(mass-mu)/(sigma*sigma))";



TEST(RooJohnson, ReferenceImplementation)
{
  MAKE_JOHNSON_AND_VARS
  // Note: Ownership bug. Deleting this might crash on Mac.
  // Therefore, it will leak because we are testing not the
  // GenericPdf.
  auto johnsonRef = new RooGenericPdf("johnsonRef",
      fixedFormula,
      RooArgSet( mass, mu, sigma, gamma, delta));

  for (double theMass : {-100., -50., -10., -1., 0., 1., 10., 50., 100.}) {
    for (double theMu : {-100., -10., 0., 10., 20., 30., 100., 150.}) {
      for (double theSig : {0.1, 1., 2., 10.}) {
        for (double theGam : {-10., -1., 0., 1., 2., 10.}) {
          for (double theDelta : {0.1, 1., 2., 10.}) {
            mass = theMass;
            mu = theMu;
            sigma = theSig;
            gamma = theGam;
            delta = theDelta;

            EXPECT_FLOAT_EQ(johnson.getVal(), johnsonRef->getVal())
            << theMass << " " << theMu << " " << theSig << " " << theGam << " " << theDelta;
          }
        }
      }
    }
  }
}




std::vector<double> simpleIntegration(RooRealVar& intVar, const RooJohnson& johnson) {
  const std::size_t nPoints = 20000;
  const double rangeMin = intVar.getMin();
  const double rangeMax = intVar.getMax();
  const double width = rangeMax - rangeMin;

  std::vector<double> values(nPoints, 0.);
  for (unsigned int i=0; i < nPoints; ++i) {
    intVar.setVal(rangeMin + width/nPoints * i);
    values[i] = johnson.getVal();
  }
  const double maxVal = *std::max_element(values.begin(), values.end());
  const auto lower = std::find_if(values.begin(), values.end(), [=](double elm){
    return elm > maxVal/1000.;});
  const auto upper = std::find_if(values.rbegin(), values.rend(), [=](double elm){
    return elm > maxVal/1000.;}).base();
  const double manualInt = std::accumulate(lower, upper, 0.) / nPoints * width;
  const double rangeLo = rangeMin + width/nPoints * (lower-values.begin());
  const double rangeHi = rangeMin + width/nPoints * (upper-values.begin());

  return {manualInt, rangeLo, rangeHi};
}



TEST(RooJohnson, Integral)
{
  MAKE_JOHNSON_AND_VARS

  mass.setRange(-199., 199);

  RooJohnson johnsonNumInt(johnson);

  RooNumIntConfig intConfig(*RooAbsReal::defaultIntegratorConfig());
  intConfig.setEpsAbs(1.E-15);
  intConfig.setEpsRel(1.E-12);

  intConfig.getConfigSection("RooIntegrator1D").setRealValue("maxSteps", 100);
  johnsonNumInt.setIntegratorConfig(intConfig);
  johnsonNumInt.forceNumInt(true);

  auto integral = johnson.createIntegral(mass);
  auto integralRanged = johnson.createIntegral(mass, "integrationRange");
  auto numInt   = johnsonNumInt.createIntegral(mass);
  auto numIntRanged = johnsonNumInt.createIntegral(mass, "integrationRange");
  mass.setRange("integrationRange", -200, 200);


  for (double theSig : {5., 10., 20., 50.}) { //Numerical integration poor for narrow peaks
    for (double theMu : {-170., -100., 0., 30., 100., 150., 180.}) {
      for (double theGam : {-10., -1., 0., 1., 2., 10.}) {
        for (double theDelta : {1., 2., 10.}) {
          mu = theMu;
          sigma = theSig;
          gamma = theGam;
          delta = theDelta;

          const double expDelta = exp(pow(theDelta, -2.));
          const double variance = theSig*theSig/2.
              * (expDelta - 1.)
              * (expDelta * cosh(2.*theGam / theDelta) + 1.);

          const double median = theMu + theSig * sinh(-1. * theGam / theDelta);

          if (-100. < median && median < 100. && variance < 50.) {//central and peaked
            EXPECT_GT(integral->getVal(), 0.95) << "Central and peaked for "
            << theMu << " " << theSig << " " << theGam << " " << theDelta;

            if (variance > 10.) {
              EXPECT_GT(numInt->getVal(), 0.95) << "Central and peaked for "
                  << theMu << " " << theSig << " " << theGam << " " << theDelta;
            }
          }

          auto result = simpleIntegration(mass, johnson);
          mass.setRange("integrationRange", result[1], result[2]);

          constexpr double accSumVsNum = 3.;
          EXPECT_NEAR(numIntRanged->getVal(), result[0], accSumVsNum/100.*result[0])
          << "Simple sum vs numerical integration within " << accSumVsNum << "%.";

          constexpr double accAnaVsNumCore = 1.;
          EXPECT_NEAR(integralRanged->getVal(), numIntRanged->getVal(),
              accAnaVsNumCore/100.*numIntRanged->getVal())
              << "Analytical vs numerical integral (core region)"
              << " within " << accAnaVsNumCore << "%. With "
              << theMu << " " << theSig << " " << theGam << " " << theDelta;

          if (integral->getVal() > 1.E-9 && variance > 10.) { //Numerical integral cannot do this
            constexpr double accAnaVsNum = 3.;
            EXPECT_NEAR(integral->getVal(), numInt->getVal(),
                accAnaVsNum/100.*numIntRanged->getVal())
            << "Analytical vs numerical integral (full range) variance=" << variance
            << " within " << accAnaVsNum << "%. With "
            << theMu << " " << theSig << " " << theGam << " " << theDelta;
          }

//          auto frame = mass.frame();
//          johnson.plotOn(frame, RooFit::LineColor(kBlue));
//          TCanvas canv;
//          frame->Draw();
//          canv.SaveAs(Form("/tmp/Johnson_%f_%f_%f_%f.png", theMu, theSig, theGam, theDelta));
        }
      }
    }
  }
}


TEST(RooJohnson, MeanValue)
{
  MAKE_JOHNSON_AND_VARS

  RooFirstMoment firstMoment("mean", "mean (numerical)", johnson, mass);

  auto literatureMean = [&]() {
    const double theMu = mu.getVal();
    const double sig = sigma.getVal();
    const double gam = gamma.getVal();
    const double del = delta.getVal();
    return theMu - sig * exp(pow(del, -2.) / 2.) * sinh(gam/del);
  };

  EXPECT_NEAR(firstMoment.getVal(), literatureMean(), 0.02*literatureMean());

  mu = 50.;
  sigma = 20.;
  gamma = -2.;
  delta = 20.;
  EXPECT_NEAR(firstMoment.getVal(), literatureMean(), 0.02*literatureMean());
}



TEST(RooJohnson, Generator)
{
  MAKE_JOHNSON_AND_VARS

  ASSERT_FALSE(mu.isConstant());

  mu = 120.;
  sigma = 30.;
  gamma = 3.;
  delta = 3.;

  auto frame = mass.frame(RooFit::Title("Johnson"));
  auto data = johnson.generate(mass, RooFit::NumEvents(10000));
  data->plotOn(frame);
  johnson.plotOn(frame, RooFit::LineColor(kRed), RooFit::LineColor(kDotted));
  johnson.fitTo(*data, RooFit::PrintLevel(-1));
  johnson.plotOn(frame, RooFit::LineColor(kBlue), RooFit::LineColor(kDashed));
  johnson.paramOn(frame);

  EXPECT_LT(frame->chiSquare(), 1.);

//  TCanvas canv;
//  frame->Draw();
//  canv.SaveAs("/tmp/Johnson_gen1.png");




  mu = -100.;
  sigma = 50.;
  gamma = 2.;
  delta = 1.;

  frame = mass.frame(RooFit::Title("Johnson"));
  auto data2 = johnson.generate(mass, 10000.);
  data2->plotOn(frame);
  johnson.plotOn(frame, RooFit::LineColor(kBlue));
  EXPECT_LT(frame->chiSquare(), 1.);


  auto res = johnson.fitTo(*data2, RooFit::Save(), RooFit::PrintLevel(-1));
  johnson.plotOn(frame, RooFit::LineColor(kRed), RooFit::LineStyle(kDashed));
  johnson.paramOn(frame);
  EXPECT_LT(frame->chiSquare(res->floatParsInit().size()), 1.);

//  TCanvas canv2;
//  frame->Draw();
//  canv2.SaveAs("/tmp/Johnson_gen2.png");
}

