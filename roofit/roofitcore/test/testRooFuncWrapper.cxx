#include <RooRealVar.h>
#include <RooWorkspace.h>
#include <RooRealProxy.h>
#include <RooGaussian.h>
#include <RooFuncWrapper.h>
#include <RooDataSet.h>
#include <RooAbsPdf.h>
#include <TROOT.h>
#include <TSystem.h>
#include <RooFitResult.h>
#include <RooListProxy.h>
#include <TMath.h>

#include "gtest/gtest.h"

TEST(RooFuncWrapper, GaussianNormalized)
{
   using namespace RooFit;

   auto inf = std::numeric_limits<double>::infinity();
   RooRealVar x("x", "x", 0, -inf, inf);
   RooRealVar mu("mu", "mu", 0, -10, 10);
   RooRealVar sigma("sigma", "sigma", 2.0, 0.01, 10);
   RooGaussian gauss{"gauss", "gauss", x, mu, sigma};

   RooArgSet normSet{x};
   RooArgSet paramsGauss;
   RooArgSet paramsMyGauss;

   std::string func = "const double arg = x[0] - x[1];"
                      "const double sig = x[2];"
                      "double out = std::exp(-0.5 * arg * arg / (sig * sig));"
                      "return 1. / (std::sqrt(TMath::TwoPi()) * sig) * out;";
   RooFuncWrapper<> gaussFunc("myGauss", "myGauss", func, {x, mu, sigma});

   // Check if functions results are the same even after changing parameters.
   EXPECT_NEAR(gauss.getVal(normSet), gaussFunc.getVal(), 1e-8);

   mu.setVal(1);
   EXPECT_NEAR(gauss.getVal(normSet), gaussFunc.getVal(), 1e-8);

   // Check if the parameter layout and size is the same.
   gauss.getParameters(&normSet, paramsGauss);
   gaussFunc.getParameters(&normSet, paramsMyGauss);

   EXPECT_TRUE(paramsMyGauss.hasSameLayout(paramsGauss));
   EXPECT_EQ(paramsMyGauss.size(), paramsGauss.size());
}
