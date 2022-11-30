#include <RooRealVar.h>
#include <RooWorkspace.h>
#include <RooRealProxy.h>
#include <RooGaussian.h>
#include <RooDataSet.h>
#include <RooAbsPdf.h>
#include <TROOT.h>
#include <TSystem.h>
#include <RooFitResult.h>
#include <RooListProxy.h>
#include <TMath.h>

#include "gtest/gtest.h"

class Interface : public ::testing::Test {
};

TEST(Interface, createNLLRooAbsGradFuncWrapper)
{

   using namespace RooFit;

   RooRealVar x("x", "x", 0, -10, 10);
   RooRealVar mu("mu", "mu", 0, -10, 10);
   RooRealVar sigma("sigma", "sigma", 2.0, 0.01, 10);

   RooGaussian gauss{"gauss", "gauss", x, mu, sigma};

   std::unique_ptr<RooDataSet> data{gauss.generate(x, 10)};

   std::unique_ptr<RooAbsReal> nll{gauss.createNLL(*data, CodeSquashing(true))};
   std::unique_ptr<RooAbsReal> nllRef{gauss.createNLL(*data)};

   EXPECT_FLOAT_EQ(nll->getVal(), nllRef->getVal());
}
