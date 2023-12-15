// Tests for the RooBifurGauss
// Authors: Jonas Rembser, CERN 2023

#include <RooBifurGauss.h>
#include <RooGaussian.h>
#include <RooRealVar.h>

#include <gtest/gtest.h>

/// Cross-check the analytical integration code with the normal Gaussian.
TEST(RooBifurGauss, AnalyticalIntegralCrossCheck)
{
   RooRealVar x("x", "x", 0.0, -10, 10);
   RooRealVar mean("mean", "mean of gaussian", 0.0, -10, 10);
   RooRealVar sigma("sigma", "width of gaussian", 1, 0.1, 10);
   RooRealVar sigmaR("sigmaR", "width of gaussian", 1, 0.1, 10);

   RooGaussian gauss("gauss", "gaussian PDF", x, mean, sigma);
   RooBifurGauss bGauss("bGauss", "gaussian PDF", x, mean, sigma, sigmaR);

   // We need to set the range also for "mean", because this test also checks
   // the analytical integral over the mean variable.
   for (RooRealVar *var : {&x, &mean}) {
      var->setRange("R1", -5, 5);
      var->setRange("R2", -5, -1);
      var->setRange("R3", 1, 5);
   }

   std::vector<std::string> rangeNames{"R1", "R2", "R3"};

   for (std::size_t i = 0; i < rangeNames.size(); ++i) {

      const char *rangeName = rangeNames[i].c_str();

      std::unique_ptr<RooAbsReal> integ{gauss.createIntegral(x, rangeName)};
      std::unique_ptr<RooAbsReal> bInteg{bGauss.createIntegral(x, rangeName)};

      std::unique_ptr<RooAbsReal> integMu{gauss.createIntegral(mean, rangeName)};
      std::unique_ptr<RooAbsReal> bIntegMu{bGauss.createIntegral(mean, rangeName)};

      const double ref = integ->getVal();

      EXPECT_FLOAT_EQ(bInteg->getVal(), ref) << "the BifurGauss should be equivalent!";
      EXPECT_FLOAT_EQ(integMu->getVal(), ref) << "integral over mu should be the same as over x for RooGaussian!";
      EXPECT_FLOAT_EQ(bIntegMu->getVal(), ref) << "integral over mu should be the same as over x for RooBifurGauss!";
   }
}
