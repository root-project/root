// Author: Jonas Rembser, CERN  01/2025

#include "RooMultiVarGaussian.h"
#include "RooStats/AsymptoticCalculator.h"

#include "gtest/gtest.h"

// Check if asymptotic datasets for counting experiments can also be generated
// from the RooMultiVarGaussian.
TEST(AsymptoticCalculator, CountingAsimovDataSetFromMultiVarGaussian)
{

   RooWorkspace ws;
   ws.factory("x1[0.0, -3.0, 3.0]");
   ws.factory("x2[0.0, -3.0, 3.0]");
   ws.factory("mu1[1.0, -3.0, 3.0]");
   ws.factory("mu2[2.0, -3.0, 3.0]");

   RooArgSet observables{*ws.var("x1"), *ws.var("x2")};
   RooArgSet means{*ws.var("mu1"), *ws.var("mu2")};

   TMatrixDSym cov{2};
   cov(0, 0) = 1.0;
   cov(0, 1) = 0.2;
   cov(1, 0) = 0.2;
   cov(1, 1) = 1.0;
   RooMultiVarGaussian multiVarGauss{"multi_var_gauss", "", observables, means, cov};

   std::unique_ptr<RooAbsData> data{RooStats::AsymptoticCalculator::GenerateAsimovData(multiVarGauss, observables)};

   RooArgSet const &dataObs = *data->get(0);

   // Check if the observable values were successfully set to the mean values.
   for (std::size_t i = 0; i < observables.size(); ++i) {
      auto const &dataX = *static_cast<RooRealVar const *>(dataObs[i]);
      auto const &mu = *static_cast<RooRealVar const *>(means[i]);
      EXPECT_EQ(dataX.getVal(), mu.getVal());
   }
}
