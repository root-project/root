// Author: Jonas Rembser, CERN  01/2025

#include <RooNumIntConfig.h>
#include <RooAbsPdf.h>
#include <RooRealVar.h>
#include <RooWorkspace.h>

#include "gtest/gtest.h"

TEST(RooHypatia2, AnalyticIntegration)
{
   RooWorkspace ws;

   // For analytical integration over x, zeta and beta need to be constant.
   ws.factory("Hypatia2::hypatia("
              "x[0.0, -10., 10.], "
              "lambda[-1.0, -100., -0.01], "
              "0.0," // zeta: constant
              "0.0," // beta: constant
              "sigma[1.0, 0.01, 10.], "
              "mu[0.0], "
              "a[50., 0.01, 1000.], "
              "n[1.5, 0.01, 10.], "
              "a2[1.0, 0.01, 1000.], "
              "n2[0.1, 0.01, 10.])");

   RooRealVar &x = *ws.var("x");

   RooAbsPdf *hypatia = ws.pdf("hypatia");

   // Numeric integrator config with sufficient precision for the reference
   // values
   RooNumIntConfig intConfig(*RooAbsReal::defaultIntegratorConfig());
   intConfig.setEpsAbs(1.E-15);
   intConfig.setEpsRel(1.E-12);

   intConfig.getConfigSection("RooIntegrator1D").setRealValue("maxSteps", 100);

   std::unique_ptr<RooAbsPdf> hypatiaNumInt{static_cast<RooAbsPdf *>(hypatia->clone())};
   hypatiaNumInt->setIntegratorConfig(intConfig);
   hypatiaNumInt->forceNumInt(true);

   // Integral objects
   std::unique_ptr<RooAbsReal> integral{hypatia->createIntegral(x)};
   std::unique_ptr<RooAbsReal> numInt{hypatiaNumInt->createIntegral(x)};

   const double integralVal = integral->getVal();
   const double integralRefVal = numInt->getVal();

   // TODO: extend test to cover all code branches of
   // RooHypatia2::analyticalIntegral().
   EXPECT_NEAR(integralVal, integralRefVal, 0.01 * integralRefVal);
}
