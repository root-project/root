// Author: Jonas Rembser, CERN  01/2023

#include <RooAbsPdf.h>
#include <RooRealVar.h>
#include <RooNumIntConfig.h>
#include <RooWorkspace.h>

#include <gtest/gtest.h>

#include <iostream>

TEST(RooLandau, Integral)
{
   RooWorkspace ws;
   ws.factory("Landau::landau(x[-100, 100], mean[-100, 100], sigma[0.01, 100])");

   RooRealVar &x = *ws.var("x");
   RooRealVar &mean = *ws.var("mean");
   RooRealVar &sigma = *ws.var("sigma");

   RooAbsPdf *landau = ws.pdf("landau");

   std::unique_ptr<RooAbsPdf> landauNumInt{static_cast<RooAbsPdf *>(landau->clone())};
   landauNumInt->forceNumInt(true);

   RooNumIntConfig intConfig(*RooAbsReal::defaultIntegratorConfig());
   intConfig.setEpsAbs(1.E-15);
   intConfig.setEpsRel(1.E-12);

   intConfig.getConfigSection("RooIntegrator1D").setRealValue("maxSteps", 100);
   landauNumInt->setIntegratorConfig(intConfig);

   std::unique_ptr<RooAbsReal> integral{landau->createIntegral(x)};
   std::unique_ptr<RooAbsReal> integralRanged{landau->createIntegral(x, "integrationRange")};
   std::unique_ptr<RooAbsReal> numInt{landauNumInt->createIntegral(x)};
   std::unique_ptr<RooAbsReal> numIntRanged{landauNumInt->createIntegral(x, "integrationRange")};

   for (double meanVal : {-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0}) {
      for (double sigmaVal : {1., 2., 5., 10.}) {
         mean.setVal(meanVal);
         sigma.setVal(sigmaVal);

         for (double r1 : {-2., -1.5, -0.5, -0.2, 0.2, 0.5, 1.5, 2.}) {
            for (double r2 : {-2., -1.5, -0.5, -0.2, 0.2, 0.5, 1.5, 2.}) {
               if (r2 <= r1) {
                  continue;
               }
               x.setRange("integrationRange", r1, r2);

               constexpr double accAnaVsNum = 1.;
               EXPECT_NEAR(integralRanged->getVal(), numIntRanged->getVal(),
                           accAnaVsNum / 100. * numIntRanged->getVal())
                  << "Analytical vs numerical integral (subrange)"
                  << " within " << accAnaVsNum << "%. With mean=" << meanVal << " sigma=" << sigmaVal
                  << ", integration from x=" << r1 << " to x=" << r2;
            }
         }

         constexpr double accAnaVsNum = 3.;
         EXPECT_NEAR(integral->getVal(), numInt->getVal(), accAnaVsNum / 100. * numIntRanged->getVal())
            << "Analytical vs numerical integral (full range) "
            << " within " << accAnaVsNum << "%. With mean=" << meanVal << " sigma=" << sigmaVal;
      }
   }
}
