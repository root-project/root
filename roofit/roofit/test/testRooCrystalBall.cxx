// Author: Jonas Rembser, CERN  02/2021

#include "RooCrystalBall.h"
#include "RooCBShape.h"

#include "RooRealVar.h"
#include "RooGaussian.h"
#include "RooLinearVar.h"
#include "RooRealConstant.h"
#include "RooFirstMoment.h"
#include "RooGenericPdf.h"
#include "RooNumIntConfig.h"
#include "RooDataSet.h"
#include "RooFormulaVar.h"
#include "RooFitResult.h"

#include "TCanvas.h"
#include "RooPlot.h"

#include <numeric>
#include <string>
#include <iostream>

// You can also validate by comparing with the RooDSCBShape and RooSDSCBShape
// classes that are floating around in the RooFit user community.
// Some commented-out lines are kept on purpose in this test to make this as
// easy as possible.
//
//#include "RooDSCBShape.h"
//#include "RooSDSCBShape.h"

#include "gtest/gtest.h"

#define MAKE_CRYSTAL_BALL_AND_VARS                         \
   RooRealVar x("x", "x", 0., -200., 200.);                \
   RooRealVar x0("x0", "x0", 100., -200., 200.);           \
   RooRealVar sigmaL("sigmaL", "sigmaL", 2., 1.E-6, 100.); \
   RooRealVar sigmaR("sigmaR", "sigmaR", 2., 1.E-6, 100.); \
   RooRealVar alphaL("aL", "aL", 1., 1.E-6, 100.);         \
   RooRealVar alphaR("aR", "aR", 1., 1.E-6, 100.);         \
   RooRealVar nL("nL", "nL", 1., 1.E-6, 100.);             \
   RooRealVar nR("nR", "nR", 1., 1.E-6, 100.);             \
   RooCrystalBall crystalBall("crystalBall", "crystalBall", x, x0, sigmaL, sigmaR, alphaL, nL, alphaR, nR);

std::string makeCrystalBallFormulaOnlyLeftTail()
{
   std::string A = "(n/a)**n * exp(-(a**2)/2)";
   std::string B = "n/a - a";
   std::string fL = A + " * (" + B + " - (x - x0) / sigma)**(-n)";
   return std::string("(x - x0) / sigma < -a ? (") + fL + ") : exp(-0.5 * ((x - x0) / sigma)**2)";
}

std::string makeCrystalBallFormulaOnlyRightTail()
{
   std::string A = "(n/a)**n * exp(-(a**2)/2)";
   std::string B = "n/a - a";
   std::string fR = A + " * (" + B + " + (x - x0) / sigma)**(-n)";
   return std::string("(x - x0) / sigma <= a ? exp(-0.5 * ((x - x0) / sigma)**2) : (") + fR + ")";
}

std::string makeCrystalBallFormulaFullySymmetric()
{
   std::string A = "(n/a)**n * exp(-(a**2)/2)";
   std::string B = "n/a - a";
   std::string fL = A + " * (" + B + " - (x - x0) / sigma)**(-n)";
   std::string fR = A + " * (" + B + " + (x - x0) / sigma)**(-n)";
   std::string fC = "exp(-0.5 * ((x - x0) / sigma)**2)";
   return std::string("(x - x0) / sigma < -a ? (") + fL + ") : ((x - x0) / sigma <= a ? (" + fC + ") : (" + fR + ") )";
}

std::string makeCrystalBallFormulaDoubleSided()
{
   std::string AL = "(nL/aL)**nL * exp(-(aL**2)/2)";
   std::string AR = "(nR/aR)**nR * exp(-(aR**2)/2)";
   std::string BL = "nL/aL - aL";
   std::string BR = "nR/aR - aR";
   std::string fL = AL + " * (" + BL + " - (x - x0) / sigma)**(-nL)";
   std::string fR = AR + " * (" + BR + " + (x - x0) / sigma)**(-nR)";
   std::string fC = "exp(-0.5 * ((x - x0) / sigma)**2)";
   return std::string("(x - x0) / sigma < -aL ? (") + fL + ") : ((x - x0) / sigma <= aR ? (" + fC + ") : (" + fR +
          ") )";
}

std::string makeCrystalBallFormula()
{
   std::string AL = "(nL/aL)**nL * exp(-(aL**2)/2)";
   std::string AR = "(nR/aR)**nR * exp(-(aR**2)/2)";
   std::string BL = "nL/aL - aL";
   std::string BR = "nR/aR - aR";
   std::string fL = AL + " * (" + BL + " - (x - x0) / sigmaL)**(-nL)";
   std::string fR = AR + " * (" + BR + " + (x - x0) / sigmaR)**(-nR)";
   std::string fC = "exp(-0.5 * ((x - x0) / (x < x0 ? sigmaL : sigmaR))**2)";
   return std::string("(x - x0) / sigmaL < -aL ? (") + fL + ") : ((x - x0) / sigmaR <= aR ? (" + fC + ") : (" + fR +
          ") )";
}

TEST(RooCrystalBall, SingleTailAndFullySymmetric)
{
   RooRealVar x("x", "x", 0., -200., 200.);
   RooRealVar x0("x0", "x0", 100., -200., 200.);
   RooRealVar sigma("sigma", "sigma", 2., 1.E-6, 100.);
   RooRealVar alpha("a", "a", 1., 1.E-6, 100.);
   RooRealVar n("n", "n", 1., 1.E-6, 100.);

   // To get a right tail we need to flip the sign of alpha
   RooFormulaVar minusAlpha("minusA", "-x[0]", RooArgList(alpha));

   // in the fully symmetric case, we also compare with the old implementation `RooCBShape`
   RooCrystalBall crystalBallOnlyLeftTail("cb1", "cb1", x, x0, sigma, alpha, n);
   RooCrystalBall crystalBallOnlyRightTail("cb2", "cb2", x, x0, sigma, minusAlpha, n);
   RooCrystalBall crystalBallFullySymmetric("cb3", "cb3", x, x0, sigma, alpha, n, true);

   RooCBShape crystalBallOnlyLeftTailOld("cb1Old", "cb3Old", x, x0, sigma, alpha, n);
   RooCBShape crystalBallOnlyRightTailOld("cb2Old", "cb2Old", x, x0, sigma, minusAlpha, n);
   //RooSDSCBShape crystalBallFullySymmetricOld("cb3Old", "cb3Old", x, x0, sigma, alpha, n);

   auto formulaOnlyLeftTail = makeCrystalBallFormulaOnlyLeftTail();
   auto formulaOnlyRightTail = makeCrystalBallFormulaOnlyRightTail();
   auto formulaFullySymmetric = makeCrystalBallFormulaFullySymmetric();

   // Note: Ownership bug. Deleting this might crash on Mac.
   // Therefore, it will leak because we are testing not the
   // GenericPdf.
   auto crystalBallOnlyLeftTailRef =
      new RooGenericPdf("cbBallRef", formulaOnlyLeftTail.c_str(), RooArgSet(x, x0, sigma, alpha, n));
   auto crystalBallOnlyRightTailRef =
      new RooGenericPdf("cbBallRef", formulaOnlyRightTail.c_str(), RooArgSet(x, x0, sigma, alpha, n));
   auto crystalBallFullySymmetricRef =
      new RooGenericPdf("cbBallRef", formulaFullySymmetric.c_str(), RooArgSet(x, x0, sigma, alpha, n));

   for (double theX : {-100., -50., -10., -1., 0., 1., 10., 50., 100.}) {
      for (double theX0 : {-100., -10., 0., 10., 20., 30., 100., 150.}) {
         for (double theSigma : {5., 10., 20., 50.}) {
            for (double theAlpha : {0.2, 1., 2., 10.}) {
               for (double theN : {0.2, 1., 2., 10.}) {
                  x = theX;
                  x0 = theX0;
                  sigma = theSigma;
                  alpha = theAlpha;
                  n = theN;

                  EXPECT_FLOAT_EQ(crystalBallOnlyLeftTail.getVal(), crystalBallOnlyLeftTailRef->getVal())
                     << theX << " " << theX0 << " " << theSigma << " " << theAlpha << " " << theN;

                  // Compare left tail only version with RooCBShape which should match
                  EXPECT_FLOAT_EQ(crystalBallOnlyLeftTailOld.getVal(), crystalBallOnlyLeftTailRef->getVal())
                     << theX << " " << theX0 << " " << theSigma << " " << theAlpha << " " << theN;

                  EXPECT_FLOAT_EQ(crystalBallOnlyRightTail.getVal(), crystalBallOnlyRightTailRef->getVal())
                     << theX << " " << theX0 << " " << theSigma << " " << theAlpha << " " << theN;

                  // Compare right tail only version with RooCBShape which should match
                  EXPECT_FLOAT_EQ(crystalBallOnlyRightTailOld.getVal(), crystalBallOnlyRightTailRef->getVal())
                     << theX << " " << theX0 << " " << theSigma << " " << theAlpha << " " << theN;

                  EXPECT_FLOAT_EQ(crystalBallFullySymmetric.getVal(), crystalBallFullySymmetricRef->getVal())
                     << theX << " " << theX0 << " " << theSigma << " " << theAlpha << " " << theN;

                  // Compare fully symmetric version with RooSDSCBShape which should match
                  //EXPECT_FLOAT_EQ(crystalBallFullySymmetricOld.getVal(), crystalBallFullySymmetricRef->getVal())
                  //  << theX << " " << theX0 << " " << theSigma << " " << theAlpha << " " << theN;
               }
            }
         }
      }
   }
}

TEST(RooCrystalBall, DoubleSided)
{
   RooRealVar x("x", "x", 0., -200., 200.);
   RooRealVar x0("x0", "x0", 100., -200., 200.);
   RooRealVar sigma("sigma", "sigma", 2., 1.E-6, 100.);
   RooRealVar alphaL("aL", "aL", 1., 1.E-6, 100.);
   RooRealVar alphaR("aR", "aR", 1., 1.E-6, 100.);
   RooRealVar nL("nL", "nL", 1., 1.E-6, 100.);
   RooRealVar nR("nR", "nR", 1., 1.E-6, 100.);

   // in the symmetric Gaussian core case, we also compare with the old implementation `RooDSCBShape`
   RooCrystalBall crystalBall("crystalBall", "crystalBall", x, x0, sigma, alphaL, nL, alphaR, nR);
   //RooDSCBShape crystalBallOld("crystalBallOld", "crystalBallOld", x, x0, sigma, alphaL, nL, alphaR, nR);

   auto formula = makeCrystalBallFormulaDoubleSided();

   // Note: Ownership bug. Deleting this might crash on Mac.
   // Therefore, it will leak because we are testing not the
   // GenericPdf.
   auto crystalBallRef =
      new RooGenericPdf("crystalBallRef", formula.c_str(), RooArgSet(x, x0, sigma, alphaL, nL, alphaR, nR));

   for (double theX : {-100., -50., -10., -1., 0., 1., 10., 50., 100.}) {
      for (double theX0 : {-100., -10., 0., 10., 20., 30., 100., 150.}) {
         for (double theSigma : {5., 10., 20., 50.}) {
            for (double theAlphaL : {1., 2., 10.}) {
               for (double theNL : {1., 2., 10.}) {
                  for (double theAlphaR : {1., 2., 10.}) {
                     for (double theNR : {1., 2., 10.}) {
                        x = theX;
                        x0 = theX0;
                        sigma = theSigma;
                        alphaL = theAlphaL;
                        nL = theNL;
                        alphaR = theAlphaR;
                        nR = theNR;

                        EXPECT_FLOAT_EQ(crystalBall.getVal(), crystalBallRef->getVal())
                           << theX << " " << theX0 << " " << theSigma << " " << theAlphaL << " " << theNL << " "
                           << theAlphaR << " " << theNR;

                        //EXPECT_FLOAT_EQ(crystalBallOld.getVal(), crystalBallRef->getVal())
                        //   << theX << " " << theX0 << " " << theSigma << " " << theAlphaL << " " << theNL << " "
                        //   << theAlphaR << " " << theNR;
                     }
                  }
               }
            }
         }
      }
   }
}

TEST(RooCrystalBall, FullyParametrized)
{
   MAKE_CRYSTAL_BALL_AND_VARS
   auto formula = makeCrystalBallFormula();

   // Note: Ownership bug. Deleting this might crash on Mac.
   // Therefore, it will leak because we are testing not the
   // GenericPdf.
   auto crystalBallRef =
      new RooGenericPdf("crystalBallRef", formula.c_str(), RooArgSet(x, x0, sigmaL, alphaL, nL, sigmaR, alphaR, nR));

   for (double theX : {-100., -50., -10., -1., 0., 1., 10., 50., 100.}) {
      for (double theX0 : {-100., -10., 0., 10., 20., 30., 100., 150.}) {
         for (double theSigmaL : {5., 10., 20., 50.}) {
            for (double theAlphaL : {1., 2., 10.}) {
               for (double theNL : {1., 2., 10.}) {
                  for (double theSigmaR : {5., 10., 20., 50.}) {
                     for (double theAlphaR : {1., 2., 10.}) {
                        for (double theNR : {1., 2., 10.}) {
                           x = theX;
                           x0 = theX0;
                           sigmaL = theSigmaL;
                           alphaL = theAlphaL;
                           nL = theNL;
                           sigmaR = theSigmaR;
                           alphaR = theAlphaR;
                           nR = theNR;

                           EXPECT_FLOAT_EQ(crystalBall.getVal(), crystalBallRef->getVal())
                              << theX << " " << theX0 << " " << theSigmaL << " " << theAlphaL << " " << theNL << " "
                              << theSigmaR << " " << theAlphaR << " " << theNR;
                        }
                     }
                  }
               }
            }
         }
      }
   }
}

TEST(RooCrystalBall, Integral)
{
   MAKE_CRYSTAL_BALL_AND_VARS

   x.setRange(-199., 199);

   RooCrystalBall crystalBallNumInt(crystalBall);

   RooNumIntConfig intConfig(*RooAbsReal::defaultIntegratorConfig());
   intConfig.setEpsAbs(1.E-15);
   intConfig.setEpsRel(1.E-12);

   intConfig.getConfigSection("RooIntegrator1D").setRealValue("maxSteps", 100);
   crystalBallNumInt.setIntegratorConfig(intConfig);
   crystalBallNumInt.forceNumInt(true);

   auto integral = crystalBall.createIntegral(x);
   auto integralRanged = crystalBall.createIntegral(x, "integrationRange");
   auto numInt = crystalBallNumInt.createIntegral(x);
   auto numIntRanged = crystalBallNumInt.createIntegral(x, "integrationRange");

   for (double theX0 : {0.}) {
      for (double theSigmaL : {10., 20.}) {
         for (double theAlphaL : {1.5}) {
            // it's important to have values of nL and nR close to one to hit the log computations
            for (double theNL : {1.0 + 0.9e-05, 0.5}) {
               for (double theSigmaR : {10., 20.}) {
                  for (double theAlphaR : {1., 2.}) {
                     for (double theNR : {1.0 - 0.3e-05, 0.3}) {
                        x0 = theX0;
                        sigmaL = theSigmaL;
                        alphaL = theAlphaL;
                        nL = theNL;
                        sigmaR = theSigmaR;
                        alphaR = theAlphaR;
                        nR = theNR;

                        // We want to hit all cases here to completely cover the analytical integral function.
                        // The RooCrystalBall has 4 separate definition regions:
                        //
                        // 1. Gaussian core left
                        // 2. Gaussian core right
                        // 3. left tail
                        // 4. right tail
                        //
                        // It's important that the integration range limits are in all possible combinations of these
                        // regions to cover all branches of the integration code.

                        for (double r1 : {-2., -1.5, -0.5, -0.2, 0.2, 0.5, 1.5, 2.}) {
                           for (double r2 : {-2., -1.5, -0.5, -0.2, 0.2, 0.5, 1.5, 2.}) {
                              if (r2 <= r1) {
                                 continue;
                              }
                              auto xLow = theX0 + (r1 > 0 ? theSigmaR : theSigmaL) * r1;
                              auto xHigh = theX0 + (r1 > 0 ? theSigmaR : theSigmaL) * r2;
                              x.setRange("integrationRange", xLow, xHigh);

                              constexpr double accAnaVsNum = 1.;
                              EXPECT_NEAR(integralRanged->getVal(), numIntRanged->getVal(),
                                          accAnaVsNum / 100. * numIntRanged->getVal())
                                 << "Analytical vs numerical integral"
                                 << " within " << accAnaVsNum << "%. With " << theX0 << " " << theSigmaL << " "
                                 << theAlphaL << " " << theNL << " " << theSigmaR << " " << theAlphaR << " " << theNR
                                 << " integration from " << r1 << " sigma to " << r2 << " sigma" << std::endl;
                           }
                        }

                        if (integral->getVal() > 1.E-9) { // Numerical integral cannot do this
                           constexpr double accAnaVsNum = 3.;
                           EXPECT_NEAR(integral->getVal(), numInt->getVal(),
                                       accAnaVsNum / 100. * numIntRanged->getVal())
                              << "Analytical vs numerical integral (full range) "
                              << " within " << accAnaVsNum << "%. With " << theX0 << " " << theSigmaL << " "
                              << theAlphaL << " " << theNL << " " << theSigmaR << " " << theAlphaR << " " << theNR;
                        }
                     }
                  }
               }
            }
         }
      }
   }
}

TEST(RooCrystalBall, Generator)
{
   MAKE_CRYSTAL_BALL_AND_VARS

   ASSERT_FALSE(x0.isConstant());

   x0 = 100.;
   sigmaL = 10.;
   alphaL = 1.;
   nL = 0.2;
   sigmaR = 20.;
   alphaR = 1.3;
   nR = 0.1;

   auto frame = x.frame(RooFit::Title("RooCrystalBall"));
   auto data = crystalBall.generate(x, RooFit::NumEvents(10000));
   data->plotOn(frame);
   crystalBall.plotOn(frame, RooFit::LineColor(kRed), RooFit::LineColor(kDotted));
   crystalBall.fitTo(*data, RooFit::PrintLevel(-1));
   crystalBall.plotOn(frame, RooFit::LineColor(kBlue), RooFit::LineColor(kDashed));
   crystalBall.paramOn(frame);

   EXPECT_LT(frame->chiSquare(), 1.);

   x0 = 50.;
   sigmaL = 20.;
   alphaL = 1.5;
   nL = 0.5;
   sigmaR = 10.;
   alphaR = 1.0;
   nR = 0.2;

   frame = x.frame(RooFit::Title("RooCrystalBall"));
   auto data2 = crystalBall.generate(x, 10000.);
   data2->plotOn(frame);
   crystalBall.plotOn(frame, RooFit::LineColor(kBlue));
   EXPECT_LT(frame->chiSquare(), 1.);

   auto res = crystalBall.fitTo(*data2, RooFit::Save(), RooFit::PrintLevel(-1));
   crystalBall.plotOn(frame, RooFit::LineColor(kRed), RooFit::LineStyle(kDashed));
   crystalBall.paramOn(frame);
   EXPECT_LT(frame->chiSquare(res->floatParsInit().size()), 1.);
}
