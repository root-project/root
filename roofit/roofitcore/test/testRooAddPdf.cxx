// Tests for RooAddPdf
// Authors: Jonas Rembser, CERN 07/2022

#include <RooAddPdf.h>
#include <RooConstVar.h>
#include <RooExponential.h>
#include <RooGaussian.h>
#include <RooMsgService.h>
#include <RooProdPdf.h>
#include <RooWorkspace.h>

#include <RooStats/SPlot.h>

#include <gtest/gtest.h>

#include <memory>

/// Verify that sPlot does work with a RooAddPdf. This reproduces GitHub issue
/// #10869, where creating an SPlot from a RooAdPdf unreasonably changed the
/// parameter values because of RooAddPdf normalization issue. The reproducer
/// is taken from the GitHub issue thread, with the plotting part removed.
TEST(RooAddPdf, TestSPlot)
{
   auto &msg = RooMsgService::instance();
   msg.setGlobalKillBelow(RooFit::WARNING);

   double lowRange = 0.;
   double highRange = 200.;
   RooRealVar invMass("invMass", "M_{inv}", lowRange, highRange, "GeV");
   RooRealVar isolation("isolation", "isolation", 0., 20., "GeV");

   // mass model: linear combination of two Gaussians of different widths
   RooRealVar mZ("mZ", "Z Mass", 91.2, lowRange, highRange);
   RooRealVar sigmaZ("sigmaZ", "Width of Gaussian", 2, 0.1, 10, "GeV");
   RooGaussian mZModel1("mZModel1", "Z+jets Model", invMass, mZ, sigmaZ);

   RooRealVar sigmaZ2("sigmaZ2", "Width of Gaussian", 1.5, 0.01, 10, "GeV");
   RooGaussian mZModel2("mZModel2", "Z+jets Model", invMass, mZ, sigmaZ2);

   RooRealVar frac_1a("frac_1a", "frac_1a", 0.5); // fraction of the first Gaussian

   RooArgList shapes_DCB;
   shapes_DCB.add(mZModel1);
   shapes_DCB.add(mZModel2);

   RooArgList yields_DCB;
   yields_DCB.add(frac_1a);

   RooArgSet normSet(invMass);

   RooAddPdf mZModel("mZModel", "mZModel", shapes_DCB, yields_DCB);

   // isolation model for Z-boson.  Only used to generate toy MC.

   RooConstVar zIsolDecayConst("zIsolDecayConst", "z isolation decay  constant", -1);
   RooExponential zIsolationModel("zIsolationModel", "z isolation model", isolation, zIsolDecayConst);

   // make the combined Z model
   RooProdPdf zModel("zModel", "2-d model for Z", {mZModel, zIsolationModel});

   // make the QCD model

   // mass model for QCD.

   RooRealVar qcdMassDecayConst("qcdMassDecayConst", "Decay const for QCD mass spectrum", -0.01, -100, 100, "1/GeV");
   RooExponential qcdMassModel("qcdMassModel", "qcd Mass Model", invMass, qcdMassDecayConst);

   // isolation model for QCD.
   RooConstVar qcdIsolDecayConst("qcdIsolDecayConst", "Et resolution constant", -.1);
   RooExponential qcdIsolationModel("qcdIsolationModel", "QCD isolation model", isolation, qcdIsolDecayConst);

   // make the 2D model
   RooProdPdf qcdModel("qcdModel", "2-d model for QCD", {qcdMassModel, qcdIsolationModel});

   // --------------------------------------
   // combined model

   // These variables represent the number of Z or QCD events
   // They will be fitted.
   RooRealVar zYield("zYield", "fitted yield for Z", 300, 0., 10000);
   RooRealVar qcdYield("qcdYield", "fitted yield for QCD", 2700, 0., 10000);

   // now make the combined model
   // this is the 2D model for generation only
   RooAddPdf model_gen("model_gen", "mygreatmodel_gen", {zModel, qcdModel}, {zYield, qcdYield});
   // this is the mass model for the fit only
   RooAddPdf model("model", "mygreatmodel", {mZModel, qcdMassModel}, {zYield, qcdYield});

   int nEvents = zYield.getVal() + qcdYield.getVal();

   // make the toy data
   std::unique_ptr<RooDataSet> data{model_gen.generate({invMass, isolation}, nEvents, RooFit::Name("mygendataset"))};

   // fit the model to the data.
   model.fitTo(*data, RooFit::Extended(), RooFit::PrintLevel(-1));

   double valYieldBefore = zYield.getVal();

   // fix the parameters that are not yields before doing the sPlot
   qcdMassDecayConst.setConstant(true);
   sigmaZ.setConstant(true);
   sigmaZ2.setConstant(true);
   mZ.setConstant(true);

   RooStats::SPlot("sData", "An SPlot", *data, &model, {zYield, qcdYield});

   double valYieldAfter = zYield.getVal();

   EXPECT_NEAR(valYieldAfter, valYieldBefore, 1e-1)
      << "Doing the SPlot should not change parameter values by orders of magnitudes!";
}

/// Test with the reproducer from GitHub issue #10988 that reported a caching
/// issues of RooAddPdf integrals.
TEST(RooAddPdf, Issue10988)
{
   auto &msg = RooMsgService::instance();
   msg.setGlobalKillBelow(RooFit::WARNING);

   using namespace RooFit;

   RooWorkspace w1;
   w1.factory("x[3., 0., 10.]");
   w1.var("x")->setRange("range_int", 0., 4.);
   w1.factory("AddPdf::sum(Gaussian(x, mean1[1.], sigma1[2., 0.1, 10.]),"
              "Gaussian(x, mean2[5.], sigma2[10., 0.1, 10.]), coef[0.3])");
   RooWorkspace w2(w1);

   auto &pdf1 = *static_cast<RooAddPdf *>(w1.pdf("sum"));
   auto &pdf2 = *static_cast<RooAddPdf *>(w2.pdf("sum"));
   auto &x1 = *w1.var("x");
   auto &x2 = *w2.var("x");

   // Call createIntegral on workspace w1 only. It's important that this
   // integral lives longer than "interal1" to reproduce the problem.
   std::unique_ptr<RooAbsReal> integral0{pdf1.createIntegral(x1, NormSet(x1), Range("range_int"))};
   double val0 = integral0->getVal();

   x1.setRange("fixCoefRange", 0., 1.);
   pdf1.fixCoefRange("fixCoefRange");
   x2.setRange("fixCoefRange", 0., 1.);
   pdf2.fixCoefRange("fixCoefRange");

   std::unique_ptr<RooAbsReal> integral1{pdf1.createIntegral(x1, NormSet(x1), Range("range_int"))};
   double val1 = integral1->getVal();
   std::unique_ptr<RooAbsReal> integral2{pdf2.createIntegral(x2, NormSet(x2), Range("range_int"))};
   double val2 = integral2->getVal();

   // Check that the integrals have differnt values, now that the coefficient
   // range was fixed to a different one.
   EXPECT_NE(val0, val1);
   EXPECT_EQ(val1, val2);
}

/// Check that the treatment of recursive coefficients works correctly.
TEST(RooAddPdf, RecursiveCoefficients)
{
   using namespace RooFit;

   RooRealVar x{"x", "x", 0, 0, 20};

   RooGaussian g1{"g1", "g1", x, RooConst(3.0), RooConst(1.0)};
   RooGaussian g2{"g2", "g2", x, RooConst(9.0), RooConst(1.0)};
   RooGaussian g3{"g3", "g3", x, RooConst(15.0), RooConst(1.0)};

   // A regular RooAddPdf
   RooAddPdf model1{"model1", "model1", {g1, g2, g3}, {RooConst(1. / 3), RooConst(1. / 3)}};

   // A RooAddPdf with recursive coefficients that should be mathematically equivalent
   RooAddPdf model2{"model2", "model2", {g1, g2, g3}, {RooConst(1. / 3), RooConst(0.5)}, true};

   std::unique_ptr<RooPlot> plot{x.frame()};

   model1.plotOn(plot.get(), LineColor(kBlue), Name("model1"));
   model2.plotOn(plot.get(), LineColor(kRed), LineStyle(kDashed), Name("model2"));

   auto curve1 = plot->getCurve("model1");
   auto curve2 = plot->getCurve("model2");

   EXPECT_TRUE(curve2->isIdentical(*curve1));
}
