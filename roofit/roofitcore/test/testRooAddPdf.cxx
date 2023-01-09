// Tests for RooAddPdf
// Authors: Jonas Rembser, CERN 07/2022

#include <RooAddPdf.h>
#include <RooConstVar.h>
#include <RooDataHist.h>
#include <RooExponential.h>
#include <RooGaussian.h>
#include <RooHelpers.h>
#include <RooHistPdf.h>
#include <RooMsgService.h>
#include <RooProdPdf.h>
#include <RooWorkspace.h>

#include <RooStats/SPlot.h>

#include <gtest/gtest.h>

// Backward compatibility for gtest version < 1.10.0
#ifndef INSTANTIATE_TEST_SUITE_P
#define INSTANTIATE_TEST_SUITE_P INSTANTIATE_TEST_CASE_P
#endif

#include <memory>

/// Verify that sPlot does work with a RooAddPdf. This reproduces GitHub issue
/// #10869, where creating an SPlot from a RooAdPdf unreasonably changed the
/// parameter values because of RooAddPdf normalization issue. The reproducer
/// is taken from the GitHub issue thread, with the plotting part removed.
TEST(RooAddPdf, TestSPlot)
{
   RooHelpers::LocalChangeMsgLevel changeMsgLvl(RooFit::WARNING);

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
   RooHelpers::LocalChangeMsgLevel changeMsgLvl(RooFit::WARNING);

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

   RooWorkspace ws;
   ws.factory("Gaussian::g1(x[0, 0, 20], 3.0, 1.0)");
   ws.factory("Gaussian::g2(x, 9.0, 1.0)");
   ws.factory("Gaussian::g3(x, 15.0, 1.0)");

   RooRealVar &x = *ws.var("x");

   RooAbsPdf &g1 = *ws.pdf("g1");
   RooAbsPdf &g2 = *ws.pdf("g2");
   RooAbsPdf &g3 = *ws.pdf("g3");

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

class ProjCacheTest : public testing::TestWithParam<std::tuple<bool>> {
   void SetUp() override
   {
      _fixCoefNormalization = std::get<0>(GetParam());
      _changeMsgLvl = std::make_unique<RooHelpers::LocalChangeMsgLevel>(RooFit::WARNING);
   }

   void TearDown() override { _changeMsgLvl.reset(); }

protected:
   bool _fixCoefNormalization = false;
private:
   std::unique_ptr<RooHelpers::LocalChangeMsgLevel> _changeMsgLvl;
};

/// Verify that the coefficient projection works for different configurations
/// of the RooAddPdf.
TEST_P(ProjCacheTest, Test)
{
   RooRealVar x{"x", "x", 0, 4};
   x.setBins(4);
   x.setRange("B1", 0, 2);
   x.setRange("B2", 2, 4);
   x.setRange("FULL", 0, 4);

   RooDataHist dataHist1{"dh1", "dh1", x};
   RooDataHist dataHist2{"dh2", "dh2", x};

   for (int i = 0; i < 4; ++i) {
      dataHist1.set(i, i + 1, 0.0);
      dataHist2.set(i, 4. - i, 0.0);
   }

   RooHistPdf pdf1{"pdf1", "pdf1", x, dataHist1};
   RooHistPdf pdf2{"pdf2", "pdf2", x, dataHist2};

   RooRealVar frac{"frac", "frac", 0.5, 0, 1};

   RooAddPdf pdf{"pdf", "pdf", {pdf1, pdf2}, frac};

   RooArgSet normSet{x};

   x.setRange(2, 4);
   pdf.fixCoefRange("FULL");
   if (_fixCoefNormalization) {
      pdf.fixCoefNormalization(normSet);
   }

   x.setVal(2.5);
   EXPECT_FLOAT_EQ(pdf.getVal(normSet), 0.5);
   x.setVal(3.5);
   EXPECT_FLOAT_EQ(pdf.getVal(normSet), 0.5);
}

INSTANTIATE_TEST_SUITE_P(RooAddPdf, ProjCacheTest,
                         testing::Values(ProjCacheTest::ParamType{false}, ProjCacheTest::ParamType{true}),
                         [](testing::TestParamInfo<ProjCacheTest::ParamType> const &paramInfo) {
                            std::stringstream ss;
                            ss << (std::get<0>(paramInfo.param) ? "fixCoefNorm" : "floatingCoefNorm");
                            return ss.str();
                         });

/// Verify that if we change the normalization set of a server to a RooAddPdf,
/// the projection caches in the RooAddPdf are cleared correctly.
TEST(RooAddPdf, ResetServerNormRange)
{
   using namespace RooFit;

   RooHelpers::LocalChangeMsgLevel changeMsgLvl(RooFit::WARNING);

   RooWorkspace ws;
   ws.factory("PROD::sig(Polynomial(x[-10, 10], {0.0, 0.0, 1.0}, 0), Polynomial(y[-10, 10], {}))");
   ws.factory("PROD::bkg(Polynomial(x, {}), Polynomial(y, {}))");
   ws.factory("SUM::model(f[0.5, 0, 1] * sig, bkg)");
   ws.factory("SUM::modelFixed(f * sig, bkg)");
   ws.factory("EXPR::modelRef('f * 3.0/2000. * x * x + (1 - f) / 20.', {x, f})");

   RooRealVar &x = *ws.var("x");
   RooRealVar &y = *ws.var("y");

   RooAddPdf &model = static_cast<RooAddPdf &>(*ws.pdf("model"));
   RooAddPdf &modelFixed = static_cast<RooAddPdf &>(*ws.pdf("modelFixed"));
   RooAbsPdf &modelRef = *ws.pdf("modelRef");

   x.setRange("SIG", 0.0, +1.0);
   y.setRange("SIG", 0.0, +1.0);

   x.setRange("FULL", -10.0, +10.0);
   y.setRange("FULL", -10.0, +10.0);

   RooArgSet normSet1{x, y};
   RooArgSet normSet2{x, y};

   const char *fitRange = "SIG";

   // We do the comparison also for a model where the normalization is fixed
   modelFixed.fixCoefNormalization(normSet1);
   modelFixed.fixCoefRange("FULL");

   // First we set the normalization range only for the RooAddPdf...
   model.setNormRange(fitRange);
   modelFixed.setNormRange(fitRange);
   modelRef.setNormRange(fitRange);

   const double val1 = model.getVal(normSet1);
   const double val1fixed = modelFixed.getVal(normSet1);
   const double val1ref = modelRef.getVal(normSet1);

   // ... then also for its servers
   for (auto *pdf : static_range_cast<RooAbsPdf *>(ws.allPdfs())) {
      pdf->setNormRange(fitRange);
   }

   // We first use the same normalization set to re-evaluate, then a different
   // one to confuse to first trigger the chache and then have a another
   // reference.
   const double val2 = model.getVal(normSet1);
   const double val3 = model.getVal(normSet2);
   const double val2fixed = modelFixed.getVal(normSet1);
   const double val3fixed = modelFixed.getVal(normSet2);
   const double val2ref = modelRef.getVal(normSet1);
   const double val3ref = modelRef.getVal(normSet2);

   EXPECT_FLOAT_EQ(val1, val1ref);
   EXPECT_FLOAT_EQ(val1fixed, val1ref);
   EXPECT_FLOAT_EQ(val2, val2ref);
   EXPECT_FLOAT_EQ(val2fixed, val2ref);
   EXPECT_FLOAT_EQ(val3, val3ref);
   EXPECT_FLOAT_EQ(val3fixed, val3ref);
}

/// Reproduces the former issue where the ranged projection of a RooAddPdf with
/// binned PDFs got normalisation wrong:
/// https://sft.its.cern.ch/jira/browse/ROOT-10483.
TEST(RooAddPdf, ROOT10483)
{
   RooHelpers::LocalChangeMsgLevel changeMsgLvl(RooFit::WARNING);

   RooWorkspace ws;

   // Create model
   ws.factory("PROD::sig(Gaussian(x[-5,5], 0.0, 1.0), Gaussian(y[-5,5], 0.0, 1.0), Gaussian(z[-5,5], 0.0, 1.0))");
   ws.factory("PROD::bkg(Polynomial(x, {-0.1, 0.004}), Polynomial(y, {0.1, -0.004}), Polynomial(z, {}))");
   ws.factory("SUM::model(fsig[0.1, 0.0, 1.0] * sig, bkg)");

   RooAbsPdf &gz = *ws.pdf("sig_3");
   RooAbsPdf &sig = *ws.pdf("sig");
   RooAbsPdf &bkg = *ws.pdf("bkg");
   RooAbsPdf &model = *ws.pdf("model");

   RooRealVar &x = *ws.var("x");
   RooRealVar &y = *ws.var("y");
   RooRealVar &z = *ws.var("z");
   RooRealVar &fsig = *ws.var("fsig");

   const int nEvents = 20000;

   // Create RooHistPdf for signal shape
   std::unique_ptr<RooDataSet> signalData{sig.generate({x, y, z}, nEvents)};

   x.setBins(40);
   y.setBins(40);

   RooDataHist dh("dh", "binned version of d", {x, y}, *signalData);
   RooHistPdf sigXYhist("sig_xy_hist", "sig_xy_hist", {x, y}, dh);
   RooProdPdf sigHist("sig_hist", "sig_hist", {sigXYhist, gz});
   RooAddPdf modelHist("model_hist", "model_hist", {sigHist, bkg}, fsig);

   std::unique_ptr<RooDataSet> data{model.generate({x, y, z}, nEvents)};

   y.setRange("sigRegion", -1, 1);
   z.setRange("sigRegion", -1, 1);

   std::unique_ptr<RooPlot> frame{x.frame(RooFit::Bins(40))};

   data->plotOn(frame.get(), RooFit::CutRange("sigRegion"));
   modelHist.plotOn(frame.get(), RooFit::ProjectionRange("sigRegion"));

   // If the normalization in the plot is right, the reduced chi-square of the
   // the plot will be good (i.e. less than one) as the data was directly
   // sampled from the model.
   EXPECT_LE(frame->chiSquare(), 1.0);
}
