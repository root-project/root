// Tests for RooAddPdf
// Authors: Jonas Rembser, CERN 07/2022

#include <RooAddPdf.h>
#include <RooConstVar.h>
#include <RooDataHist.h>
#include <RooFit/Evaluator.h>
#include <RooFormulaVar.h>
#include <RooGaussian.h>
#include <RooGenericPdf.h>
#include <RooHelpers.h>
#include <RooHistPdf.h>
#include <RooMsgService.h>
#include <RooPolynomial.h>
#include <RooProdPdf.h>
#include <RooRealIntegral.h>
#include <RooRealSumPdf.h>
#include <RooUniform.h>
#include <RooWorkspace.h>

#include <RooStats/SPlot.h>

#include "gtest_wrapper.h"

#include <memory>

/// Verify that sPlot does work with a RooAddPdf. This reproduces GitHub issue
/// #10869, where creating an SPlot from a RooAdPdf unreasonably changed the
/// parameter values because of RooAddPdf normalization issue. The reproducer
/// is taken from the GitHub issue thread, with the plotting part removed.
TEST(RooAddPdf, TestSPlot)
{
   RooHelpers::LocalChangeMsgLevel changeMsgLvl(RooFit::WARNING);

   RooWorkspace ws;

   // mass model: linear combination of two Gaussians of different widths
   ws.factory("Gaussian::mZModel1(invMass[0., 200.], mZ[91.2, 0., 200.], sigmaZ[2, 0.1, 10])");
   ws.factory("Gaussian::mZModel2(invMass[0., 200.], mZ[91.2, 0., 200.], sigmaZ2[1.5, 0.01, 10])");
   ws.factory("SUM::mZModel(frac_1a[0.5] * mZModel1, mZModel2)");

   // isolation model for Z-boson.  Only used to generate toy MC.
   ws.factory("Exponential::zIsolationModel(isolation[0., 20.], zIsolDecayConst[-1])");

   // make the combined Z model
   ws.factory("PROD::zModel(mZModel, zIsolationModel)");

   // mass model for QCD.
   ws.factory("Exponential::qcdMassModel(invMass, qcdMassDecayConst[-0.01, -100, 100])");

   // isolation model for QCD.
   ws.factory("Exponential::qcdIsolationModel(isolation, qcdIsolDecayConst[-0.1])");

   // make the 2D QCD model
   ws.factory("PROD::qcdModel(qcdMassModel, qcdIsolationModel)");

   // now make the combined model
   // this is the 2D model for generation only
   ws.factory("SUM::model_gen(zYield[300, 0., 10000] * zModel, qcdYield[2700, 0., 10000] * qcdModel)");
   // this is the mass model for the fit only
   ws.factory("SUM::model(zYield[300, 0., 10000] * mZModel, qcdYield[2700, 0., 10000] * qcdMassModel)");

   RooRealVar &invMass = *ws.var("invMass");
   RooRealVar &isolation = *ws.var("isolation");
   RooRealVar &zYield = *ws.var("zYield");
   RooRealVar &qcdYield = *ws.var("qcdYield");
   RooAbsPdf &model_gen = *ws.pdf("model_gen");
   RooAbsPdf &model = *ws.pdf("model");

   RooArgSet normSet{invMass};

   // make the toy data
   std::unique_ptr<RooDataSet> data{model_gen.generate({invMass, isolation}, RooFit::Name("mygendataset"))};

   // fit the model to the data.
   model.fitTo(*data, RooFit::Extended(), RooFit::PrintLevel(-1));

   const double valYieldBefore = zYield.getVal();

   // fix the parameters that are not yields before doing the sPlot
   ws.var("qcdMassDecayConst")->setConstant(true);
   ws.var("sigmaZ")->setConstant(true);
   ws.var("sigmaZ2")->setConstant(true);
   ws.var("mZ")->setConstant(true);

   RooStats::SPlot("sData", "An SPlot", *data, &model, {zYield, qcdYield});

   const double valYieldAfter = zYield.getVal();

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
   // integral lives longer than "integral1" to reproduce the problem.
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

   // Check that the integrals have different values, now that the coefficient
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
   // one to confuse to first trigger the cache and then have a another
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

/// If you add components where each component only depends on a subset of the
/// union set of the observables, the RooAddPdf should understand that the
/// component is uniform in the missing observables. This is validated in the
/// following test for both the getVal() interface and evaluation with the
/// RooFit::Evaluator.
TEST(RooAddPdf, ImplicitDimensions)
{
   RooHelpers::LocalChangeMsgLevel changeMsgLvl(RooFit::WARNING);

   using namespace RooFit;

   RooRealVar x{"x", "x", 5, 0, 10};
   RooRealVar y{"y", "y", 5, 0, 10};

   // Define the PDFs such that they explicitly depend on only one of the
   // observables each. This means the RooAddPdf needs to figure out itself
   // that in the other dimensions (y for uniformX and x for uniformY), the
   // distribution is implicitly uniform.
   RooGenericPdf uniformX{"uniform_x", "x - x + 1.0", x};
   RooGenericPdf uniformY{"uniform_y", "y - y + 1.0", y};

   RooAddPdf pdf{"pdf", "pdf", {uniformX, uniformY}, {RooConst(0.5)}};

   RooArgSet normSet{x, y};

   std::unique_ptr<RooAbsReal> pdfCompiled{RooFit::Detail::compileForNormSet(pdf, normSet)};
   RooFit::Evaluator evaluator{*pdfCompiled};

   pdf.fixAddCoefNormalization(normSet);
   pdfCompiled->fixAddCoefNormalization(normSet);

   // A uniform distribution in x and y should have this value everywhere,
   // given our limits for x and y.
   const double refVal = 0.01;

   EXPECT_DOUBLE_EQ(pdf.getVal(normSet), refVal);
   EXPECT_DOUBLE_EQ(evaluator.run()[0], refVal);
}

// Make sure that there are no superfluous integrals when one does a ranged
// with equivalent coefficient reference range.
// Covers GitHub issue 12645.
TEST(RooAddPdf, IntegralsForRangedFitWithIdenticalCoefRange)
{
   RooHelpers::LocalChangeMsgLevel changeMsgLvl(RooFit::WARNING);

   using namespace RooFit;

   RooRealVar x("x", "", 0, 1);

   RooGaussian g("g", "", x, 0.5, 0.2);
   RooUniform u("u", "", x);

   RooRealVar f("f", "", 0.5, 0, 1);
   RooAddPdf a("a", "", {g, u}, {f});

   std::unique_ptr<RooAbsData> dt{a.generate(x, 1000)};

   x.setRange("limited", 0.2, 0.8);

   std::unique_ptr<RooAbsReal> nll{a.createNLL(*dt, Range("limited"), SumCoefRange("limited"), EvalBackend("cpu"))};

   RooArgList nodes;
   nll->treeNodeServerList(&nodes);

   int iIntegrals = 0;

   for (auto *arg : nodes) {
      if (dynamic_cast<RooRealIntegral const *>(arg)) {
         ++iIntegrals;
      }
   }

   // We expect only two integral objects: one normalization integral for each
   // of the component pdfs.
   EXPECT_EQ(iIntegrals, 2);
}

/// Test that we get the right expectedEvents() in conditional fits when
/// getting the expected number of events from the coefficients.
TEST(RooAddPdf, ConditionalExpectedEventsFromCoefs)
{
   RooHelpers::LocalChangeMsgLevel changeMsgLvl(RooFit::WARNING);

   using namespace RooFit;

   const double yInterval = 5.0;

   // Create observables
   RooRealVar x("x", "x", 0, 5);
   RooRealVar y("y", "y", 0, yInterval);

   // Create uniform signal and background
   RooPolynomial gx("gx", "gx", x);
   RooPolynomial gy("gy", "gy", y);
   RooProdPdf sig("sig", "sig", RooArgSet(gx, gy));
   RooPolynomial ux("ux", "ux", x);
   RooPolynomial uy("uy", "uy", y);
   RooProdPdf bkg("bkg", "bkg", RooArgSet(ux, uy));

   // Create composite pdf sig+bkg
   RooRealVar nsig("nsig", "", 100, 0., 1000.);
   RooRealVar nbkg("nbkg", "", 1000, 0., 10000.);
   RooAddPdf model("model", "model", {sig, bkg}, {nsig, nbkg});

   RooArgSet nsetX{x};
   RooArgSet nsetXY{x, y};

   // As necessary for conditional fits, we need to fix the coefficient
   // normalization set to the union set of the observables and conditional
   // observables.
   model.fixAddCoefNormalization(nsetXY);

   // Test both the method to get expectedEvents directly, and the method that
   // returns a RooAbsReal representing the expected number of events.
   const double nExpected = model.expectedEvents(&nsetXY);
   std::unique_ptr<RooAbsReal> nExpectedArg = model.createExpectedEventsFunc(&nsetXY);

   // In conditional fits, the conditional observable is taken out of the
   // normalization set.
   const double nExpectedConditional = model.expectedEvents(&nsetX);
   std::unique_ptr<RooAbsReal> nExpectedConditionalArg = model.createExpectedEventsFunc(&nsetX);

   // Since we don't integrate the uniform expected events over the conditional
   // observable y, we expect there to be a factor ymax - ymin.
   EXPECT_DOUBLE_EQ(nExpectedConditional * yInterval, nExpected);

   EXPECT_DOUBLE_EQ(nExpectedArg->getVal(), nExpected);
   EXPECT_DOUBLE_EQ(nExpectedConditionalArg->getVal(), nExpectedConditional);
}

/// Test that we get the right expectedEvents() in conditional fits when
/// getting the expected number of events from the component pdfs.
TEST(RooAddPdf, ConditionalExpectedEventsFromPdfs)
{
   RooHelpers::LocalChangeMsgLevel changeMsgLvl(RooFit::WARNING);

   using namespace RooFit;

   // Observables
   RooRealVar x("x", "x", 0, 5);
   RooRealVar y("y", "y", 0, 5);

   // Yield functions: uniform yield that articficially depends on x and y
   // because RooRealSumPdf::expectedEvents() uses the normalization integrals
   // as the expected events via getNorm(nset). As getNorm() strips away the
   // observables the pdf doesn't depend on (different from
   // createIntegral(nset)), we would not get the desired expected events of
   // xmax-xmin and ymax-ymin if we would not do this trick.
   RooFormulaVar yieldSig{"yield_sig", "1 + x - x + y - y", {x, y}};
   RooFormulaVar yieldBkg{"yield_bkg", "1 + x - x + y - y", {x, y}};

   RooRealSumPdf pdfSig{"pdf_sig", "", yieldSig, RooArgList{1.0}, true};
   RooRealSumPdf pdfBkg{"pdf_bkg", "", yieldBkg, RooArgList{1.0}, true};

   RooAddPdf pdf{"pdf", "", {pdfSig, pdfBkg}};

   RooArgSet nsetX{x};
   RooArgSet nsetXY{x, y};

   // As necessary for conditional fits, we need to fix the coefficient
   // normalization set to the union set of the observables and conditional
   // observables.
   pdf.fixAddCoefNormalization(nsetXY);

   // Test both the method to get expectedEvents directly, and the method that
   // returns a RooAbsReal representing the expected number of events.
   double nSig = pdfSig.expectedEvents(&nsetXY);
   double nBkg = pdfBkg.expectedEvents(&nsetXY);
   double nSum = pdf.expectedEvents(&nsetXY);
   std::unique_ptr<RooAbsReal> nSigArg = pdfSig.createExpectedEventsFunc(&nsetXY);
   std::unique_ptr<RooAbsReal> nBkgArg = pdfBkg.createExpectedEventsFunc(&nsetXY);
   std::unique_ptr<RooAbsReal> nSumArg = pdf.createExpectedEventsFunc(&nsetXY);

   // In conditional fits, the conditional observable is taken out of the
   // normalization set.
   double nSigCond = pdfSig.expectedEvents(&nsetX);
   double nBkgCond = pdfBkg.expectedEvents(&nsetX);
   double nSumCond = pdf.expectedEvents(&nsetX);
   std::unique_ptr<RooAbsReal> nSigCondArg = pdfSig.createExpectedEventsFunc(&nsetX);
   std::unique_ptr<RooAbsReal> nBkgCondArg = pdfBkg.createExpectedEventsFunc(&nsetX);
   std::unique_ptr<RooAbsReal> nSumCondArg = pdf.createExpectedEventsFunc(&nsetX);

   // We expect that the expected events of the AddPdf is the sum of the
   // components expectedEvents(), for both normalization sets.
   EXPECT_DOUBLE_EQ(nSum, nSig + nBkg);
   EXPECT_DOUBLE_EQ(nSumCond, nSigCond + nBkgCond);

   EXPECT_DOUBLE_EQ(nSigArg->getVal(), nSig);
   EXPECT_DOUBLE_EQ(nBkgArg->getVal(), nBkg);
   EXPECT_DOUBLE_EQ(nSumArg->getVal(), nSum);
   EXPECT_DOUBLE_EQ(nSigCondArg->getVal(), nSigCond);
   EXPECT_DOUBLE_EQ(nBkgCondArg->getVal(), nBkgCond);
   EXPECT_DOUBLE_EQ(nSumCondArg->getVal(), nSumCond);
}
