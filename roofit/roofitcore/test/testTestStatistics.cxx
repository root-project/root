// Tests for RooNLLVar and the other test statistics
// Authors: Stephan Hageboeck, CERN 10/2020
//          Jonas Rembser, CERN 10/2022

#include <RooBinning.h>
#include <RooCategory.h>
#include <RooConstVar.h>
#include <RooDataHist.h>
#include <RooDataSet.h>
#include <RooExtendPdf.h>
#include <RooFitResult.h>
#include <RooGlobalFunc.h>
#include <RooHelpers.h>
#include <RooHistFunc.h>
#include <RooHistPdf.h>
#ifdef ROOFIT_LEGACY_EVAL_BACKEND
#include "../src/RooNLLVar.h"
#endif
#include <RooPlot.h>
#include <RooPolyVar.h>
#include <RooProdPdf.h>
#include <RooRandom.h>
#include <RooRealSumPdf.h>
#include <RooRealVar.h>
#include <RooSimultaneous.h>
#include <RooWorkspace.h>

#include <TMath.h>

#include "gtest_wrapper.h"

#include <cmath>
#include <memory>

namespace {

double getVal(const char *name, const RooArgSet &set)
{
   return static_cast<const RooRealVar &>(set[name]).getVal();
}

double getErr(const char *name, const RooArgSet &set)
{
   return static_cast<const RooRealVar &>(set[name]).getError();
}

std::unique_ptr<RooDataHist> generateBinnedAsimov(RooAbsPdf const &pdf, RooRealVar &x, int nEvents)
{
   auto dataH = std::make_unique<RooDataHist>("dataH", "dataH", RooArgSet{x});
   RooAbsBinning &xBinning = x.getBinning();
   for (int iBin = 0; iBin < x.numBins(); ++iBin) {
      x.setRange("bin", xBinning.binLow(iBin), xBinning.binHigh(iBin));
      std::unique_ptr<RooAbsReal> integ{pdf.createIntegral(x, RooFit::NormSet(x), RooFit::Range("bin"))};
      integ->getVal();
      dataH->set(iBin, nEvents * integ->getVal(), -1);
   }
   return dataH;
}

} // namespace

class TestStatisticTest : public testing::TestWithParam<std::tuple<RooFit::EvalBackend>> {
public:
   TestStatisticTest() : _evalBackend{RooFit::EvalBackend::Legacy()} {}

private:
   void SetUp() override
   {
      RooRandom::randomGenerator()->SetSeed(1337ul);
      _evalBackend = std::get<0>(GetParam());
      _changeMsgLvl = std::make_unique<RooHelpers::LocalChangeMsgLevel>(RooFit::WARNING);
   }

   void TearDown() override { _changeMsgLvl.reset(); }

protected:
   RooFit::EvalBackend _evalBackend;

private:
   std::unique_ptr<RooHelpers::LocalChangeMsgLevel> _changeMsgLvl;
};

TEST_P(TestStatisticTest, IntegrateBins)
{
   RooWorkspace ws;
   ws.factory("PowerSum::pow(x[0.1, 5.1], {1.0}, {a[-0.3, -5., 5.]})");

   RooRealVar &x = *ws.var("x");
   RooRealVar &a = *ws.var("a");
   RooAbsPdf &pdf = *ws.pdf("pow");

   x.setBins(10);

   RooArgSet targetValues;
   RooArgSet(a).snapshot(targetValues);

   using namespace RooFit;

   std::unique_ptr<RooDataHist> dataH(generateBinnedAsimov(pdf, x, 10000));
   auto dataS = std::make_unique<RooDataSet>("data", "data", x, Import(*dataH));

   std::unique_ptr<RooPlot> frame(x.frame());
   dataH->plotOn(frame.get(), MarkerColor(kRed));
   dataS->plotOn(frame.get(), Name("data"));

   a.setVal(3.);
   std::unique_ptr<RooFitResult> fit1(pdf.fitTo(*dataS, Save(), PrintLevel(-1), _evalBackend, SumW2Error(false)));
   pdf.plotOn(frame.get(), LineColor(kRed), Name("standard"));

   a.setVal(3.);
   std::unique_ptr<RooFitResult> fit2(
      pdf.fitTo(*dataS, Save(), PrintLevel(-1), _evalBackend, SumW2Error(false), IntegrateBins(1.E-3)));
   pdf.plotOn(frame.get(), LineColor(kBlue), Name("highRes"));

   EXPECT_GT(std::abs(getVal("a", targetValues) - getVal("a", fit1->floatParsFinal())),
             1. * getErr("a", fit1->floatParsFinal()))
      << "Expecting a bias when sampling PDF in bin centre.";

   EXPECT_NEAR(getVal("a", targetValues), getVal("a", fit2->floatParsFinal()), 1. * getErr("a", fit2->floatParsFinal()))
      << "Expect reduced bias with high-resolution sampling.";

   EXPECT_GT(frame->chiSquare("standard", "data", 1) * 0.9, frame->chiSquare("highRes", "data", 1))
      << "Expect chi2/ndf at least 10% better.";
}

/// Prepare a RooDataSet that looks like the one that HistFactory uses:
/// It pretends to be an unbinned dataset, but instead of single events,
/// events are aggregated in the bin centres using weights.
TEST_P(TestStatisticTest, IntegrateBins_SubRange)
{
   RooWorkspace ws;
   ws.factory("PowerSum::pow(x[0.1, 5.1], {1.0}, {a[-0.3, -5., 5.]})");

   RooRealVar &x = *ws.var("x");
   RooRealVar &a = *ws.var("a");
   RooAbsPdf &pdf = *ws.pdf("pow");

   x.setBins(10);
   x.setRange("range", 0.1, 4.1);
   x.setBins(8, "range"); // consistent binning

   RooArgSet targetValues;
   RooArgSet(a).snapshot(targetValues);

   using namespace RooFit;

   std::unique_ptr<RooDataHist> dataH(generateBinnedAsimov(pdf, x, 10000));
   auto dataS = std::make_unique<RooDataSet>("data", "data", x, Import(*dataH));

   std::unique_ptr<RooPlot> frame(x.frame());
   dataH->plotOn(frame.get(), MarkerColor(kRed));
   dataS->plotOn(frame.get(), Name("data"));

   a.setVal(3.);
   std::unique_ptr<RooFitResult> fit1(
      pdf.fitTo(*dataS, Save(), PrintLevel(-1), Optimize(0), Range("range"), _evalBackend, SumW2Error(false)));
   pdf.plotOn(frame.get(), LineColor(kRed), Name("standard"), Range("range"), NormRange("range"));

   a.setVal(3.);
   std::unique_ptr<RooFitResult> fit2(pdf.fitTo(*dataS, Save(), PrintLevel(-1), Optimize(0), Range("range"),
                                                _evalBackend, SumW2Error(false), IntegrateBins(1.E-3)));
   pdf.plotOn(frame.get(), LineColor(kBlue), Name("highRes"), Range("range"), NormRange("range"));

   EXPECT_GT(std::abs(getVal("a", targetValues) - getVal("a", fit1->floatParsFinal())),
             1. * getErr("a", fit1->floatParsFinal()))
      << "Expecting a bias when sampling PDF in bin centre.";

   EXPECT_NEAR(getVal("a", targetValues), getVal("a", fit2->floatParsFinal()), 1. * getErr("a", fit2->floatParsFinal()))
      << "Expect reduced bias with high-resolution sampling.";

   EXPECT_GT(frame->chiSquare("standard", "data", 1) * 0.9, frame->chiSquare("highRes", "data", 1))
      << "Expect chi2/ndf at least 10% better.";
}

/// Prepare a RooDataSet that looks like the one that HistFactory uses:
/// It pretends to be an unbinned dataset, but instead of single events,
/// events are aggregated in the bin centres using weights.
TEST_P(TestStatisticTest, IntegrateBins_CustomBinning)
{
   RooWorkspace ws;
   ws.factory("PowerSum::pow(x[1.0, 5.], {1.0}, {a[-0.3, -5., 5.]})");

   RooRealVar &x = *ws.var("x");
   RooRealVar &a = *ws.var("a");
   RooAbsPdf &pdf = *ws.pdf("pow");

   RooBinning binning(1., 5.);
   binning.addBoundary(1.5);
   binning.addBoundary(2.0);
   binning.addBoundary(3.);
   binning.addBoundary(4.);
   x.setBinning(binning);

   RooArgSet targetValues;
   RooArgSet(a).snapshot(targetValues);

   using namespace RooFit;

   std::unique_ptr<RooDataHist> dataH(generateBinnedAsimov(pdf, x, 50000));
   auto dataS = std::make_unique<RooDataSet>("data", "data", x, Import(*dataH));

   std::unique_ptr<RooPlot> frame(x.frame());
   dataH->plotOn(frame.get(), Name("dataHist"), MarkerColor(kRed));
   dataS->plotOn(frame.get(), Name("data"));

   a.setVal(3.);
   std::unique_ptr<RooFitResult> fit1(
      pdf.fitTo(*dataS, Save(), PrintLevel(-1), _evalBackend, SumW2Error(false), Optimize(0)));
   pdf.plotOn(frame.get(), LineColor(kRed), Name("standard"));

   a.setVal(3.);
   std::unique_ptr<RooFitResult> fit2(
      pdf.fitTo(*dataS, Save(), PrintLevel(-1), Optimize(0), _evalBackend, SumW2Error(false), IntegrateBins(1.E-3)));
   pdf.plotOn(frame.get(), LineColor(kBlue), Name("highRes"));

   EXPECT_GT(std::abs(getVal("a", targetValues) - getVal("a", fit1->floatParsFinal())),
             1. * getErr("a", fit1->floatParsFinal()))
      << "Expecting a bias when sampling PDF in bin centre.";

   EXPECT_NEAR(getVal("a", targetValues), getVal("a", fit2->floatParsFinal()), 1. * getErr("a", fit2->floatParsFinal()))
      << "Expect reduced bias with high-resolution sampling.";

   // Note: We cannot compare with the unbinned dataset here, because when it's plotted, it's filled into a
   // histogram with uniform binning. It therefore creates a jumpy distribution. When comparing with the original
   // data hist, we don't get those jumps.
   EXPECT_GT(frame->chiSquare("standard", "dataHist", 1) * 0.9, frame->chiSquare("highRes", "dataHist", 1))
      << "Expect chi2/ndf at least 10% better.";
}

/// Test the same, but now with RooDataHist. Here, the feature should switch on automatically.
TEST_P(TestStatisticTest, IntegrateBins_RooDataHist)
{
   RooWorkspace ws;
   ws.factory("PowerSum::pow(x[0.1, 5.0], {1.0}, {a[-0.3, -5., 5.]})");

   RooRealVar &x = *ws.var("x");
   RooRealVar &a = *ws.var("a");
   RooAbsPdf &pdf = *ws.pdf("pow");

   x.setBins(10);

   RooArgSet targetValues;
   RooArgSet(a).snapshot(targetValues);

   using namespace RooFit;

   std::unique_ptr<RooDataHist> data(generateBinnedAsimov(pdf, x, 10000));

   std::unique_ptr<RooPlot> frame(x.frame());
   data->plotOn(frame.get(), Name("data"));

   a.setVal(3.);
   // Disable IntegrateBins forcefully
   std::unique_ptr<RooFitResult> fit1(
      pdf.fitTo(*data, Save(), PrintLevel(-1), _evalBackend, SumW2Error(false), IntegrateBins(-1.)));
   pdf.plotOn(frame.get(), LineColor(kRed), Name("standard"));

   a.setVal(3.);
   // Auto-enable IntegrateBins for all RooDataHists.
   std::unique_ptr<RooFitResult> fit2(
      pdf.fitTo(*data, Save(), PrintLevel(-1), _evalBackend, SumW2Error(false), IntegrateBins(0.)));
   pdf.plotOn(frame.get(), LineColor(kBlue), Name("highRes"));

   EXPECT_GT(std::abs(getVal("a", targetValues) - getVal("a", fit1->floatParsFinal())),
             1. * getErr("a", fit1->floatParsFinal()))
      << "Expecting a bias when sampling PDF in bin centre.";

   EXPECT_NEAR(getVal("a", targetValues), getVal("a", fit2->floatParsFinal()), 1. * getErr("a", fit2->floatParsFinal()))
      << "Expect reduced bias with high-resolution sampling.";

   EXPECT_GT(frame->chiSquare("standard", "data", 1) * 0.9, frame->chiSquare("highRes", "data", 1))
      << "Expect chi2/ndf at least 10% better.";
}

// Verify that fitting an empty RooDataSet or a RooDataHist with only empty
// bins does not do anything to the parameters. The point of this test is to
// validate that the new CPU backend behaves the same as the legacy evaluation
// backend for empty data objects.
TEST_P(TestStatisticTest, EmptyData)
{
   RooWorkspace ws;
   ws.factory("Gaussian::model(x[0, 10], mean[6, 0, 10], sigma[2.0, 0.01, 10.0])");

   RooRealVar &x = *ws.var("x");
   RooRealVar &mean = *ws.var("mean");
   RooRealVar &sigma = *ws.var("sigma");
   RooAbsPdf &model = *ws.pdf("model");

   const double meanOrigVal = mean.getVal();
   const double sigmaOrigVal = sigma.getVal();

   std::unique_ptr<RooDataSet> data{model.generate(x, 0)};
   std::unique_ptr<RooDataHist> dataHist{data->binnedClone()};

   model.fitTo(*data, _evalBackend, RooFit::PrintLevel(-1));

   EXPECT_EQ(mean.getVal(), meanOrigVal) << "Fitting an empty RooDataSet changed \"mean\" value!";
   EXPECT_EQ(sigma.getVal(), sigmaOrigVal) << "Fitting an empty RooDataSet changed \"sigma\" value!";

   // Reset the parameters for the check with the RooDataHist
   mean.setVal(meanOrigVal);
   sigma.setVal(sigmaOrigVal);
   mean.setError(0.0);
   sigma.setError(0.0);

   model.fitTo(*dataHist, _evalBackend, RooFit::PrintLevel(-1));

   EXPECT_EQ(mean.getVal(), meanOrigVal) << "Fitting an empty RooDataSet changed \"mean\" value!";
   EXPECT_EQ(sigma.getVal(), sigmaOrigVal) << "Fitting an empty RooDataSet changed \"sigma\" value!";
}

TEST(RooChi2Var, IntegrateBins)
{
   RooHelpers::LocalChangeMsgLevel changeMsgLvl(RooFit::WARNING);

   RooRandom::randomGenerator()->SetSeed(1337ul);

   RooWorkspace ws;
   ws.factory("PowerSum::pow(x[0.1, 5.1], {1.0}, {a[-0.3, -5., 5.]})");

   RooRealVar &x = *ws.var("x");
   RooRealVar &a = *ws.var("a");
   RooAbsPdf &pdf = *ws.pdf("pow");

   x.setBins(10);

   RooArgSet targetValues;
   RooArgSet(a).snapshot(targetValues);

   using namespace RooFit;

   std::unique_ptr<RooDataHist> dataH(generateBinnedAsimov(pdf, x, 10000));

   std::unique_ptr<RooPlot> frame(x.frame());
   dataH->plotOn(frame.get(), MarkerColor(kRed));

   a.setVal(3.);
   std::unique_ptr<RooFitResult> fit1(pdf.chi2FitTo(*dataH, Save(), PrintLevel(-1)));
   pdf.plotOn(frame.get(), LineColor(kRed), Name("standard"));

   a.setVal(3.);
   std::unique_ptr<RooFitResult> fit2(pdf.chi2FitTo(*dataH, Save(), PrintLevel(-1), IntegrateBins(1.E-3)));
   pdf.plotOn(frame.get(), LineColor(kBlue), Name("highRes"));

   EXPECT_GT(std::abs(getVal("a", targetValues) - getVal("a", fit1->floatParsFinal())),
             1. * getErr("a", fit1->floatParsFinal()))
      << "Expecting a bias when sampling PDF in bin centre.";

   EXPECT_NEAR(getVal("a", targetValues), getVal("a", fit2->floatParsFinal()), 1. * getErr("a", fit2->floatParsFinal()))
      << "Expect reduced bias with high-resolution sampling.";

   EXPECT_GT(frame->chiSquare("standard", nullptr, 1) * 0.9, frame->chiSquare("highRes", nullptr, 1))
      << "Expect chi2/ndf at least 10% better.";
}

#ifdef ROOFIT_LEGACY_EVAL_BACKEND
/// Verifies that a ranged RooNLLVar has still the correct value when copied,
/// as it happens when it is plotted Covers JIRA ticket ROOT-9752.
TEST(RooNLLVar, CopyRangedNLL)
{
   RooHelpers::LocalChangeMsgLevel changeMsgLvl(RooFit::WARNING);

   RooWorkspace ws;
   ws.factory("Gaussian::model(x[0, 10], mean[5, 0, 10], sigma[0.5, 0.01, 5.0])");

   RooRealVar &x = *ws.var("x");
   RooAbsPdf &model = *ws.pdf("model");

   x.setRange("fitrange", 0, 10);

   std::unique_ptr<RooDataSet> ds{model.generate(x, 20)};

   // This bug is related to the implementation details of the old test
   // statistics, so the EvalBackend is forced to be Legacy
   using namespace RooFit;
   std::unique_ptr<RooAbsReal> nll{model.createNLL(*ds, EvalBackend::Legacy())};
   std::unique_ptr<RooAbsReal> nllrange{model.createNLL(*ds, Range("fitrange"), EvalBackend::Legacy())};

   auto nllClone = std::make_unique<RooNLLVar>(static_cast<RooNLLVar &>(*nll));
   auto nllrangeClone = std::make_unique<RooNLLVar>(static_cast<RooNLLVar &>(*nllrange));

   EXPECT_FLOAT_EQ(nll->getVal(), nllClone->getVal());
   EXPECT_FLOAT_EQ(nll->getVal(), nllrange->getVal());
   EXPECT_FLOAT_EQ(nllrange->getVal(), nllrangeClone->getVal());
}
#endif

/// When using the Integrate() command argument in chi2FitTo, the result should
/// be identical to a fit without bin integration if the fit function is
/// linear. This is a good cross check to see if the integration works.
/// Inspired by the rf609_xychi2fit tutorial.
TEST(RooXYChi2Var, IntegrateLinearFunction)
{
   using namespace RooFit;

   // Make weighted XY dataset with asymmetric errors stored The StoreError()
   // argument is essential as it makes the dataset store the error in addition
   // to the values of the observables. If errors on one or more observables
   // are asymmetric, one can store the asymmetric error using the
   // StoreAsymError() argument
   RooRealVar x("x", "x", -11, 11);
   RooRealVar y("y", "y", -10, 200);
   RooDataSet dxy("dxy", "dxy", {x, y}, StoreError({x, y}));

   const double aTrue = 0.1;
   const double bTrue = 10.0;

   // Fill an example dataset with X,err(X),Y,err(Y) values
   for (int i = 0; i <= 10; i++) {

      // Set X value and error
      x = -10 + 2 * i;
      x.setError(i < 5 ? 0.5 / 1. : 1.0 / 1.);

      // Set Y value and error
      y = aTrue * x.getVal() + bTrue;
      y.setError(std::sqrt(y.getVal()));

      dxy.add({x, y});
   }

   // Make linear fit function
   RooRealVar a("a", "a", 0.0, -10, 10);
   RooRealVar b("b", "b", 0.0, -100, 100);
   RooArgList coefs{b, a};
   RooPolyVar f("f", "f", x, coefs);

   RooArgSet savedValues;
   coefs.snapshot(savedValues);

   // Fit chi^2 using X and Y errors
   std::unique_ptr<RooFitResult> fit1{f.chi2FitTo(dxy, YVar(y), Save(), PrintLevel(-1), Optimize(0))};

   coefs.assign(savedValues);
   // Alternative: fit chi^2 integrating f(x) over ranges defined by X errors,
   // rather than taking point at center of bin
   std::unique_ptr<RooFitResult> fit2{f.chi2FitTo(dxy, YVar(y), Integrate(true), Save(), PrintLevel(-1), Optimize(0))};

   // Verify that the fit result is compatible with true values within the error
   EXPECT_NEAR(getVal("a", fit1->floatParsFinal()), aTrue, getErr("a", fit1->floatParsFinal()));
   EXPECT_NEAR(getVal("b", fit1->floatParsFinal()), bTrue, getErr("b", fit1->floatParsFinal()));

   EXPECT_NEAR(getVal("a", fit2->floatParsFinal()), aTrue, getErr("a", fit2->floatParsFinal()));
   EXPECT_NEAR(getVal("b", fit2->floatParsFinal()), bTrue, getErr("b", fit2->floatParsFinal()));
}

class OffsetBinTest : public testing::TestWithParam<std::tuple<RooFit::EvalBackend, bool, bool, bool, bool, bool>> {
public:
   OffsetBinTest() : _evalBackend{RooFit::EvalBackend::Legacy()} {}

private:
   void SetUp() override
   {
      _changeMsgLvl = std::make_unique<RooHelpers::LocalChangeMsgLevel>(RooFit::WARNING);
      _evalBackend = std::get<0>(GetParam());
      _binned = std::get<1>(GetParam());
      _ext = std::get<2>(GetParam());
      _sumw2 = std::get<3>(GetParam());
      _simPdf = std::get<4>(GetParam());
      _binnedL = std::get<5>(GetParam());
   }

   void TearDown() override { _changeMsgLvl.reset(); }

protected:
   RooFit::EvalBackend _evalBackend;
   bool _binned = false;
   bool _ext = false;
   bool _sumw2 = false;
   bool _simPdf = false;
   bool _binnedL = false;

private:
   std::unique_ptr<RooHelpers::LocalChangeMsgLevel> _changeMsgLvl;
};

// Test the `Offset("bin")` feature of RooAbsPdf::createNLL. Doing the
// bin-by-bin offset is equivalent to calculating the likelihood ratio with the
// NLL of a template histogram that is based of the dataset, so we use this
// relation to do a cross check: if we create a template pdf from the fit data
// and fit this template to the data with the `Offset("bin")` option, the
// resulting NLL should always be zero (within some numerical errors).
TEST_P(OffsetBinTest, CrossCheck)
{
   using namespace RooFit;
   using RealPtr = std::unique_ptr<RooAbsReal>;

   // Create extended PDF model
   RooWorkspace ws;
   ws.factory("Gaussian::gauss(x[-10, 10], mean[0, -10, 10], sigma[4, 0.1, 10])");
   ws.factory("AddPdf::extGauss({gauss}, {nEvents[10000, 100, 100000]})");

   RooRealVar &x = *ws.var("x");
   RooRealVar &nEvents = *ws.var("nEvents");
   RooAbsPdf &extGauss = *ws.pdf("extGauss");

   // To make sure that the bin width is unity and we don't have to worry about
   // bin width corrections in the test model.
   x.setBins(20);

   // We have to generate double the number of events because in the next step
   // we will weight down each event by a factor of two.
   std::unique_ptr<RooDataSet> data{extGauss.generate(x, 2. * nEvents.getVal())};

   // Replace dataset with a clone where the weights are different from unity
   // such that the effect of the SumW2Error option is not trivial and we test
   // it correctly.
   {
      // Create weighted dataset and hist to test SumW2 feature
      auto dataW = std::make_unique<RooDataSet>("dataW", "dataW", x, RooFit::WeightVar());
      for (int i = 0; i < data->numEntries(); ++i) {
         dataW->add(*data->get(i), 0.5);
      }
      std::swap(dataW, data);
   }

   std::unique_ptr<RooDataHist> hist{data->binnedClone()};

   // Create template PDF based on data.
   // For the extended case, we use a RooExtendPdf, and for the non-extended
   // case a RooRealSumPdf so we can also test the binned likelihood code path.
   RooHistPdf histPdf{"histPdf", "histPdf", x, *hist};
   RooHistFunc histFunc{"histFunc", "histFunc", x, *hist};
   RooExtendPdf extHistPdf("extHistPdf", "extHistPdf", histPdf, nEvents);
   RooArgList coefs{RooFit::RooConst(1.0)};
   RooRealSumPdf realSumPdf("realSumPdf", "realSumPdf", {histFunc}, coefs);
   RooAbsPdf &channelPdf = _ext ? static_cast<RooAbsPdf &>(extHistPdf) : static_cast<RooAbsPdf &>(realSumPdf);

   if (_binnedL)
      channelPdf.setAttribute("BinnedLikelihood");

   // Create a RooSimultaneous that wraps this pdf
   RooCategory sample("sample", "sample");
   sample.defineType("physics");
   RooSimultaneous simPdf{"simPdf", "simPdf", {{"physics", &channelPdf}}, sample};

   RooAbsData *absData = _binned ? static_cast<RooAbsData *>(hist.get()) : static_cast<RooAbsData *>(data.get());

   // The dataset used for simultaneous fits
   RooDataSet combData("combData", "combined data", *absData->get(), Index(sample), Import("physics", *absData));

   RooAbsData *fitData = _simPdf ? &combData : absData;
   RooAbsPdf *fitPdf = _simPdf ? static_cast<RooAbsPdf *>(&simPdf) : static_cast<RooAbsPdf *>(&channelPdf);

   RealPtr nll0{fitPdf->createNLL(*fitData, _evalBackend, Extended(_ext))};
   RealPtr nll1{fitPdf->createNLL(*fitData, Offset("bin"), _evalBackend, Extended(_ext))};

   if (_sumw2) {
      nll0->applyWeightSquared(true);
      nll1->applyWeightSquared(true);
   }

   double nllVal0 = nll0->getVal();
   double nllVal1 = nll1->getVal();

   // For all configurations, the bin offset should have the effect of bringing
   // the NLL to zero, modulo some numerical imprecisions:
   EXPECT_NEAR(nllVal1, 0.0, 1e-8) << "NLL with bin offsetting is " << nllVal1 << ", and " << nllVal0 << " without it.";
}

// Verify that the binned likelihood optimization works also when fitting a
// single-channel RooRealSumPdf or RooProdPdf.
TEST_P(TestStatisticTest, BinnedLikelihood)
{
   using namespace RooFit;

   int nEvents = 1000;
   int numBins = 5;

   RooWorkspace ws;

   ws.factory("x[0, 0, " + std::to_string(numBins) + "]");

   auto &x = *ws.var("x");
   x.setBins(numBins);

   {
      // Uniform RooDataHist, but with some empty bins to validate that empty
      // bins are treated correctly.
      RooDataHist dataHist{"data_hist", "data_hist", x};
      for (int iBin = 2; iBin < numBins; ++iBin) {
         dataHist.set(iBin, nEvents / numBins, -1);
      }

      RooHistFunc histFunc{"hist_func", "hist_func", x, dataHist};
      RooRealSumPdf pdf{"pdf", "pdf", histFunc, RooArgList{1.0}};

      // Enable the binned likelihood optimization to avoid integrals
      // (like in HistFactory).
      pdf.setAttribute("BinnedLikelihood");

      // Wrap the channel pdf in a RooProdPdf to mimic HistFactory
      RooProdPdf prodPdf{"prod_pdf", "prod_pdf", pdf};

      ws.import(prodPdf);
   }

   ws.factory("SIMUL::simPdf( cat[A=0], A=pdf)");
   auto &realSumPdf = *ws.pdf("pdf");
   auto &prodPdf = *ws.pdf("prod_pdf");
   auto &simPdf = *ws.pdf("simPdf");
   auto &cat = *ws.cat("cat");

   std::unique_ptr<RooDataHist> data{simPdf.generateBinned({x, cat}, nEvents)};

   std::unique_ptr<RooAbsReal> realSumNll{realSumPdf.createNLL(*data, _evalBackend)};
   std::unique_ptr<RooAbsReal> prodNll{prodPdf.createNLL(*data, _evalBackend)};
   std::unique_ptr<RooAbsReal> simNll{simPdf.createNLL(*data, _evalBackend)};

   double realSumNllVal = realSumNll->getVal();
   double prodNllVal = prodNll->getVal();
   double simNllVal = simNll->getVal();

   // If using the RooRealSumPdf or RooProdPdf directly is successfully hitting
   // the binned likelihood code path, the likelihood values will be identical
   // with the one of the RooSimultaneous.
   EXPECT_DOUBLE_EQ(realSumNllVal, simNllVal);
   EXPECT_DOUBLE_EQ(prodNllVal, simNllVal);
}

// Make sure that the offset is correctly hidden for the likelihoods, even if
// we evaluated the same likelihood without hiding before. This is tested
// because it was fragile before: a change in offset hiding was not considered
// in the dirty state propagation in the new CPU backend.
TEST_P(TestStatisticTest, HideOffset)
{
   RooWorkspace ws;
   ws.factory("Gaussian::model(x[0, 10], mean[6, 0, 10], sigma[2.0, 0.01, 10.0])");

   RooRealVar &x = *ws.var("x");
   RooAbsPdf &model = *ws.pdf("model");

   std::unique_ptr<RooDataSet> data{model.generate(x, 1000)};
   std::unique_ptr<RooAbsReal> nllNoOffset{model.createNLL(*data, _evalBackend)};
   std::unique_ptr<RooAbsReal> nll{model.createNLL(*data, RooFit::Offset("initial"), _evalBackend)};

   bool hideOffsetOrig = RooAbsReal::hideOffset();
   RooAbsReal::setHideOffset(false);
   EXPECT_FLOAT_EQ(nll->getVal(), 0.);
   RooAbsReal::setHideOffset(true);
   EXPECT_FLOAT_EQ(nll->getVal(), nllNoOffset->getVal());

   RooAbsReal::setHideOffset(hideOffsetOrig);
}

INSTANTIATE_TEST_SUITE_P(RooNLLVar, TestStatisticTest, testing::Values(ROOFIT_EVAL_BACKENDS),
                         [](testing::TestParamInfo<TestStatisticTest::ParamType> const &paramInfo) {
                            std::stringstream ss;
                            ss << "EvalBackend" << std::get<0>(paramInfo.param).name();
                            return ss.str();
                         });

INSTANTIATE_TEST_SUITE_P(RooNLLVar, OffsetBinTest,
                         testing::Combine(testing::Values(ROOFIT_EVAL_BACKENDS), // EvalBackend
                                          testing::Values(false, true),          // unbinned or binned
                                          testing::Values(false, true),          // extended fit
                                          testing::Values(false, true),          // use sumW2
                                          testing::Values(false, true),          // wrap in a RooSimultaneous
                                          testing::Values(false)                 // binned likelihood code path
                                          ),
                         [](testing::TestParamInfo<OffsetBinTest::ParamType> const &paramInfo) {
                            std::stringstream ss;
                            ss << "EvalBackend" << std::get<0>(paramInfo.param).name();
                            ss << (std::get<1>(paramInfo.param) ? "Binned" : "Unbinned");
                            ss << (std::get<2>(paramInfo.param) ? "Extended" : "");
                            ss << (std::get<3>(paramInfo.param) ? "SumW2" : "");
                            ss << (std::get<4>(paramInfo.param) ? "SimPdf" : "");
                            ss << (std::get<5>(paramInfo.param) ? "BinnedL" : "");
                            return ss.str();
                         });

INSTANTIATE_TEST_SUITE_P(RooNLLVarBinnedL, OffsetBinTest,
                         testing::Combine(testing::Values(ROOFIT_EVAL_BACKENDS), // EvalBackend
                                          testing::Values(true),                 // unbinned or binned
                                          testing::Values(false),                // extended fit
                                          testing::Values(false),                // use sumW2
                                          testing::Values(false, true),          // wrap in a RooSimultaneous
                                          testing::Values(true)                  // binned likelihood code path
                                          ),
                         [](testing::TestParamInfo<OffsetBinTest::ParamType> const &paramInfo) {
                            std::stringstream ss;
                            ss << "EvalBackend" << std::get<0>(paramInfo.param).name();
                            ss << (std::get<1>(paramInfo.param) ? "Binned" : "Unbinned");
                            ss << (std::get<2>(paramInfo.param) ? "Extended" : "");
                            ss << (std::get<3>(paramInfo.param) ? "SumW2" : "");
                            ss << (std::get<4>(paramInfo.param) ? "SimPdf" : "");
                            ss << (std::get<5>(paramInfo.param) ? "BinnedL" : "");
                            return ss.str();
                         });

// Test if the data can be correctly reset for both individual and simultaneous
// pdfs.
TEST(NLL, SetData)
{
   RooHelpers::LocalChangeMsgLevel changeMsgLvl(RooFit::WARNING);

   RooWorkspace workspace;
   workspace.factory("Gaussian::pdf_1(x[-10, 10], mu[0, -10, 10], sigma[1, 0.1, 10])");
   workspace.factory("SIMUL::pdf(index_cat[A=0], A=pdf_1)");

   RooAbsPdf &pdf1 = *workspace.pdf("pdf_1");
   RooAbsPdf &pdf = *workspace.pdf("pdf");
   RooRealVar &x = *workspace.var("x");
   RooAbsCategory &cat = *workspace.cat("index_cat");

   {
      // Build simple single-entry datasets so that it's easy to know the
      // reference result.
      const double xa = 0.0;
      const double xb = 1.0;

      RooDataSet data1a{"data_1_a", "data_1_a", {x, cat}};
      x.setVal(xa);
      data1a.add({x, cat});
      workspace.import(data1a);

      RooDataSet data1b{"data_1_b", "data_1_b", {x, cat}};
      x.setVal(xb);
      data1b.add({x, cat});
      workspace.import(data1b);
   }
   RooAbsData &data1a = *workspace.data("data_1_a");
   RooAbsData &data1b = *workspace.data("data_1_b");

   std::unique_ptr<RooAbsReal> nll1{pdf1.createNLL(data1a)};
   std::unique_ptr<RooAbsReal> nllSim{pdf.createNLL(data1a)};

   double val1A = nll1->getVal();
   double valSimA = nllSim->getVal();

   nll1->setData(data1b);
   nllSim->setData(data1b);

   double val1B = nll1->getVal();
   double valSimB = nllSim->getVal();

   // Make sure the combined and single channel pdfs give consistent results.
   EXPECT_DOUBLE_EQ(val1A, valSimA);
   EXPECT_DOUBLE_EQ(val1B, valSimB);

   // The dataset was built in such a way that the updated NLL is shifted by
   // 0.5, so this is what we analytically expect.
   EXPECT_DOUBLE_EQ(valSimB, valSimA + 0.5);
}

// In some frameworks like CMS combine, all constraints are contained in a
// single RooProdPdf, even if the constrained parameters are not used in this
// pdf. The RooFit logic to figure out constrained parameters should however
// now be confused by this, and not strip away these parameters from the list
// of constrained parameters.
TEST(CreateNLL, CombineStyleConstraints)
{
   RooHelpers::LocalChangeMsgLevel changeMsgLvl(RooFit::WARNING);

   RooWorkspace ws;
   ws.factory("Gaussian::g_main_1(x_1[0., -10, 10], mu_1[0., -10, 10], 1.0)");
   ws.factory("Gaussian::g_main_2(x_2[0., -10, 10], mu_2[0., -10, 10], 1.0)");
   ws.factory("Gaussian::g_constr_1(mu_1, 0.0, 1.0)");
   ws.factory("Gaussian::g_constr_2(mu_2, 0.0, 1.0)");

   // The RooProdPdf model_1 will contain all the constraints, also the one
   // that applies to g_main_1. This is the corner case that this test is
   // covering.
   ws.factory("PROD::model_1(g_main_1, g_constr_1, g_constr_2)");
   ws.factory("PROD::model_2(g_main_2)");
   ws.factory("SIMUL::model(cat[A=0, B=1], A=model_1, B=model_2)");

   auto &x1 = *ws.var("x_1");
   auto &x2 = *ws.var("x_2");
   auto &cat = *ws.cat("cat");
   auto &model = static_cast<RooSimultaneous &>(*ws.pdf("model"));

   RooArgSet vars{x1, x2, cat};
   RooDataSet data{"data", "data", vars};
   cat.setIndex(0);
   data.add(vars);
   cat.setIndex(1);
   data.add(vars);

   RooArgSet parameters;
   model.getParameters(data.get(), parameters);

   model.getAllConstraints(*data.get(), parameters, true);

   EXPECT_EQ(parameters.size(), 2);
   EXPECT_EQ(parameters.find("mu_1"), ws.var("mu_1"));
   EXPECT_EQ(parameters.find("mu_2"), ws.var("mu_2"));

   // Now, try it out in the context of createNLL
   std::unique_ptr<RooAbsReal> nll{model.createNLL(data)};
   const double proba = TMath::Gaus(0, 0, 1, true);
   const double nChannels = model.indexCat().size();
   //                                     main Gaussians                one constraint per channel
   const double refNllVal = -nChannels * (std::log(proba / nChannels) + std::log(proba));
   EXPECT_FLOAT_EQ(nll->getVal(), refNllVal);
}
