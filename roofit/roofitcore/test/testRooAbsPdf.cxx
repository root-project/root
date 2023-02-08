// Tests for RooAbsPdf
// Authors: Stephan Hageboeck, CERN 04/2020
//          Jonas Rembser, CERN 04/2021

#include <RooAddition.h>
#include <RooAddPdf.h>
#include <RooCategory.h>
#include <RooConstVar.h>
#include <RooDataHist.h>
#include <RooDataSet.h>
#include <RooFitResult.h>
#include <RooFormulaVar.h>
#include <RooGenericPdf.h>
#include <RooHelpers.h>
#include <RooProdPdf.h>
#include <RooProduct.h>
#include <RooRealVar.h>
#include <RooSimultaneous.h>
#include <RooWorkspace.h>
#include <RooRandom.h>

#include <TClass.h>
#include <TRandom.h>

#include <gtest/gtest.h>

// Backward compatibility for gtest version < 1.10.0
#ifndef INSTANTIATE_TEST_SUITE_P
#define INSTANTIATE_TEST_SUITE_P INSTANTIATE_TEST_CASE_P
#endif

#include <memory>

class FitTest : public testing::TestWithParam<std::tuple<std::string>> {
   void SetUp() override
   {
      RooRandom::randomGenerator()->SetSeed(1337ul);
      _batchMode = std::get<0>(GetParam());
      _changeMsgLvl = std::make_unique<RooHelpers::LocalChangeMsgLevel>(RooFit::WARNING);
   }

   void TearDown() override { _changeMsgLvl.reset(); }

protected:
   std::string _batchMode;

private:
   std::unique_ptr<RooHelpers::LocalChangeMsgLevel> _changeMsgLvl;
};

// ROOT-10668: Asympt. correct errors don't work when title and name differ
TEST_P(FitTest, AsymptoticallyCorrectErrors)
{
   using namespace RooFit;

   RooRealVar x("x", "xxx", 0, 0, 10);
   RooRealVar a("a", "aaa", 2, 0, 10);
   // Cannot play with RooAbsPdf, since abstract.
   RooGenericPdf pdf("pdf", "std::pow(x,a)", RooArgSet(x, a));
   RooFormulaVar formula("w", "(x-5)*(x-5)*1.2", RooArgSet(x));

   std::unique_ptr<RooDataSet> data(pdf.generate(x, 5000));
   data->addColumn(formula);
   RooRealVar w("w", "weight", 1, 0, 20);
   RooDataSet weightedData("weightedData", "weightedData", {x, w}, Import(*data), WeightVar(w));

   ASSERT_TRUE(weightedData.isWeighted());
   weightedData.get(0);
   ASSERT_NE(weightedData.weight(), 1);

   a = 1.2;
   auto result = pdf.fitTo(weightedData, Save(), AsymptoticError(true), PrintLevel(-1), BatchMode(_batchMode));
   a = 1.2;
   auto result2 = pdf.fitTo(weightedData, Save(), SumW2Error(false), PrintLevel(-1), BatchMode(_batchMode));

   // Set relative tolerance for errors to large value to only check for values
   EXPECT_TRUE(result->isIdenticalNoCov(*result2, 1e-6, 10.0)) << "Fit results should be very similar.";
   // Set non-verbose because we expect the comparison to fail
   EXPECT_FALSE(result->isIdenticalNoCov(*result2, 1e-6, 1e-3, false))
      << "Asymptotically correct errors should be significantly larger.";
}

// Test a conditional fit with batch mode
//
// In a conditional fit, it happens that the value normalization integrals can
// be different for every event because a pdf is conditional on another
// observable. That's why the integral also has to be evaluated with the batch
// interface in general.
//
// This test checks if the results of a conditional fit are the same for batch
// and scalar mode.  It also verifies that for non-conditional fits, the batch
// mode recognizes that the integral only needs to be evaluated once.  This is
// checked by hijacking the FastEvaluations log. If a RooRealIntegral is
// evaluated in batch mode and data size is greater than one, the batch mode
// will inform that a batched evaluation function is missing.
TEST(RooAbsPdf, ConditionalFitBatchMode)
{
   using namespace RooFit;
   constexpr bool verbose = false;

   if (!verbose) {
      auto &msg = RooMsgService::instance();
      msg.getStream(1).removeTopic(RooFit::Minimization);
      msg.getStream(1).removeTopic(RooFit::Fitting);
   }

   auto makeFakeDataXY = []() {
      RooRealVar x("x", "x", 0, 10);
      RooRealVar y("y", "y", 1.0, 5);
      RooArgSet coord(x, y);

      auto d = std::make_unique<RooDataSet>("d", "d", RooArgSet(x, y));

      for (int i = 0; i < 10000; i++) {
         double tmpy = gRandom->Gaus(3, 2);
         double tmpx = gRandom->Poisson(tmpy);
         if (std::abs(tmpy) > 1 && std::abs(tmpy) < 5 && std::abs(tmpx) < 10) {
            x = tmpx;
            y = tmpy;
            d->add(coord);
         }
      }

      return d;
   };

   auto data = makeFakeDataXY();

   RooWorkspace ws;
   ws.factory("Product::mean1({factor[1.0, 0.0, 10.0], y[1.0, 5]})");
   ws.factory("Product::mean2({factor})");
   ws.factory("Poisson::model1(x[0, 10], mean1)");
   ws.factory("Poisson::model2(x, mean2)");

   RooRealVar &factor = *ws.var("factor");
   RooRealVar &y = *ws.var("y");

   std::vector<bool> expectFastEvaluationsWarnings{true, false};

   int iMean = 0;
   for (RooAbsPdf *model : {ws.pdf("model1"), ws.pdf("model2")}) {

      std::vector<std::unique_ptr<RooFitResult>> fitResults;

      RooHelpers::HijackMessageStream hijack(RooFit::INFO, RooFit::FastEvaluations);

      for (bool batchMode : {false, true}) {
         factor.setVal(1.0);
         factor.setError(0.0);
         fitResults.emplace_back(
            model->fitTo(*data, ConditionalObservables(y), Save(), PrintLevel(-1), BatchMode(batchMode)));
         if (verbose) {
            fitResults.back()->Print();
         }
      }

      EXPECT_TRUE(fitResults[1]->isIdentical(*fitResults[0]));
      EXPECT_EQ(hijack.str().find("does not implement the faster batch") != std::string::npos,
                expectFastEvaluationsWarnings[iMean])
         << "Stream contents: " << hijack.str();
      ++iMean;
   }
}

// ROOT-9530: RooFit side-band fit inconsistent with fit to full range
TEST_P(FitTest, MultiRangeFit)
{
   using namespace RooFit;

   RooWorkspace ws;

   ws.factory("Gaussian::model_simple(x[-10., 10.], mean[-1, -10, 10], width[3, 0.1, 10])");
   // model for extended fit
   ws.factory("AddPdf::model_extended({model_simple}, {nsig[100, 0, 2000]})");

   auto &x = *ws.var("x");
   auto &mean = *ws.var("mean");
   auto &width = *ws.var("width");
   auto &nsig = *ws.var("nsig");

   RooAbsPdf &modelSimple = *ws.pdf("model_simple");
   RooAbsPdf &modelExtended = *ws.pdf("model_extended");

   const double cut = -5;
   x.setRange("full", -10, 10);
   x.setRange("low", -10, cut);
   x.setRange("high", cut, 10);

   const std::size_t nEvents = nsig.getVal();

   auto resetValues = [&]() {
      mean.setVal(-1);
      width.setVal(3);
      nsig.setVal(nEvents);
      mean.setError(0.0);
      width.setError(0.0);
      nsig.setError(0.0);
   };

   // loop over non-extended and extended fit
   for (auto *model : {static_cast<RooAbsPdf *>(&modelSimple), static_cast<RooAbsPdf *>(&modelExtended)}) {

      std::unique_ptr<RooAbsData> dataSet{model->generate(x, nEvents)};
      std::unique_ptr<RooAbsData> dataHist{static_cast<RooDataSet &>(*dataSet).binnedClone()};

      // loop over binned fit and unbinned fit
      for (auto *data : {dataSet.get(), dataHist.get()}) {
         // full range
         resetValues();
         std::unique_ptr<RooFitResult> fitResultFull{
            model->fitTo(*data, Range("full"), Save(), PrintLevel(-1), BatchMode(_batchMode))};

         // part (side band fit, but the union of the side bands is the full range)
         resetValues();
         std::unique_ptr<RooFitResult> fitResultPart{
            model->fitTo(*data, Range("low,high"), Save(), PrintLevel(-1), BatchMode(_batchMode))};

         EXPECT_TRUE(fitResultPart->isIdentical(*fitResultFull))
            << "Results of fitting " << model->GetName() << " to a " << data->ClassName() << " should be very similar.";
      }
   }
}

// ROOT-9530: RooFit side-band fit inconsistent with fit to full range (2D case)
TEST_P(FitTest, MultiRangeFit2D)
{
   using namespace RooFit;

   // model taken from the rf312_multirangefit.C tutorial
   RooWorkspace ws;

   // Construct the signal pdf gauss(x)*gauss(y)
   ws.factory("Gaussian::gx(x[-10, 10], mx[1, -10, 10], 1.0)");
   ws.factory("Gaussian::gy(y[-10, 10], my[1, -10, 10], 1.0)");
   ws.factory("ProdPdf::sig(gx, gy)");

   // Construct the background pdf (flat in x,y)
   ws.factory("Polynomial::px(x)");
   ws.factory("Polynomial::py(y)");
   ws.factory("ProdPdf::bkg(px, py)");

   // Construct the composite model sig+bkg
   ws.factory("AddPdf::model({sig, bkg}, f[0.5, 0., 1.])");

   RooRealVar &x = *ws.var("x");
   RooRealVar &y = *ws.var("y");
   RooRealVar &mx = *ws.var("mx");
   RooRealVar &my = *ws.var("my");
   RooRealVar &f = *ws.var("f");

   RooAbsPdf &model = *ws.pdf("model");

   x.setRange("SB1", -10, +10);
   y.setRange("SB1", -10, 0);

   x.setRange("SB2", -10, 0);
   y.setRange("SB2", 0, +10);

   x.setRange("SIG", 0, +10);
   y.setRange("SIG", 0, +10);

   x.setRange("FULL", -10, +10);
   y.setRange("FULL", -10, +10);

   auto resetValues = [&]() {
      mx.setVal(1.0);
      my.setVal(1.0);
      f.setVal(0.5);
      mx.setError(0.0);
      my.setError(0.0);
      f.setError(0.0);
   };

   std::size_t nEvents = 100;

   // try out with both binned and unbinned data
   std::unique_ptr<RooDataSet> dataSet{model.generate({x, y}, nEvents)};
   std::unique_ptr<RooDataHist> dataHist{dataSet->binnedClone()};

   // loop over binned fit and unbinned fit
   for (auto *data : {static_cast<RooAbsData *>(dataSet.get()), static_cast<RooAbsData *>(dataHist.get())}) {
      // full range
      resetValues();
      std::unique_ptr<RooFitResult> fitResultFull{
         model.fitTo(*data, Range("FULL"), Save(), PrintLevel(-1), BatchMode(_batchMode))};

      // part (side band fit, but the union of the side bands is the full range)
      resetValues();
      std::unique_ptr<RooFitResult> fitResultPart{
         model.fitTo(*data, Range("SB1,SB2,SIG"), Save(), PrintLevel(-1), BatchMode(_batchMode))};

      EXPECT_TRUE(fitResultPart->isIdentical(*fitResultFull))
         << "Results of fitting " << model.GetName() << " to a " << data->ClassName() << " should be very similar.";
   }
}

// This test will crash if the cached normalization sets are not reset
// correctly after servers are redirected. This is a reduced version of a code
// provided in the ROOT forum that originally unveiled this problem:
// https://root-forum.cern.ch/t/problems-with-2d-simultaneous-fit/48249/4
TEST_P(FitTest, ProblemsWith2DSimultaneousFit)
{
   using namespace RooFit;

   RooWorkspace ws;
   ws.factory("Uniform::uniform1({x[1.0, 2.], y[1.0, 2.]})");
   ws.factory("Uniform::uniform2({x, y})");
   ws.factory("Gaussian::gauss1(x, mu1[2., 0., 5.], 0.1)");
   ws.factory("Gaussian::gauss2(x, mu1[2., 0., 5.], 0.1)");
   ws.factory("Gaussian::gauss3(x, mu1[2., 0., 5.], 0.1)");
   ws.factory("AddPdf::gauss12(gauss1, gauss2, 0.1)");
   ws.factory("AddPdf::sig_x(gauss3, gauss12, 0.1)");
   ws.factory("Uniform::sig_y(y)");
   ws.factory("ProdPdf::sig(sig_y, sig_x)");

   // Complete model
   ws.factory("AddPdf::model({sig, sig_y, uniform2, uniform1}, {yield[100], yield, yield, yield})");

   RooAbsPdf &model = *ws.pdf("model");

   // Define category to distinguish d0 and d0bar samples events
   RooCategory sample("sample", "sample", {{"cat0", 0}, {"cat1", 1}});

   // Construct a dummy dataset
   RooDataSet data("data", "data", RooArgSet(sample, *ws.var("x"), *ws.var("y")));

   // Construct a simultaneous pdf using category sample as index
   RooSimultaneous simPdf("simPdf", "simultaneous pdf", sample);
   simPdf.addPdf(model, "cat0");
   simPdf.addPdf(model, "cat1");

   simPdf.fitTo(data, PrintLevel(-1), BatchMode(_batchMode));
}

// Verifies that a server pdf gets correctly reevaluated when the normalization
// set is changed.
TEST(RooAbsPdf, NormSetChange)
{
   using namespace RooFit;

   RooWorkspace ws;
   ws.factory("Gaussian::gauss(x[0, -10, 10], 0., 2.)");

   RooRealVar &x = *ws.var("x");
   RooAbsPdf &gauss = *ws.pdf("gauss");

   RooAddition add("add", "add", {gauss});

   double v1 = add.getVal();
   double v2 = add.getVal(x);

   // The change of normalization set should trigger a recomputation of the
   // value, so val2 should be different from val1. }
   EXPECT_NE(v1, v2);
}

INSTANTIATE_TEST_SUITE_P(RooAbsPdf, FitTest, testing::Values("Off", "Cpu"),
                         [](testing::TestParamInfo<FitTest::ParamType> const &paramInfo) {
                            std::stringstream ss;
                            ss << "BatchMode" << std::get<0>(paramInfo.param);
                            return ss.str();
                         });
