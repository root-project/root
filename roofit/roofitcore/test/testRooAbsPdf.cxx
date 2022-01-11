// Tests for RooAbsPdf
// Author: Stephan Hageboeck, CERN 04/2020

#include <RooAddPdf.h>                                                                               
#include <RooCategory.h>                                                                             
#include <RooConstVar.h>                                                                             
#include <RooDataSet.h>
#include <RooFitResult.h>  
#include <RooFormulaVar.h>
#include <RooGaussian.h>
#include <RooGenericPdf.h>
#include <RooHelpers.h>
#include <RooPoisson.h>
#include <RooProdPdf.h>                                                                              
#include <RooProduct.h>
#include <RooRealVar.h>
#include <RooSimultaneous.h>
#include <RooUniform.h>

#include <TClass.h>
#include <TRandom.h>
#include <ROOT/RMakeUnique.hxx>

#include <gtest/gtest.h>

#include <memory>

// ROOT-10668: Asympt. correct errors don't work when title and name differ
TEST(RooAbsPdf, AsymptoticallyCorrectErrors)
{
  RooRealVar x("x", "xxx", 0, 0, 10);
  RooRealVar a("a", "aaa", 2, 0, 10);
  // Cannot play with RooAbsPdf, since abstract.
  RooGenericPdf pdf("pdf", "std::pow(x,a)", RooArgSet(x, a));
  RooFormulaVar formula("w", "(x-5)*(x-5)*1.2", RooArgSet(x));

  std::unique_ptr<RooDataSet> data(pdf.generate(x, 5000));
  data->addColumn(formula);
  RooRealVar w("w", "weight", 1, 0, 20);
  RooDataSet weightedData("weightedData", "weightedData",
      RooArgSet(x, w), RooFit::Import(*data), RooFit::WeightVar(w));

  ASSERT_TRUE(weightedData.isWeighted());
  weightedData.get(0);
  ASSERT_NE(weightedData.weight(), 1);

  a = 1.2;
  auto result = pdf.fitTo(weightedData, RooFit::Save(), RooFit::AsymptoticError(true));
  const double aError = a.getError();
  a = 1.2;
  auto result2 = pdf.fitTo(weightedData, RooFit::Save());

  EXPECT_TRUE(result->isIdentical(*result2)) << "Fit results should be very similar.";
  EXPECT_GT(aError, a.getError()*2.) << "Asymptotically correct errors should be significantly larger.";
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

  auto makeFakeDataXY = []() {
    RooRealVar x("x", "x", 0, 10);
    RooRealVar y("y", "y", 1.0, 5);
    RooArgSet coord(x, y);

    auto d = std::make_unique<RooDataSet>("d", "d", RooArgSet(x, y));

    for (int i = 0; i < 10000; i++) {
      Double_t tmpy = gRandom->Gaus(3, 2);
      Double_t tmpx = gRandom->Poisson(tmpy);
      if (fabs(tmpy) > 1 && fabs(tmpy) < 5 && fabs(tmpx) < 10) {
        x = tmpx;
        y = tmpy;
        d->add(coord);
      }
    }

    return d;
  };

  auto data = makeFakeDataXY();

  RooRealVar x("x", "x", 0, 10);
  RooRealVar y("y", "y", 1.0, 5);

  RooRealVar factor("factor", "factor", 1.0, 0.0, 10.0);

  std::vector<RooProduct> means{{"mean", "mean", {factor, y}},
                                {"mean", "mean", {factor}}};
  std::vector<bool> expectFastEvaluationsWarnings{true, false};

  int iMean = 0;
  for(auto& mean : means) {

    RooPoisson model("model", "model", x, mean);

    std::vector<std::unique_ptr<RooFitResult>> fitResults;

    RooHelpers::HijackMessageStream hijack(RooFit::INFO, RooFit::FastEvaluations);

    for(bool batchMode : {false, true}) {
      factor.setVal(1.0);
      fitResults.emplace_back(
        model.fitTo(
              *data,
              ConditionalObservables(y),
              Save(),
              PrintLevel(-1),
              BatchMode(batchMode)
         ));
      fitResults.back()->Print();
    }

    EXPECT_TRUE(fitResults[1]->isIdentical(*fitResults[0]));
    EXPECT_TRUE(hijack.str().empty() != expectFastEvaluationsWarnings[iMean]);
    ++iMean;
  }

}

// This test will crash if the cached normalization sets are not reset
// correctly after servers are redirected. This is a reduced version of a code
// provided in the ROOT forum that originally unveiled this problem:
// https://root-forum.cern.ch/t/problems-with-2d-simultaneous-fit/48249/4
TEST(RooAbsPdf, ProblemsWith2DSimultaneousFit)
{
   using namespace RooFit;

   RooRealVar x("x", "y", 1.0, 2.);
   RooRealVar y("y", "y", 1.0, 2.);

   RooRealVar mu1("mu1", "mu1", 2., 0, 5);

   RooUniform uniform1("uniform1", "uniform1", {x, y});
   RooUniform uniform2("uniform2", "uniform2", {x, y});

   RooGaussian gauss1("gauss1", "gauss1", x, mu1, RooConst(0.1));
   RooGaussian gauss2("gauss2", "gauss2", x, mu1, RooConst(0.1));
   RooGaussian gauss3("gauss3", "gauss3", x, mu1, RooConst(0.1));

   RooAddPdf gauss12("gauss12", "gauss12", gauss1, gauss2, RooConst(0.1));

   RooAddPdf sig_x("sig_x", "sig_x", gauss3, gauss12, RooConst(0.1));

   RooUniform sig_y("sig_y", "sig_y", y);
   RooProdPdf sig("sig", "sig", sig_y, sig_x);

   RooRealVar yield{"yield", "yield", 100};

   // Complete model
   RooAddPdf model("model", "model", {sig, sig_y, uniform2, uniform1}, {yield, yield, yield, yield});

   // Define category to distinguish d0 and d0bar samples events
   RooCategory sample("sample", "sample", {{"cat0", 0}, {"cat1", 1}});

   // Construct a dummy dataset
   RooDataSet data("data", "data", RooArgSet(sample, x, y));

   // Construct a simultaneous pdf using category sample as index
   RooSimultaneous simPdf("simPdf", "simultaneous pdf", sample);
   simPdf.addPdf(model, "cat0");
   simPdf.addPdf(model, "cat1");

   simPdf.fitTo(data, PrintLevel(-1));
}
