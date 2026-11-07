// Author: Jonas Rembser, CERN  01/2025

#include "RooDataSet.h"
#include "RooFitResult.h"
#include "RooMultiVarGaussian.h"
#include "RooRealVar.h"
#include "RooWorkspace.h"
#include "RooStats/AsymptoticCalculator.h"
#include "RooStats/HypoTestResult.h"
#include "RooStats/ModelConfig.h"

#include "Math/ProbFuncMathCore.h"
#include "ROOT/TestSupport.hxx"

#include "gtest/gtest.h"

#include <memory>

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

// Check the signed (uncapped) one-sided profile likelihood test statistic
// enabled with AsymptoticCalculator::SetSigned() (ROOT-8257).
TEST(AsymptoticCalculator, SignedTestStatistic)
{
   using namespace RooStats;

   RooWorkspace ws;
   ws.factory("Gaussian::model(x[1.5, -5, 5], mu[1.0, -5, 5], 1.0)");

   RooRealVar &x = *ws.var("x");
   RooRealVar &mu = *ws.var("mu");

   // A single observation at x = 1.5, so muHat = 1.5 is beyond the tested value mu = 1.
   RooDataSet data{"data", "data", {x}};
   data.add({x});

   ModelConfig nullModel{"null_model", &ws};
   nullModel.SetPdf(*ws.pdf("model"));
   nullModel.SetObservables({x});
   nullModel.SetParametersOfInterest({mu});
   mu.setVal(1.0);
   nullModel.SetSnapshot({mu});

   ModelConfig altModel{nullModel};
   altModel.SetName("alt_model");
   mu.setVal(0.0);
   altModel.SetSnapshot({mu});

   AsymptoticCalculator calc{data, altModel, nullModel};
   calc.SetOneSided(true);

   // For a single Gaussian observation, muHat = x = 1.5 with sigma(muHat) = 1,
   // so qmu = (x - mu)^2 = 0.25, and on the alt-hypothesis Asimov data (x = 0)
   // qmu_A = mu^2 = 1.
   const double sqrtqmu = 0.5;
   const double sqrtqmuA = 1.0;

   // With the capped one-sided statistic, qmu is set to zero because muHat is
   // beyond the tested value.
   std::unique_ptr<HypoTestResult> resCapped{calc.GetHypoTest()};
   EXPECT_NEAR(resCapped->NullPValue(), ROOT::Math::normal_cdf_c(0.0), 1e-3);
   EXPECT_NEAR(resCapped->AlternatePValue(), ROOT::Math::normal_cdf(sqrtqmuA), 1e-3);

   // With the signed statistic, sqrt(qmu) enters the asymptotic formulae with
   // a negative sign instead.
   calc.SetSigned();
   std::unique_ptr<HypoTestResult> resSigned{calc.GetHypoTest()};
   EXPECT_NEAR(resSigned->NullPValue(), ROOT::Math::normal_cdf_c(-sqrtqmu), 1e-3);
   EXPECT_NEAR(resSigned->AlternatePValue(), ROOT::Math::normal_cdf(sqrtqmuA + sqrtqmu), 1e-3);

   // On the "right side" (muHat below the tested value) the signed and capped
   // statistics must agree: test mu = 2, where sqrt(qmu) = 0.5.
   mu.setVal(2.0);
   nullModel.SetSnapshot({mu});
   calc.SetNullModel(nullModel);
   std::unique_ptr<HypoTestResult> resRightSide{calc.GetHypoTest()};
   EXPECT_NEAR(resRightSide->NullPValue(), ROOT::Math::normal_cdf_c(sqrtqmu), 1e-3);

   calc.SetSigned(false);
   std::unique_ptr<HypoTestResult> resRightSideCapped{calc.GetHypoTest()};
   EXPECT_NEAR(resRightSideCapped->NullPValue(), resRightSide->NullPValue(), 1e-6);
}

// Check that counting Asimov datasets can be generated no matter which
// parameters are floating and even if the mean or width of a Gaussian are
// derived quantities (covers JIRA ROOT-10096).
TEST(AsymptoticCalculator, CountingAsimovDataSetFloatingParams)
{
   RooWorkspace ws;
   ws.factory("obs[10.0, 0.0, 1000.0]");
   ws.factory("Poisson::poisson(obs, mean[20.0, 0.0, 1000.0])");
   ws.factory("Gaussian::gauss1(obs, mean, sigma[3.0, 1.0, 10.0])");
   ws.factory("expr::sqrt_mean('sqrt(@0)', mean)");
   ws.factory("Gaussian::gauss2(obs, mean, sqrt_mean)");
   ws.factory("expr::mean2('2 * @0', mean)");
   ws.factory("expr::sqrt_mean2('sqrt(@0)', mean2)");
   ws.factory("Gaussian::gauss3(obs, mean2, sqrt_mean2)");

   RooArgSet observables{*ws.var("obs")};

   auto checkAsimov = [&](const char *pdfName, double expectedObsVal) {
      std::unique_ptr<RooAbsData> data{
         RooStats::AsymptoticCalculator::GenerateAsimovData(*ws.pdf(pdfName), observables)};
      ASSERT_NE(data, nullptr) << pdfName;
      ASSERT_EQ(data->numEntries(), 1) << pdfName;
      EXPECT_DOUBLE_EQ(data->get(0)->getRealValue("obs"), expectedObsVal) << pdfName;
   };

   checkAsimov("poisson", 20.0);
   // Both mean and sigma floating: used to fail with "Has two non-const arguments".
   checkAsimov("gauss1", 20.0);
   // Width derived from the mean: also used to fail, with no workaround for gauss3.
   checkAsimov("gauss2", 20.0);
   checkAsimov("gauss3", 40.0);

   // With a constant mean and a floating sigma, the old server-based heuristic
   // silently set the observable to the value of the sigma parameter.
   ws.var("mean")->setConstant(true);
   checkAsimov("gauss1", 20.0);
}

// Check that the fit results of the fits performed in Initialize() and
// GetHypoTest() are stored and can be accessed by the user (JIRA ROOT-10066).
TEST(AsymptoticCalculator, StoredFitResults)
{
   using namespace RooStats;

   // On/off Poisson counting model: signal region with s + b expected events,
   // control region constraining the background via tau * b.
   RooWorkspace ws;
   ws.factory("Poisson::px(x[150, 0, 500], sum::splusb(s[0, 0, 100], b[100, 0, 300]))");
   ws.factory("Poisson::py(y[100, 0, 500], prod::taub(tau[1.0], b))");
   ws.factory("PROD::model(px, py)");

   RooRealVar &s = *ws.var("s");
   RooArgSet obs{*ws.var("x"), *ws.var("y")};

   RooDataSet data{"data", "data", obs};
   data.add(obs);

   ModelConfig sbModel{"sbModel", &ws};
   sbModel.SetPdf(*ws.pdf("model"));
   sbModel.SetObservables(obs);
   sbModel.SetParametersOfInterest(RooArgSet{s});
   sbModel.SetNuisanceParameters(RooArgSet{*ws.var("b")});
   s.setVal(50.0);
   sbModel.SetSnapshot(RooArgSet{s});

   std::unique_ptr<ModelConfig> bModel{static_cast<ModelConfig *>(sbModel.Clone("bModel"))};
   s.setVal(0.0);
   bModel->SetSnapshot(RooArgSet{s});

   // Some of the fits start at parameter values that already correspond to the
   // minimum, in which case Minuit2 emits a harmless line-search warning.
   ROOT::TestSupport::CheckDiagsRAII checkDiag;
   checkDiag.optionalDiag(kWarning, "Minuit2", "VariableMetricBuilder No improvement in line search", false);

   AsymptoticCalculator calc{data, *bModel, sbModel};
   calc.SetOneSided(true);

   // Before running the hypothesis test, no fit results are available.
   EXPECT_EQ(calc.GetFitResultCondObs(), nullptr);
   EXPECT_EQ(calc.GetFitResultCondAsimov(), nullptr);

   std::unique_ptr<HypoTestResult> result{calc.GetHypoTest()};
   ASSERT_NE(result, nullptr);

   const RooFitResult *uncondObs = calc.GetFitResultUncondObs();
   const RooFitResult *condObs = calc.GetFitResultCondObs();
   const RooFitResult *uncondAsimov = calc.GetFitResultUncondAsimov();
   const RooFitResult *condAsimov = calc.GetFitResultCondAsimov();

   ASSERT_NE(uncondObs, nullptr);
   ASSERT_NE(condObs, nullptr);
   ASSERT_NE(uncondAsimov, nullptr);
   ASSERT_NE(condAsimov, nullptr);

   EXPECT_EQ(uncondObs->status(), 0);
   EXPECT_EQ(condObs->status(), 0);
   EXPECT_EQ(uncondAsimov->status(), 0);
   EXPECT_EQ(condAsimov->status(), 0);

   // The best-fit POI value from the unconditional fit result must be
   // consistent with the one stored by the calculator.
   auto *sFitUncond = static_cast<RooRealVar *>(uncondObs->floatParsFinal().find("s"));
   ASSERT_NE(sFitUncond, nullptr);
   EXPECT_DOUBLE_EQ(sFitUncond->getVal(), calc.GetMuHat()->getVal());

   // In the conditional fits the POI is fixed to the tested value from the
   // null-model snapshot, so it appears in the constant parameter list.
   auto *sFitCond = static_cast<RooRealVar *>(condObs->constPars().find("s"));
   ASSERT_NE(sFitCond, nullptr);
   EXPECT_DOUBLE_EQ(sFitCond->getVal(), 50.0);

   // The profile likelihood ratio test statistic reconstructed from the stored
   // fit results must be non-negative, up to the same numerical tolerance that
   // the calculator itself uses for qmu.
   EXPECT_GE(condObs->minNll() - uncondObs->minNll(), -1.e-3);
   EXPECT_GE(condAsimov->minNll() - uncondAsimov->minNll(), -1.e-3);
}
