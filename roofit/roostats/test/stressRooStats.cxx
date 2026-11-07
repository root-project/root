// Authors: Ioan Gabriel Bucur, Lorenzo Moneta, Wouter Verkerke
//
// Googletest version of the old RooStats S.T.R.E.S.S. suite. Each of the 48
// tests of the original suite is translated to a (parameterized) gtest test
// case with the same models, calculator configurations, random seeds and
// tolerances.
//
// The original suite compared results with references stored in
// stressRooStats_ref.root. Depending on the test, these references were either
// computed analytically, taken from a publication, or produced by the tested
// calculator itself at reference-writing time (pure regression tests). In this
// version, analytic references are computed inline, and published or
// regression reference values are hardcoded (the latter extracted from the
// last stressRooStats_ref.root).
//
// The tests are parameterized over the RooFit evaluation backends, matching
// the backend coverage of the removed stressRooStats invocations. Like the
// old "-minim" command line option, the minimizer can be chosen via the
// STRESSROOSTATS_MINIMIZER environment variable (default is Minuit2).

#include "../../roofitcore/test/gtest_wrapper.h"

// Global functions that build the more complex RooStats models
#include "stressRooStats_models.h"

// RooStats headers
#include <RooStats/AsymptoticCalculator.h>
#include <RooStats/BayesianCalculator.h>
#include <RooStats/FrequentistCalculator.h>
#include <RooStats/HybridCalculator.h>
#include <RooStats/HypoTestCalculatorGeneric.h>
#include <RooStats/HypoTestInverter.h>
#include <RooStats/HypoTestInverterResult.h>
#include <RooStats/HypoTestResult.h>
#include <RooStats/LikelihoodInterval.h>
#include <RooStats/MCMCCalculator.h>
#include <RooStats/MCMCInterval.h>
#include <RooStats/NumberCountingUtils.h>
#include <RooStats/ProfileLikelihoodCalculator.h>
#include <RooStats/ProfileLikelihoodTestStat.h>
#include <RooStats/RatioOfProfiledLikelihoodsTestStat.h>
#include <RooStats/SequentialProposal.h>
#include <RooStats/SimpleInterval.h>
#include <RooStats/SimpleLikelihoodRatioTestStat.h>
#include <RooStats/TestStatistic.h>
#include <RooStats/ToyMCSampler.h>

// RooFit headers
#include <RooAbsReal.h>
#include <RooCFunction1Binding.h>
#include <RooDataSet.h>
#include <RooGlobalFunc.h>
#include <RooMsgService.h>
#include <RooNumIntConfig.h>
#include <RooRandom.h>
#include <RooRealVar.h>
#include <RooWorkspace.h>

// ROOT headers
#include <Math/MinimizerOptions.h>
#include <TF1.h>
#include <TRandom.h>
#include <TString.h>
#include <TSystem.h>

#include <cmath>
#include <memory>
#include <string>
#include <tuple>
#include <utility>

using namespace ROOT::Math;
using namespace RooFit;
using namespace RooStats;

namespace {

enum ECalculatorType { kAsymptotic = 0, kFrequentist = 1, kHybrid = 2 };
enum ETestStatType {
   kSimpleLR = 0,
   kRatioLR = 1,
   kProfileLR = 2,
   kProfileLROneSided = 3,
   kProfileLROneSidedDiscovery = 4
};

// Confidence levels corresponding to one, two and three Gaussian sigmas
const double kCL1Sigma = 2 * normal_cdf(1) - 1;
const double kCL2Sigma = 2 * normal_cdf(2) - 1;
const double kCL3Sigma = 2 * normal_cdf(3) - 1;

// Value test tolerance of the original RooUnitTest
const double kVTol = 1e-3;

std::unique_ptr<HypoTestCalculatorGeneric> buildHypoTestCalculator(const ECalculatorType calculatorType,
                                                                   RooAbsData &data, const ModelConfig &nullModel,
                                                                   const ModelConfig &altModel, const UInt_t toysNull,
                                                                   const UInt_t toysAlt)
{
   if (calculatorType == kAsymptotic) {
      return std::make_unique<AsymptoticCalculator>(data, altModel, nullModel);
   } else if (calculatorType == kFrequentist) {
      auto fc = std::make_unique<FrequentistCalculator>(data, altModel, nullModel);
      // set toys for speedup
      fc->SetToys(toysNull, toysAlt);
      return fc;
   }
   // kHybrid
   auto hc = std::make_unique<HybridCalculator>(data, altModel, nullModel);
   // set toys for speedup
   hc->SetToys(toysNull, toysAlt);
   return hc;
}

std::unique_ptr<TestStatistic>
buildTestStatistic(const ETestStatType testStatType, const ModelConfig &nullModel, const ModelConfig &altModel)
{
   if (testStatType == kSimpleLR) {
      auto slrts = std::make_unique<SimpleLikelihoodRatioTestStat>(*nullModel.GetPdf(), *altModel.GetPdf());
      if (nullModel.GetSnapshot()) {
         RooArgSet nullParams(*nullModel.GetSnapshot());
         if (nullModel.GetNuisanceParameters())
            nullParams.add(*nullModel.GetNuisanceParameters());
         slrts->SetNullParameters(nullParams);
      }
      if (altModel.GetSnapshot()) {
         RooArgSet altParams(*altModel.GetSnapshot());
         if (altModel.GetNuisanceParameters())
            altParams.add(*altModel.GetNuisanceParameters());
         slrts->SetAltParameters(altParams);
      }
      slrts->SetAlwaysReuseNLL(true);
      return slrts;
   } else if (testStatType == kRatioLR) {
      auto roplts = std::make_unique<RatioOfProfiledLikelihoodsTestStat>(*nullModel.GetPdf(), *altModel.GetPdf(),
                                                                         altModel.GetSnapshot());
      roplts->SetSubtractMLE(false);
      roplts->SetAlwaysReuseNLL(true);
      return roplts;
   }
   // kProfileLR, kProfileLROneSided and kProfileLROneSidedDiscovery
   auto plts = std::make_unique<ProfileLikelihoodTestStat>(*nullModel.GetPdf());
   if (testStatType == kProfileLROneSided) {
      plts->SetOneSided(true);
   } else if (testStatType == kProfileLROneSidedDiscovery) {
      plts->SetOneSidedDiscovery(true);
   }
   plts->SetAlwaysReuseNLL(true);
   return plts;
}

/// Create the Poisson product model in the workspace and add the observed
/// values to its data set. Returns the "S+B" model configuration (or nullptr
/// if the workspace content is unexpected, to be caught by the caller).
ModelConfig *setupPoissonProductModel(RooWorkspace &ws, int obsValueX, int obsValueY)
{
   buildPoissonProductModel(&ws);
   auto model = dynamic_cast<ModelConfig *>(ws.obj("S+B"));
   if (model) {
      ws.var("x")->setVal(obsValueX);
      ws.var("y")->setVal(obsValueY);
      ws.data("data")->add(*model->GetObservables());
   }
   return model;
}

/// Global setup that mirrors the environment of the original stressRooStats
/// executable: minimizer selection and RooFit message streams silenced below
/// the ERROR level.
class StressRooStatsEnvironment : public ::testing::Environment {
public:
   void SetUp() override
   {
      const char *minimizer = gSystem->Getenv("STRESSROOSTATS_MINIMIZER");
      ROOT::Math::MinimizerOptions::SetDefaultMinimizer(minimizer ? minimizer : "Minuit2");

      // Disable RooFit messages below the ERROR level, but keep a dedicated
      // error stream active so that problems remain visible in the test log.
      auto &msgSvc = RooMsgService::instance();
      msgSvc.setSilentMode(true);
      for (int i = 0; i < msgSvc.numStreams(); ++i) {
         if (msgSvc.getStream(i).minLevel < RooFit::ERROR) {
            msgSvc.setStreamStatus(i, false);
         }
      }
      msgSvc.addStream(RooFit::ERROR);

      AsymptoticCalculator::SetPrintLevel(0);

      // NOTE: RooIntegrator1D is too slow and gives poor results
#ifdef ROOFITMORE
      RooAbsReal::defaultIntegratorConfig()->method1D().setLabel("RooAdaptiveGaussKronrodIntegrator1D");
#endif
   }
};

[[maybe_unused]] const auto gStressRooStatsEnv = ::testing::AddGlobalTestEnvironment(new StressRooStatsEnvironment);

// Reset random generator seeds to make results independent of test ordering,
// like RooUnitTest::runTest() did in the original suite.
void setUpStressRooStatsTest()
{
   gRandom->SetSeed(12345);
   RooRandom::randomGenerator()->SetSeed(12345);
   RooMsgService::instance().clearErrorCount();
}

// The original suite failed a test if RooFit ERROR messages were logged.
void tearDownStressRooStatsTest()
{
   EXPECT_EQ(RooMsgService::instance().errorCount(), 0) << "RooFit ERROR messages were logged during the test";
}

/// Fixture for tests that only take the evaluation backend as parameter. The
/// RooStats calculators create their likelihoods internally, so the backend
/// parameter has to be applied via the global default backend (restored again
/// in TearDown to not leak it into other tests).
class StressRooStatsBackendTest : public ::testing::TestWithParam<std::tuple<RooFit::EvalBackend>> {
protected:
   void SetUp() override
   {
      _prevBackend = RooFit::EvalBackend::defaultValue();
      RooFit::EvalBackend::defaultValue() = std::get<0>(GetParam()).value();
      setUpStressRooStatsTest();
   }
   void TearDown() override
   {
      tearDownStressRooStatsTest();
      RooFit::EvalBackend::defaultValue() = _prevBackend;
   }

private:
   RooFit::EvalBackend::Value _prevBackend = RooFit::EvalBackend::Value::Cpu;
};

/// Fixture for tests that are parameterized over the evaluation backend and a
/// test-specific parameter set (see StressRooStatsBackendTest).
template <typename ParamType>
class StressRooStatsParamTest : public ::testing::TestWithParam<std::tuple<RooFit::EvalBackend, ParamType>> {
protected:
   void SetUp() override
   {
      _prevBackend = RooFit::EvalBackend::defaultValue();
      RooFit::EvalBackend::defaultValue() = std::get<0>(this->GetParam()).value();
      setUpStressRooStatsTest();
   }
   void TearDown() override
   {
      tearDownStressRooStatsTest();
      RooFit::EvalBackend::defaultValue() = _prevBackend;
   }

   const ParamType &param() const { return std::get<1>(this->GetParam()); }

private:
   RooFit::EvalBackend::Value _prevBackend = RooFit::EvalBackend::Value::Cpu;
};

std::string backendName(const ::testing::TestParamInfo<std::tuple<RooFit::EvalBackend>> &info)
{
   return "EvalBackend" + std::get<0>(info.param).name();
}

template <typename ParamType>
std::string backendParamName(const ::testing::TestParamInfo<std::tuple<RooFit::EvalBackend, ParamType>> &info)
{
   return "EvalBackend" + std::get<0>(info.param).name() + "_" + std::get<1>(info.param).name;
}

} // namespace

///////////////////////////////////////////////////////////////////////////////
//
// PART ONE: PROFILE LIKELIHOOD CALCULATOR UNIT TESTS
//
///////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////
//
// PROFILE LIKELIHOOD CALCULATOR - LIKELIHOOD INTERVAL - GAUSSIAN DISTRIBUTION
//
// Test the likelihood interval computed by the profile likelihood calculator
// on a Gaussian distribution. Reference interval limits are computed via
// analytic methods: solve equation 2*(ln(LL(xMax))-ln(LL(x)) = q, where q =
// normal_quantile_c(testSize/2, 1). In the case of a Gaussian distribution,
// the interval limits are equal to:
// mean +- normal_quantile_c(testSize/2, sigma/sqrt(N)).
//
// ModelConfig (implicit) :
//    Observable -> x
//    Parameter of Interest -> mean
//    Nuisance parameter (constant!) -> sigma
//
// Original tests 1-5 (TestProfileLikelihoodCalculator1), with the confidence
// level probing the boundaries of the (0,1) range.
//
///////////////////////////////////////////////////////////////////////////////

struct PlcGaussianParams {
   std::string name;
   double confidenceLevel;
};

class PlcGaussianInterval : public StressRooStatsParamTest<PlcGaussianParams> {};

TEST_P(PlcGaussianInterval, CompareWithAnalyticInterval)
{
   const double confidenceLevel = param().confidenceLevel;
   const int N = 10; // number of observations

   // Create Gaussian model and generate a data set
   RooWorkspace ws{"w"};
   ws.factory("Gaussian::gauss(x[-5,5], mean[0,-5,5], sigma[1])");
   std::unique_ptr<RooDataSet> data{ws.pdf("gauss")->generate(*ws.var("x"), N)};

   // Reference likelihood interval limits computed via analytic methods
   const double estMean = data->mean(*ws.var("x"));
   const double intervalHalfWidth =
      normal_quantile_c((1.0 - confidenceLevel) / 2.0, ws.var("sigma")->getValV() / std::sqrt((double)N));

   // Calculate likelihood interval using the ProfileLikelihoodCalculator
   ProfileLikelihoodCalculator plc{*data, *ws.pdf("gauss"), *ws.var("mean")};
   plc.SetConfidenceLevel(confidenceLevel);
   std::unique_ptr<LikelihoodInterval> interval{plc.GetInterval()};

   EXPECT_NEAR(interval->LowerLimit(*ws.var("mean")), estMean - intervalHalfWidth, kVTol);
   EXPECT_NEAR(interval->UpperLimit(*ws.var("mean")), estMean + intervalHalfWidth, kVTol);
}

INSTANTIATE_TEST_SUITE_P(RooStats, PlcGaussianInterval,
                         ::testing::Combine(::testing::Values(ROOFIT_EVAL_BACKENDS),
                                            ::testing::Values(
                                               PlcGaussianParams{"CLNearOne", 0.99999}, // boundary case CL -> 1
                                               PlcGaussianParams{"CL3Sigma", kCL3Sigma},
                                               PlcGaussianParams{"CL2Sigma", kCL2Sigma},
                                               PlcGaussianParams{"CL1Sigma", kCL1Sigma},
                                               PlcGaussianParams{"CLNearZero", 0.00001})), // boundary case CL -> 0
                         backendParamName<PlcGaussianParams>);

///////////////////////////////////////////////////////////////////////////////
//
// PROFILE LIKELIHOOD CALCULATOR - LIKELIHOOD INTERVAL - POISSON DISTRIBUTION
//
// Test the 68% likelihood interval computed by the profile likelihood
// calculator on a Poisson distribution, from only one observed value.
// Reference values are computed via analytic methods: solve equation
// 2*[ln(LL(xMax)) - ln(LL(x))] = 1.
//
// ModelConfig (implicit) :
//    Observable -> x
//    Parameter of Interest -> mean
//
// Original tests 6-10 (TestProfileLikelihoodCalculator2), with the observed
// value probing the boundaries of the [0,1000] range.
//
///////////////////////////////////////////////////////////////////////////////

struct PlcPoissonParams {
   std::string name;
   int obsValue;
};

class PlcPoissonInterval : public StressRooStatsParamTest<PlcPoissonParams> {};

TEST_P(PlcPoissonInterval, CompareWithAnalyticInterval)
{
   const int obsValue = param().obsValue;

   // Set a 68% confidence level for the interval
   const double confidenceLevel = kCL1Sigma;

   // Create Poisson model and dataset
   RooWorkspace ws{"w"};
   ws.factory(TString::Format("Poisson::poiss(x[%d,0,1000], mean[0,1000])", obsValue).Data());
   RooDataSet data{"data", "data", *ws.var("x")};
   data.add(*ws.var("x"));

   // Calculate likelihood interval using the ProfileLikelihoodCalculator
   ProfileLikelihoodCalculator plc{data, *ws.pdf("poiss"), *ws.var("mean")};
   plc.SetConfidenceLevel(confidenceLevel);
   std::unique_ptr<LikelihoodInterval> interval{plc.GetInterval()};

   // Reference limits are the solutions of 2*[ln(LL(xMax)) - ln(LL(x))] = 1,
   // where xMax is the point of maximum likelihood. For the special case of
   // the Poisson distribution with N = 1, xMax = obsValue.
   TString llRatioExpression =
      TString::Format("2*(x-%d*log(x)-%d+%d*log(%d))", obsValue, obsValue, obsValue, obsValue);
   // Special case obsValue = 0 because log(0) is not computable, the limit of
   // n * log(n), n->0 must be taken
   if (obsValue == 0)
      llRatioExpression = "2*x";
   TF1 llRatio{"llRatio", llRatioExpression, 1e-100, double(obsValue)}; // lowerLimit < obsValue

   // For obsValue = 0 there is no analytic lower limit (the reference value in
   // the original suite was NaN, which made its comparison pass trivially)
   if (obsValue != 0) {
      EXPECT_NEAR(interval->LowerLimit(*ws.var("mean")), llRatio.GetX(1), kVTol);
   }
   llRatio.SetRange(obsValue, 1000); // upperLimit > obsValue
   EXPECT_NEAR(interval->UpperLimit(*ws.var("mean")), llRatio.GetX(1), kVTol);
}

INSTANTIATE_TEST_SUITE_P(RooStats, PlcPoissonInterval,
                         ::testing::Combine(::testing::Values(ROOFIT_EVAL_BACKENDS),
                                            ::testing::Values(
                                               PlcPoissonParams{"Obs0", 0}, // boundary Poisson value (0)
                                               PlcPoissonParams{"Obs1", 1}, PlcPoissonParams{"Obs5", 5},
                                               PlcPoissonParams{"Obs100", 100},
                                               PlcPoissonParams{"Obs800", 800})), // boundary Poisson value
                         backendParamName<PlcPoissonParams>);

///////////////////////////////////////////////////////////////////////////////
//
// PROFILE LIKELIHOOD CALCULATOR - LIKELIHOOD INTERVAL - POISSON PRODUCT MODEL
//
// Test the 68% likelihood interval computed by the ProfileLikelihoodCalculator
// on a complex model. Reference values and test values are both computed with
// the ProfileLikelihoodCalculator. As such, this test can only confirm if the
// ProfileLikelihoodCalculator has the same behaviour across different computer
// platforms or RooStats revisions.
//
// ModelConfig (explicit) : Poisson Product Model
//    built in stressRooStats_models.h
//
// Original tests 11-13 (TestProfileLikelihoodCalculator3).
//
///////////////////////////////////////////////////////////////////////////////

struct PoissonProductParams {
   std::string name;
   int obsValueX; // observed value "x" when measuring sig + bkg1
   int obsValueY; // observed value "y" when measuring 2*sig*1.2^beta + bkg2
   double confidenceLevel;
   double refLowerLimit;
   double refUpperLimit;
};

class PlcPoissonProductInterval : public StressRooStatsParamTest<PoissonProductParams> {};

TEST_P(PlcPoissonProductInterval, RegressionInterval)
{
   // Create workspace and model with the observed values in the data set
   RooWorkspace ws{"w"};
   ModelConfig *model = setupPoissonProductModel(ws, param().obsValueX, param().obsValueY);
   ASSERT_NE(model, nullptr);

   // build likelihood interval with ProfileLikelihoodCalculator
   ProfileLikelihoodCalculator plc{*ws.data("data"), *model};
   plc.SetConfidenceLevel(param().confidenceLevel);
   std::unique_ptr<LikelihoodInterval> interval{plc.GetInterval()};

   EXPECT_NEAR(interval->LowerLimit(*ws.var("sig")), param().refLowerLimit, kVTol);
   EXPECT_NEAR(interval->UpperLimit(*ws.var("sig")), param().refUpperLimit, kVTol);
}

INSTANTIATE_TEST_SUITE_P(
   RooStats, PlcPoissonProductInterval,
   ::testing::Combine(
      ::testing::Values(ROOFIT_EVAL_BACKENDS),
      ::testing::Values(
         PoissonProductParams{"Obs10_30_CL1Sigma", 10, 30, kCL1Sigma, 7.1428311212946145, 12.216799640536237},
         PoissonProductParams{"Obs20_25_CL1Sigma", 20, 25, kCL1Sigma, 8.7079993530346087, 14.911357163771072},
         PoissonProductParams{"Obs15_20_CL2Sigma", 15, 20, kCL2Sigma, 3.4494551349741052, 14.067570959270299})),
   backendParamName<PoissonProductParams>);

///////////////////////////////////////////////////////////////////////////////
//
// PROFILE LIKELIHOOD CALCULATOR HYPOTHESIS TEST - ON / OFF MODEL
//
// Perform a hypothesis test using the ProfileLikelihoodCalculator on the
// on/off model. The reference values are taken from the paper: "Evaluation
// of three methods for calculating statistical significance when incorporating
// a systematic uncertainty into a test of the background-only hypothesis for
// a Poisson process" by Robert D. Cousins, James T. Linnemann, Jordan Tucker.
//
// ModelConfig (explicit) : Poisson On / Off Model
//    built in stressRooStats_models.h
//
// Original test 14 (TestProfileLikelihoodCalculator4).
//
///////////////////////////////////////////////////////////////////////////////

class PlcOnOffHypoTest : public StressRooStatsBackendTest {};

TEST_P(PlcOnOffHypoTest, CompareSignificanceWithPaperValues)
{
   // A larger tolerance is needed since the values in the Cousins paper are
   // given with 1e-2 precision
   const double tol = 1e-2;

   // For testing purposes, we consider three special cases for which the
   // values are known from the Cousins et al. paper mentioned above. The
   // inputs for each of these cases are (using the notations from the paper):
   // n_on, n_off, tau and Z_PL.
   const int numberTestSets = 3;
   const int numberOnEvents[numberTestSets] = {4, 50, 67};
   const int numberOffEvents[numberTestSets] = {5, 55, 15};
   const double tau[numberTestSets] = {5.0, 2.0, 0.5};
   const double significance[numberTestSets] = {1.95, 3.02, 3.04};

   for (int i = 0; i < numberTestSets; ++i) {

      // build workspace and model
      RooWorkspace ws{"w"};
      buildOnOffModel(ws);
      auto sbModel = dynamic_cast<ModelConfig *>(ws.obj("S+B"));
      auto bModel = dynamic_cast<ModelConfig *>(ws.obj("B"));
      ASSERT_NE(sbModel, nullptr);
      ASSERT_NE(bModel, nullptr);

      // add observable values to data set
      ws.var("n_on")->setVal(numberOnEvents[i]);
      ws.var("n_off")->setVal(numberOffEvents[i]);
      ws.var("tau")->setVal(tau[i]);
      ws.var("tau")->setConstant();
      ws.data("data")->add(*sbModel->GetObservables());

      // set snapshots
      ws.var("sig")->setVal(numberOnEvents[i] - numberOffEvents[i] / tau[i]);
      sbModel->SetSnapshot(*sbModel->GetParametersOfInterest());
      ws.var("sig")->setVal(0);
      bModel->SetSnapshot(*bModel->GetParametersOfInterest());

      // has as initial value a non-zero value for sig (i.e start with the S+B value)
      sbModel->LoadSnapshot();

      // get significance using the ProfileLikelihoodCalculator
      ProfileLikelihoodCalculator plc{*ws.data("data"), *sbModel};
      plc.SetNullParameters(*bModel->GetSnapshot());

      std::unique_ptr<HypoTestResult> result{plc.GetHypoTest()};
      EXPECT_NEAR(result->Significance(), significance[i], tol)
         << "for n_on = " << numberOnEvents[i] << ", n_off = " << numberOffEvents[i] << ", tau = " << tau[i];
   }
}

INSTANTIATE_TEST_SUITE_P(RooStats, PlcOnOffHypoTest, ::testing::Values(ROOFIT_EVAL_BACKENDS), backendName);

///////////////////////////////////////////////////////////////////////////////
//
// PART TWO: BAYESIAN CALCULATOR UNIT TESTS
//
///////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////
//
// BAYESIAN CENTRAL INTERVAL - SIMPLE POISSON MODEL
//
// Test the Bayesian central interval computed by the BayesianCalculator on a
// Poisson distribution, using different priors. The parameter of interest is
// the mean of the Poisson distribution, and there are no nuisance parameters.
// The priors used are:
//    1. constant / uniform
//    2. inverse of the mean
//    3. square root of the inverse of the mean
//    4. gamma distribution
// The posterior distribution is easily obtained analytically for these cases.
// Therefore, the reference interval limits are computed analytically.
//
// ModelConfig (implicit) :
//    Observable -> x
//    Parameter of Interest -> mean
//
// Original tests 15-18 (TestBayesianCalculator1).
//
///////////////////////////////////////////////////////////////////////////////

namespace {
double priorInvFunc(double mean)
{
   return 1.0 / mean;
}
double priorInvSqrtFunc(double mean)
{
   return 1.0 / std::sqrt(mean);
}
} // namespace

struct BcPoissonParams {
   std::string name;
   int obsValue;
};

class BcCentralInterval : public StressRooStatsParamTest<BcPoissonParams> {};

TEST_P(BcCentralInterval, CompareWithAnalyticInterval)
{
   const int obsValue = param().obsValue;

   // Set the confidence level for a 68.3% CL central interval
   const double confidenceLevel = kCL1Sigma;
   const double gammaShape = 2;   // shape of the gamma distribution prior (gamma = alpha)
   const double gammaRate = 1;    // rate = 1/scale of the gamma distribution prior (beta = 1/theta)
   const int numberScans = 10000; // tested to be sufficient for the scan of the Bayesian posterior

   // Create Poisson model
   RooWorkspace ws{"w"};
   ws.factory("Poisson::poiss(x[0,100], mean[1e-6,100])");

   // create prior pdfs
   ws.factory("Uniform::prior(mean)");
   ws.import(RooCFunction1PdfBinding<double, double>("priorInv", "priorInv", &priorInvFunc, *ws.var("mean")));
   ws.import(
      RooCFunction1PdfBinding<double, double>("priorInvSqrt", "priorInvSqrt", &priorInvSqrtFunc, *ws.var("mean")));
   ws.factory(TString::Format("Gamma::priorGamma(mean, %lf, %lf, 0)", gammaShape, gammaRate).Data());

   // build argument sets and data set
   ws.defineSet("obs", "x");
   ws.defineSet("poi", "mean");
   ws.var("x")->setVal(obsValue);
   RooDataSet data{"data", "data", *ws.set("obs")};
   data.add(*ws.set("obs"));

   // Compute the interval with the BayesianCalculator for the given prior and
   // compare with the analytically computed reference limits: the posterior
   // for a Poisson model with the priors used here is a gamma distribution.
   auto testPrior = [&](const char *priorName, double refLowerLimit, double refUpperLimit) {
      BayesianCalculator bc{data, *ws.pdf("poiss"), *ws.set("poi"), *ws.pdf(priorName), nullptr};
      bc.SetConfidenceLevel(confidenceLevel);
      bc.SetScanOfPosterior(numberScans);
      std::unique_ptr<SimpleInterval> interval{bc.GetInterval()};
      EXPECT_NEAR(interval->LowerLimit(), refLowerLimit, kVTol) << "lower limit for prior " << priorName;
      EXPECT_NEAR(interval->UpperLimit(), refUpperLimit, kVTol) << "upper limit for prior " << priorName;
   };

   const double testSize = (1.0 - confidenceLevel) / 2;

   // Uniform prior on mean
   testPrior("prior", gamma_quantile(testSize, obsValue + 1, 1), // integrate to 16%
             gamma_quantile_c(testSize, obsValue + 1, 1));       // integrate to 84%
   // Inverse of mean prior
   testPrior("priorInv", gamma_quantile(testSize, obsValue, 1), gamma_quantile_c(testSize, obsValue, 1));
   // Square root of inverse of mean prior
   testPrior("priorInvSqrt", gamma_quantile(testSize, obsValue + 0.5, 1),
             gamma_quantile_c(testSize, obsValue + 0.5, 1));
   // Gamma distribution prior
   testPrior("priorGamma", gamma_quantile(testSize, obsValue + gammaShape, 1.0 / (1 + gammaRate)),
             gamma_quantile_c(testSize, obsValue + gammaShape, 1.0 / (1 + gammaRate)));
}

INSTANTIATE_TEST_SUITE_P(RooStats, BcCentralInterval,
                         ::testing::Combine(::testing::Values(ROOFIT_EVAL_BACKENDS),
                                            ::testing::Values(BcPoissonParams{"Obs1", 1}, BcPoissonParams{"Obs3", 3},
                                                              BcPoissonParams{"Obs10", 10},
                                                              BcPoissonParams{"Obs50", 50})),
                         backendParamName<BcPoissonParams>);

///////////////////////////////////////////////////////////////////////////////
//
// BAYESIAN SHORTEST INTERVAL - SIMPLE POISSON MODEL
//
// Test the Bayesian shortest interval computed by the BayesianCalculator on a
// Poisson distribution, using different priors. The parameter of interest is
// the mean of the Poisson distribution, and there are no nuisance parameters.
// The priors used are:
//    1. constant / uniform
//    2. inverse of the mean
// The reference interval limits are taken from the paper: "Why isn't every
// physicist a Bayesian?" by Robert D. Cousins.
//
// Original test 19 (TestBayesianCalculator2).
//
///////////////////////////////////////////////////////////////////////////////

class BcShortestInterval : public StressRooStatsBackendTest {};

TEST_P(BcShortestInterval, CompareWithPaperValues)
{
   // the reference values in the paper have a precision of only two decimal
   // points, so we increase the value tolerance accordingly
   const double tol = 1e-2;

   // Put the confidence level so that we obtain a 68% confidence interval
   const double confidenceLevel = kCL1Sigma;
   const int obsValue = 3;         // observed experiment value
   const int numberScans = 100000; // sufficient number of scans

   // Create Poisson model
   RooWorkspace ws{"w"};
   ws.factory("Poisson::poiss(x[0,100], mean[1e-6,100])");
   ws.factory("Uniform::prior(mean)");
   ws.factory("EXPR::priorInv('1/mean', mean)");

   // build argument sets and data set
   ws.defineSet("poi", "mean");
   ws.defineSet("obs", "x");
   ws.var("x")->setVal(obsValue);
   RooDataSet data{"data", "data", *ws.set("obs")};
   data.add(*ws.set("obs"));

   auto testPrior = [&](const char *priorName, double refLowerLimit, double refUpperLimit) {
      BayesianCalculator bc{data, *ws.pdf("poiss"), *ws.set("poi"), *ws.pdf(priorName), nullptr};
      bc.SetConfidenceLevel(confidenceLevel);
      bc.SetShortestInterval();
      bc.SetScanOfPosterior(numberScans);
      std::unique_ptr<SimpleInterval> interval{bc.GetInterval()};
      EXPECT_NEAR(interval->LowerLimit(), refLowerLimit, tol) << "lower limit for prior " << priorName;
      EXPECT_NEAR(interval->UpperLimit(), refUpperLimit, tol) << "upper limit for prior " << priorName;
   };

   // Uniform prior on mean
   testPrior("prior", 1.55, 5.15);
   // Inverse of mean prior
   testPrior("priorInv", 0.86, 3.85);
}

INSTANTIATE_TEST_SUITE_P(RooStats, BcShortestInterval, ::testing::Values(ROOFIT_EVAL_BACKENDS), backendName);

///////////////////////////////////////////////////////////////////////////////
//
// BAYESIAN CENTRAL INTERVAL - POISSON PRODUCT MODEL
//
// Test the validity of the central interval computed by the BayesianCalculator
// on a complex Poisson model distribution. Reference values and test values
// are both computed with the BayesianCalculator. As such, this test can only
// confirm if the BayesianCalculator has the same behaviour across different
// computing platforms or RooStats revisions. A uniform prior PDF is used for
// the parameter of interest ("sig").
//
// ModelConfig (explicit) : Poisson Product Model
//    built in stressRooStats_models.h
//
// Original tests 20-22 (TestBayesianCalculator3).
//
///////////////////////////////////////////////////////////////////////////////

class BcPoissonProductInterval : public StressRooStatsParamTest<PoissonProductParams> {};

TEST_P(BcPoissonProductInterval, RegressionInterval)
{
   const int numberScans = 10; // sufficient number of scans

   // Create workspace and model with the observed values in the data set
   RooWorkspace ws{"w"};
   ModelConfig *model = setupPoissonProductModel(ws, param().obsValueX, param().obsValueY);
   ASSERT_NE(model, nullptr);

   // Create BayesianCalculator
   BayesianCalculator bc{*ws.data("data"), *model};
   bc.SetConfidenceLevel(param().confidenceLevel);
   bc.SetScanOfPosterior(numberScans);

   // Obtain confidence interval by scanning the posterior function in the
   // given number of points
   std::unique_ptr<SimpleInterval> interval{bc.GetInterval()};
   EXPECT_NEAR(interval->LowerLimit(), param().refLowerLimit, kVTol);
   EXPECT_NEAR(interval->UpperLimit(), param().refUpperLimit, kVTol);
}

INSTANTIATE_TEST_SUITE_P(
   RooStats, BcPoissonProductInterval,
   ::testing::Combine(
      ::testing::Values(ROOFIT_EVAL_BACKENDS),
      ::testing::Values(
         PoissonProductParams{"Obs10_30_CL1Sigma", 10, 30, kCL1Sigma, 7.1665080051981258, 12.312785237149114},
         PoissonProductParams{"Obs20_25_CL1Sigma", 20, 25, kCL1Sigma, 8.7831170668504654, 14.874961130060584},
         PoissonProductParams{"Obs15_20_CL2Sigma", 15, 20, kCL2Sigma, 3.4603352565856857, 14.186182799724543})),
   backendParamName<PoissonProductParams>);

///////////////////////////////////////////////////////////////////////////////
//
// PART THREE: MARKOV CHAIN MONTE CARLO CALCULATOR UNIT TESTS
//
///////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////
//
// MCMC INTERVAL CALCULATOR - POISSON PRODUCT MODEL
//
// Test the validity of the confidence interval computed by the MCMCCalculator
// on a complex Poisson model distribution. Reference values and test values
// are both computed with the MCMCCalculator. As such, this test can only
// confirm if the MCMCCalculator has the same behaviour across different
// computing platforms or RooStats revisions.
//
// ModelConfig (explicit) : Poisson Product Model
//    built in stressRooStats_models.h
//
// Original tests 23-25 (TestMCMCCalculator).
//
///////////////////////////////////////////////////////////////////////////////

class McmcPoissonProductInterval : public StressRooStatsParamTest<PoissonProductParams> {};

TEST_P(McmcPoissonProductInterval, RegressionInterval)
{
   // Create workspace and model with the observed values in the data set
   RooWorkspace ws{"w"};
   ModelConfig *model = setupPoissonProductModel(ws, param().obsValueX, param().obsValueY);
   ASSERT_NE(model, nullptr);

   // create and configure MCMC calculator
   SequentialProposal sp{0.1};
   MCMCCalculator mcmcc{*ws.data("data"), *model};
   mcmcc.SetProposalFunction(sp);
   mcmcc.SetNumIters(100000);   // Metropolis-Hastings algorithm iterations
   mcmcc.SetNumBurnInSteps(50); // first 50 steps to be ignored as burn-in
   mcmcc.SetConfidenceLevel(param().confidenceLevel);

   // calculate the confidence interval
   std::unique_ptr<MCMCInterval> interval{mcmcc.GetInterval()};
   EXPECT_NEAR(interval->LowerLimit(*ws.var("sig")), param().refLowerLimit, kVTol);
   EXPECT_NEAR(interval->UpperLimit(*ws.var("sig")), param().refUpperLimit, kVTol);
}

INSTANTIATE_TEST_SUITE_P(
   RooStats, McmcPoissonProductInterval,
   ::testing::Combine(::testing::Values(ROOFIT_EVAL_BACKENDS),
                      ::testing::Values(PoissonProductParams{"Obs10_30_CL1Sigma", 10, 30, kCL1Sigma, 7.0, 11.4},
                                        PoissonProductParams{"Obs20_25_CL1Sigma", 20, 25, kCL1Sigma, 9.0,
                                                             14.600000000000001},
                                        PoissonProductParams{"Obs15_20_CL2Sigma", 15, 20, kCL2Sigma,
                                                             3.4000000000000004, 13.800000000000001})),
   backendParamName<PoissonProductParams>);

///////////////////////////////////////////////////////////////////////////////
//
// PART FOUR: HYPOTHESIS TEST CALCULATOR UNIT TESTS
//
///////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////
//
// ZBI - ON / OFF MODEL
//
// Evaluate the functionality of the top level function
// NumberCountingUtils::BinomialWithTauObsZ, which computes the significance
// of a hypothesis test via a frequentist solution. This significance, called
// ZBi, is detailed in the article "Evaluation of three methods for calculating
// statistical significance when incorporating a systematic uncertainty into a
// test of the background-only hypothesis for a Poisson process" by Robert D.
// Cousins, James T. Linnemann, Jordan Tucker. The reference values are taken
// from the paper.
//
// This computation involves no likelihood fits, so it is not parameterized
// over the evaluation backends.
//
// Original test 26 (TestZBi).
//
///////////////////////////////////////////////////////////////////////////////

TEST(ZBiSignificance, CompareWithPaperValues)
{
   setUpStressRooStatsTest();

   // A larger tolerance is needed since the values in the Cousins paper are
   // given with 1e-2 precision
   const double tol = 1e-2;

   // For testing purposes, we consider four special cases for which the values
   // are known from the Cousins et al. paper mentioned above. The inputs for
   // each of these cases are (using the notations from the paper): n_on, n_off
   // and tau.
   const int numberTestSets = 4;
   const int numberOnEvents[numberTestSets] = {4, 50, 67, 200};
   const int numberOffEvents[numberTestSets] = {5, 55, 15, 10};
   const double tau[numberTestSets] = {5.0, 2.0, 0.5, 0.1};
   const double significance[numberTestSets] = {1.66, 2.93, 2.89, 2.2};

   for (int i = 0; i < numberTestSets; ++i) {
      EXPECT_NEAR(NumberCountingUtils::BinomialWithTauObsZ(numberOnEvents[i], numberOffEvents[i], tau[i]),
                  significance[i], tol)
         << "for n_on = " << numberOnEvents[i] << ", n_off = " << numberOffEvents[i] << ", tau = " << tau[i];
   }

   tearDownStressRooStatsTest();
}

///////////////////////////////////////////////////////////////////////////////
//
// ASYMPTOTIC CALCULATOR VS PROFILE LIKELIHOOD CALCULATOR HYPOTHESIS TEST
//
// This test evaluates the functionality of the AsymptoticCalculator by
// comparing the significance given from a hypothesis test on the on/off model
// with the significance given by the ProfileLikelihoodCalculator. If working
// properly, the two methods should yield identical results. On top of the
// direct comparison of the two methods, both significances are also compared
// with the frozen reference values of the original suite (which were produced
// with the ProfileLikelihoodCalculator).
//
// ModelConfig (explicit) : Poisson On / Off Model
//    built in stressRooStats_models.h
//
// Original tests 27-31 (TestHypoTestCalculator1).
//
///////////////////////////////////////////////////////////////////////////////

struct OnOffSignificanceParams {
   std::string name;
   int obsValueOn;  // observed value "n_on" of sig + bkg
   int obsValueOff; // observed value "n_off" of tau * bkg
   double tau;      // parameter of the model (constant with regard to integration)
   double refSignificance;
};

class AsymptoticVsPlcSignificance : public StressRooStatsParamTest<OnOffSignificanceParams> {};

TEST_P(AsymptoticVsPlcSignificance, CompareSignificances)
{
   // build workspace and model, add observable values to the data set and fix
   // other parameters, then make the S+B and B snapshots
   auto setupWorkspace = [&](RooWorkspace &ws) -> std::pair<ModelConfig *, ModelConfig *> {
      buildOnOffModel(ws);
      auto sbModel = dynamic_cast<ModelConfig *>(ws.obj("S+B"));
      auto bModel = dynamic_cast<ModelConfig *>(ws.obj("B"));
      if (!sbModel || !bModel)
         return {nullptr, nullptr};

      ws.var("n_on")->setVal(param().obsValueOn);
      ws.var("n_off")->setVal(param().obsValueOff);
      ws.var("tau")->setVal(param().tau);
      ws.var("tau")->setConstant();
      ws.data("data")->add(*sbModel->GetObservables());
      ws.var("bkg")->setVal(param().obsValueOff / param().tau);

      ws.var("sig")->setVal(param().obsValueOn - param().obsValueOff / param().tau);
      sbModel->SetSnapshot(*sbModel->GetParametersOfInterest());
      ws.var("sig")->setVal(0.0);
      bModel->SetSnapshot(*bModel->GetParametersOfInterest());

      return std::make_pair(sbModel, bModel);
   };

   // Hypothesis test with the ProfileLikelihoodCalculator
   double significancePlc = 0.0;
   {
      RooWorkspace ws{"w"};
      auto [sbModel, bModel] = setupWorkspace(ws);
      ASSERT_NE(sbModel, nullptr);
      ASSERT_NE(bModel, nullptr);

      ProfileLikelihoodCalculator plc{*ws.data("data"), *sbModel};
      plc.SetNullParameters(*bModel->GetSnapshot());
      plc.SetAlternateParameters(*sbModel->GetSnapshot());
      std::unique_ptr<HypoTestResult> result{plc.GetHypoTest()};
      significancePlc = result->Significance();
   }

   // Hypothesis test with the AsymptoticCalculator
   double significanceAc = 0.0;
   {
      RooWorkspace ws{"w"};
      auto [sbModel, bModel] = setupWorkspace(ws);
      ASSERT_NE(sbModel, nullptr);
      ASSERT_NE(bModel, nullptr);

      AsymptoticCalculator atc{*ws.data("data"), *sbModel, *bModel};
      atc.SetOneSidedDiscovery(true);
      std::unique_ptr<HypoTestResult> result{atc.GetHypoTest()};
      significanceAc = result->Significance();
   }

   EXPECT_NEAR(significancePlc, param().refSignificance, kVTol);
   EXPECT_NEAR(significanceAc, param().refSignificance, kVTol);
}

INSTANTIATE_TEST_SUITE_P(
   RooStats, AsymptoticVsPlcSignificance,
   ::testing::Combine(
      ::testing::Values(ROOFIT_EVAL_BACKENDS),
      ::testing::Values(OnOffSignificanceParams{"Obs150_100_Tau1", 150, 100, 1.0, 3.1729725711197787},
                        OnOffSignificanceParams{"Obs200_100_Tau1", 200, 100, 1.0, 5.8292198861163191},
                        OnOffSignificanceParams{"Obs105_100_Tau1", 105, 100, 1.0, 0.34923144876992013},
                        OnOffSignificanceParams{"Obs150_10_Tau0p1", 150, 10, 0.1, 1.3181915918803733},
                        OnOffSignificanceParams{"Obs150_400_Tau4", 150, 400, 4.0, 4.0985771507906898})),
   backendParamName<OnOffSignificanceParams>);

///////////////////////////////////////////////////////////////////////////////
//
// HYPOTHESIS TEST CALCULATOR TEST - SIMULTANEOUS PDF MODEL
//
// This test evaluates the functionality of the HypoTestCalculator by
// calculating the significance of the signal on a simple Simultaneous Pdf
// model with two channels. Reference values and test values are both computed
// with the HypoTestCalculator. As such, this test can only confirm if the
// HypoTestCalculator has the same behaviour across different computing
// platforms or RooStats revisions.
//
// ModelConfig (explicit) : Simultaneous Model
//    built in stressRooStats_models.h
//
// Original tests 32-36 (TestHypoTestCalculator2).
//
///////////////////////////////////////////////////////////////////////////////

struct HypoTestCalculatorParams {
   std::string name;
   ECalculatorType calculatorType;
   ETestStatType testStatType;
   double refSignificance;
};

class HypoTestCalculatorSignificance : public StressRooStatsParamTest<HypoTestCalculatorParams> {};

TEST_P(HypoTestCalculatorSignificance, RegressionSignificance)
{
   // Build workspace and models
   RooWorkspace ws{"w"};
   buildSimultaneousModel(&ws);
   auto sbModel = dynamic_cast<ModelConfig *>(ws.obj("S+B"));
   auto bModel = dynamic_cast<ModelConfig *>(ws.obj("B"));
   ASSERT_NE(sbModel, nullptr);
   ASSERT_NE(bModel, nullptr);

   // set snapshots
   sbModel->SetSnapshot(*sbModel->GetParametersOfInterest()); // value set in model
   ws.var("sig")->setVal(0);
   bModel->SetSnapshot(*bModel->GetParametersOfInterest());

   // the test statistic is declared before the calculator because the
   // calculator's sampler will hold a raw pointer to it
   std::unique_ptr<TestStatistic> testStat{buildTestStatistic(param().testStatType, *bModel, *sbModel)};
   std::unique_ptr<HypoTestCalculatorGeneric> calc{
      buildHypoTestCalculator(param().calculatorType, *ws.data("data"), *bModel, *sbModel, 500, 50)};
   if (param().calculatorType == kAsymptotic) {
      static_cast<AsymptoticCalculator &>(*calc).SetOneSidedDiscovery(true);
   }

   // ToyMCSampler configuration
   auto tmcs = static_cast<ToyMCSampler *>(calc->GetTestStatSampler());
   tmcs->SetTestStatistic(testStat.get());
   tmcs->SetUseMultiGen(true); // speedup

   std::unique_ptr<HypoTestResult> result{calc->GetHypoTest()};
   EXPECT_NEAR(result->Significance(), param().refSignificance, kVTol);
}

INSTANTIATE_TEST_SUITE_P(
   RooStats, HypoTestCalculatorSignificance,
   ::testing::Combine(
      ::testing::Values(ROOFIT_EVAL_BACKENDS),
      ::testing::Values(HypoTestCalculatorParams{"Asymptotic_ProfileLROneSidedDiscovery", kAsymptotic,
                                                 kProfileLROneSidedDiscovery, 2.2499193885237001},
                        HypoTestCalculatorParams{"Frequentist_SimpleLR", kFrequentist, kSimpleLR, 2.3263478740408408},
                        HypoTestCalculatorParams{"Frequentist_RatioLR", kFrequentist, kRatioLR, 2.4089155458154612},
                        HypoTestCalculatorParams{"Frequentist_ProfileLROneSidedDiscovery", kFrequentist,
                                                 kProfileLROneSidedDiscovery, 2.5121443279304616},
                        HypoTestCalculatorParams{"Hybrid_ProfileLROneSidedDiscovery", kHybrid,
                                                 kProfileLROneSidedDiscovery, 2.2571292444862254})),
   backendParamName<HypoTestCalculatorParams>);

///////////////////////////////////////////////////////////////////////////////
//
// PART FIVE: HYPOTHESIS TEST INVERTER UNIT TESTS
//
///////////////////////////////////////////////////////////////////////////////

///////////////////////////////////////////////////////////////////////////////
//
// HYPOTESTINVERTER INTERVAL - POISSON PRODUCT MODEL
//
// Test the validity of the confidence interval computed by the
// HypoTestInverter on a complex Poisson model distribution. Reference values
// and test values are both computed with the HypoTestInverter. As such, this
// test can only confirm if the HypoTestInverter has the same behaviour across
// different computing platforms or RooStats revisions.
//
// ModelConfig (explicit) : Poisson Product Model
//    built in stressRooStats_models.h
//
// Original tests 37-43 (TestHypoTestInverter1).
//
///////////////////////////////////////////////////////////////////////////////

struct HypoTestInverterParams {
   std::string name;
   ECalculatorType calculatorType;
   ETestStatType testStatType;
   int obsValueX; // observed value "x" when measuring sig + bkg1
   int obsValueY; // observed value "y" when measuring 2*sig*1.2^beta + bkg2
   double refLowerLimit;
   double refUpperLimit;
};

class HypoTestInverterInterval : public StressRooStatsParamTest<HypoTestInverterParams> {};

TEST_P(HypoTestInverterInterval, RegressionInterval)
{
   const double confidenceLevel = kCL1Sigma;

   // larger value test tolerance especially when using toys (difference of
   // <~ 0.1 observed between using Minuit or Minuit2)
   const double tol = (param().calculatorType == kAsymptotic) ? 0.01 : 0.1;

   // Create workspace and model with the observed values in the data set
   RooWorkspace ws{"w"};
   ModelConfig *sbModel = setupPoissonProductModel(ws, param().obsValueX, param().obsValueY);
   auto bModel = dynamic_cast<ModelConfig *>(ws.obj("B"));
   ASSERT_NE(sbModel, nullptr);
   ASSERT_NE(bModel, nullptr);

   // set snapshots
   ws.var("sig")->setVal(param().obsValueX - ws.var("bkg1")->getValV());
   sbModel->SetSnapshot(*sbModel->GetParametersOfInterest());
   ws.var("sig")->setVal(0);
   bModel->SetSnapshot(*bModel->GetParametersOfInterest());

   // build and configure HypoTestInverter (the test statistic is declared
   // first because the calculator will hold a raw pointer to it)
   std::unique_ptr<TestStatistic> testStat{buildTestStatistic(param().testStatType, *sbModel, *bModel)};
   std::unique_ptr<HypoTestCalculatorGeneric> calc{
      buildHypoTestCalculator(param().calculatorType, *ws.data("data"), *sbModel, *bModel, 100, 1)};
   HypoTestInverter hti{*calc, nullptr, 1.0 - confidenceLevel};
   hti.SetTestStatistic(*testStat);

   int nscanPoints = 10;
   if (param().calculatorType == kAsymptotic) {
      static_cast<AsymptoticCalculator &>(*calc).SetTwoSided();
      nscanPoints = 40;
   }

   hti.SetFixedScan(nscanPoints, ws.var("sig")->getMin(), ws.var("sig")->getMax()); // significant speedup

   // ToyMCSampler configuration
   auto tmcs = static_cast<ToyMCSampler *>(hti.GetHypoTestCalculator()->GetTestStatSampler());
   tmcs->SetNEventsPerToy(1);  // needed because we don't have an extended pdf
   tmcs->SetUseMultiGen(true); // speedup

   std::unique_ptr<HypoTestInverterResult> interval{hti.GetInterval()};
   EXPECT_NEAR(interval->LowerLimit(), param().refLowerLimit, tol);
   EXPECT_NEAR(interval->UpperLimit(), param().refUpperLimit, tol);
}

INSTANTIATE_TEST_SUITE_P(
   RooStats, HypoTestInverterInterval,
   ::testing::Combine(::testing::Values(ROOFIT_EVAL_BACKENDS),
                      ::testing::Values(HypoTestInverterParams{"Asymptotic_ProfileLR_Obs10_30", kAsymptotic,
                                                               kProfileLR, 10, 30, 7.1386012828228216,
                                                               12.223793911203613},
                                        HypoTestInverterParams{"Asymptotic_ProfileLR_Obs20_25", kAsymptotic,
                                                               kProfileLR, 20, 25, 8.7069484829792732,
                                                               14.91407242494005},
                                        HypoTestInverterParams{"Asymptotic_ProfileLR_Obs15_20", kAsymptotic,
                                                               kProfileLR, 15, 20, 5.750533033549976,
                                                               10.948013504988015},
                                        HypoTestInverterParams{"Frequentist_ProfileLR_Obs10_30", kFrequentist,
                                                               kProfileLR, 10, 30, 7.2115486004250906,
                                                               12.341346144450787},
                                        HypoTestInverterParams{"Frequentist_ProfileLR_Obs20_25", kFrequentist,
                                                               kProfileLR, 20, 25, 8.6294497270104724,
                                                               15.078331964885663},
                                        HypoTestInverterParams{"Frequentist_ProfileLR_Obs15_20", kFrequentist,
                                                               kProfileLR, 15, 20, 5.6610441994727472,
                                                               10.87954812525739},
                                        HypoTestInverterParams{"Hybrid_ProfileLR_Obs10_30", kHybrid, kProfileLR, 10,
                                                               30, 7.1079825543826782, 12.553916197221213})),
   backendParamName<HypoTestInverterParams>);

///////////////////////////////////////////////////////////////////////////////
//
// HYPOTESTINVERTER UPPER LIMIT - SIGNAL + BACKGROUND + EFFICIENCY MODEL
//
// Test the validity of the upper limit computed by the HypoTestInverter
// on a complex model distribution with signal, background and efficiency.
// Reference values and test values are both computed with the
// HypoTestInverter. As such, this test can only confirm if the
// HypoTestInverter has the same behaviour across different computing platforms
// or RooStats revisions.
//
// ModelConfig (explicit) : Poisson Signal + Background + Efficiency
//    built in stressRooStats_models.h
//
// Original tests 44-48 (TestHypoTestInverter2).
//
///////////////////////////////////////////////////////////////////////////////

struct HypoTestInverterUpperLimitParams {
   std::string name;
   ECalculatorType calculatorType;
   ETestStatType testStatType;
   int obsValueX; // observed value "x" when measuring sig * eff + bkg
   double confidenceLevel;
   double refUpperLimit;
   double refExpUpperLimit;
   double refExpUpperLimitMinus2;
   double refExpUpperLimitMinus1;
   double refExpUpperLimitPlus1;
   double refExpUpperLimitPlus2;
};

class HypoTestInverterUpperLimit : public StressRooStatsParamTest<HypoTestInverterUpperLimitParams> {};

TEST_P(HypoTestInverterUpperLimit, RegressionUpperLimit)
{
   // larger value test tolerance especially when using toys (difference of
   // <~ 0.1 observed between using Minuit or Minuit2)
   const double tol = (param().calculatorType == kAsymptotic) ? 0.02 : 0.1;

   // Create workspace and model
   RooWorkspace ws{"w"};
   buildPoissonEfficiencyModel(ws);
   auto sbModel = dynamic_cast<ModelConfig *>(ws.obj("S+B"));
   auto bModel = dynamic_cast<ModelConfig *>(ws.obj("B"));
   ASSERT_NE(sbModel, nullptr);
   ASSERT_NE(bModel, nullptr);

   // add observed values to data set
   ws.var("x")->setVal(param().obsValueX);
   ws.data("data")->add(*sbModel->GetObservables());

   // set snapshots
   sbModel->SetSnapshot(*sbModel->GetParametersOfInterest());
   ws.var("sig")->setVal(0);
   bModel->SetSnapshot(*bModel->GetParametersOfInterest());

   // calculate upper limit with HypoTestInverter (the test statistic is
   // declared first because the calculator will hold a raw pointer to it)
   std::unique_ptr<TestStatistic> testStat{buildTestStatistic(param().testStatType, *sbModel, *bModel)};
   std::unique_ptr<HypoTestCalculatorGeneric> calc{
      buildHypoTestCalculator(param().calculatorType, *ws.data("data"), *sbModel, *bModel, 100, 100)};
   HypoTestInverter hti{*calc, nullptr, 1.0 - param().confidenceLevel};
   hti.SetTestStatistic(*testStat);

   int nscanPoints = 10;
   if (param().calculatorType == kAsymptotic) {
      static_cast<AsymptoticCalculator &>(*calc).SetOneSided(true);
      nscanPoints = 40;
   }

   hti.SetFixedScan(nscanPoints, ws.var("sig")->getMin(), ws.var("sig")->getMax()); // significant speedup

   // needed because we have no extended pdf and the ToyMC Sampler evaluation
   // returns an error
   auto tmcs = static_cast<ToyMCSampler *>(hti.GetHypoTestCalculator()->GetTestStatSampler());
   tmcs->SetNEventsPerToy(1);
   tmcs->SetUseMultiGen(true); // make ToyMCSampler faster

   // calculate interval and extract observed upper limit and expected upper
   // limit (+- 1, 2 sigma)
   std::unique_ptr<HypoTestInverterResult> interval{hti.GetInterval()};
   EXPECT_NEAR(interval->UpperLimit(), param().refUpperLimit, tol);
   EXPECT_NEAR(interval->GetExpectedUpperLimit(0), param().refExpUpperLimit, tol);
   EXPECT_NEAR(interval->GetExpectedUpperLimit(-2), param().refExpUpperLimitMinus2, tol);
   EXPECT_NEAR(interval->GetExpectedUpperLimit(-1), param().refExpUpperLimitMinus1, tol);
   EXPECT_NEAR(interval->GetExpectedUpperLimit(1), param().refExpUpperLimitPlus1, tol);
   EXPECT_NEAR(interval->GetExpectedUpperLimit(2), param().refExpUpperLimitPlus2, tol);
}

INSTANTIATE_TEST_SUITE_P(
   RooStats, HypoTestInverterUpperLimit,
   ::testing::Combine(::testing::Values(ROOFIT_EVAL_BACKENDS),
                      ::testing::Values(HypoTestInverterUpperLimitParams{"Asymptotic_ProfileLROneSided_Obs10_CL0p95",
                                                                         kAsymptotic, kProfileLROneSided, 10, 0.95,
                                                                         24.415175188860971, 11.71352859205691, 0.0,
                                                                         4.1446917240185863, 21.580917321964726,
                                                                         34.586630899315331},
                                        HypoTestInverterUpperLimitParams{"Asymptotic_ProfileLROneSided_Obs20_CL1Sigma",
                                                                         kAsymptotic, kProfileLROneSided, 20,
                                                                         kCL1Sigma, 35.455729668223988,
                                                                         3.8701997338910363, 0.0, 0.0,
                                                                         13.115199373487766, 24.476388830684247},
                                        HypoTestInverterUpperLimitParams{"Frequentist_RatioLR_Obs10_CL0p95",
                                                                         kFrequentist, kRatioLR, 10, 0.95,
                                                                         27.777734839933814, 13.425920689335342,
                                                                         5.2777802875157622, 6.4814835475786641,
                                                                         20.000008863407345, 30.555555555555557},
                                        HypoTestInverterUpperLimitParams{"Frequentist_ProfileLROneSided_Obs10_CL0p95",
                                                                         kFrequentist, kProfileLROneSided, 10, 0.95,
                                                                         23.611111111111089, 14.444453307851788,
                                                                         5.2777802875157622, 6.4814835475786641,
                                                                         20.251355120121389, 29.629641114958538},
                                        HypoTestInverterUpperLimitParams{"Hybrid_SimpleLR_Obs10_CL0p95", kHybrid,
                                                                         kSimpleLR, 10, 0.95, 25.0,
                                                                         11.111095211258252, 5.2777802875157622,
                                                                         7.6388877462703384, 20.201996699382633,
                                                                         26.376273186118574})),
   backendParamName<HypoTestInverterUpperLimitParams>);
