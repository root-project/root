// ROOT headers
#include "TCanvas.h"
#include "TMath.h"

// RooFit headers
#ifndef __CINT__
#include "RooGlobalFunc.h"
#endif
#include "RooPlot.h"
#include "RooUnitTest.h"
#include "RooRealVar.h"
#include "RooDataSet.h"

// RooStats headers
#include "RooStats/NumberCountingUtils.h"
#include "RooStats/RooStatsUtils.h"
#include "RooStats/TestStatistic.h"

#include "stressRooStats_models.cxx" // Global functions that build complex RooStats models

using namespace ROOT::Math;
using namespace RooFit;
using namespace RooStats;
using namespace std;

// testStatType = 0 Simple Likelihood Ratio (the LEP TestStat)
//              = 1 Ratio of Profiled Likelihood Ratios (the Tevatron TestStat)
//              = 2 Profile Likelihood Ratio (the LHC TestStat)
//              = 3 Profile Likelihood One Sided (pll = 0 if mu < mu_hat)
//              = 4 Profile Likelihood Signed (pll = -pll if mu < mu_hat)
//              = 5 Max Likelihood Estimate as test statistic
//              = 6 Number of Observed Events as test statistic
enum ETestStatType { kSimpleLR = 0, kRatioLR = 1, kProfileLR = 2, kProfileLROneSided = 3, kProfileLRSigned = 4, kMLE = 5, kNObs = 6 };
static const char * const kECalculatorTypeString[] = { "Undefined", "Hybrid", "Frequentist", "Asymptotic" };
static const char * const kETestStatTypeString[] = { "Simple Likelihood Ratio", "Ratio Likelihood Ratio", "Profile Likelihood Ratio", "Profile Likelihood One Sided", "Profile Likelihood Signed", "Max Likelihood Estimate", "Number of Observed Events" };
static TestStatistic *buildTestStatistic(const ETestStatType testStatType, const ModelConfig &sbModel, const ModelConfig &bModel);




//_____________________________________________________________________________
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
//
// PART ONE:
//    PROFILE LIKELIHOOD CALCULATOR UNIT TESTS
//

#include "RooStats/ProfileLikelihoodCalculator.h"
#include "RooStats/LikelihoodInterval.h"
#include "RooStats/LikelihoodIntervalPlot.h"
#include "RooStats/HypoTestResult.h"

///////////////////////////////////////////////////////////////////////////////
//
// PROFILE LIKELIHOOD CALCULATOR - LIKELIHOOD INTERVAL - GAUSSIAN DISTRIBUTION
//
// Test the likelihood interval computed by the profile likelihood calculator
// on a Gaussian distribution. Reference interval limits are computed via
// analytic methods: solve equation 2*(ln(LL(xMax))-ln(LL(x)) = q, where q =
// normal_quantile_c(testSize/2, 1). In the case of a Gaussian distribution, the
// interval limits are equal to: mean +- normal_quantile_c(testSize/2, sigma/sqrt(N)).
//
// ModelConfig (implicit) :
//    Observable -> x
//    Parameter of Interest -> mean
//    Nuisance parameter (Constant !) -> sigma
//
// Input Parameters:
//    confidenceLevel -> Confidence Level of the interval we are calculating
//
// 03/2012 - Ioan Gabriel Bucur
//
///////////////////////////////////////////////////////////////////////////////

class TestProfileLikelihoodCalculator1 : public RooUnitTest {
private:
   Double_t fConfidenceLevel;

public:
   TestProfileLikelihoodCalculator1(
      TFile* refFile,
      Bool_t writeRef,
      Int_t verbose,
      Double_t confidenceLevel = 0.95
   ) :
      RooUnitTest("ProfileLikelihoodCalculator Interval - Gaussian Model", refFile, writeRef, verbose),
      fConfidenceLevel(confidenceLevel)
   {};

   // Basic checks for the parameters passed to the test
   // In case of invalid parameters, a warning is printed and the test is skipped
   Bool_t isTestAvailable() {
      if (fConfidenceLevel <= 0.0 || fConfidenceLevel >= 1.0) {
         Warning("isTestAvailable", "Confidence level must be in the range (0,1). Skipping test...");
         return kFALSE;
      }
      return kTRUE;
   }

   Bool_t testCode() {

      const Int_t N = 10; // number of observations
      // the compared values / objects must have the same name in write / compare modes
      const TString lowerLimitString = TString::Format("tplc2_lower_limit_mean_%lf", fConfidenceLevel);
      const TString upperLimitString = TString::Format("tplc2_upper_limit_mean_%lf", fConfidenceLevel);

      //TODO: see why it fails for a small number of observations
      // Create Gaussian model, generate data set and define
      RooWorkspace* w = new RooWorkspace("w", kTRUE);
      w->factory("Gaussian::gauss(x[-5,5], mean[0,-5,5], sigma[1])");
      RooDataSet *data = w->pdf("gauss")->generate(*w->var("x"), N);

      if (_write == kTRUE) {

         // Calculate likelihood interval from data via analytic methods
         Double_t estMean = data->mean(*w->var("x"));
         Double_t intervalHalfWidth =
            normal_quantile_c((1.0 - fConfidenceLevel) / 2.0, w->var("sigma")->getValV() / sqrt(N));
         Double_t lowerLimit = estMean - intervalHalfWidth;
         Double_t upperLimit = estMean + intervalHalfWidth;

         // Compare the limits obtained via ProfileLikelihoodCalculator with analytically estimated values
         regValue(lowerLimit, lowerLimitString);
         regValue(upperLimit, upperLimitString);

      } else {

         // Calculate likelihood interval using the ProfileLikelihoodCalculator
         ProfileLikelihoodCalculator *plc = new ProfileLikelihoodCalculator(*data, *w->pdf("gauss"), *w->var("mean"));
         plc->SetConfidenceLevel(fConfidenceLevel);
         LikelihoodInterval *interval = plc->GetInterval();

         // Register analytically computed limits in the reference file
         regValue(interval->LowerLimit(*w->var("mean")), lowerLimitString);
         regValue(interval->UpperLimit(*w->var("mean")), upperLimitString);

         // Cleanup branch objects
         delete plc;
         delete interval;
      }

      // Cleanup local objects
      delete data;
      delete w;

      return kTRUE ;
   }
};


///////////////////////////////////////////////////////////////////////////////
//
// PROFILE LIKELIHOOD CALCULATOR - LIKELIHOOD INTERVAL - POISSON DISTRIBUTION
//
// Test the 68% likelihood interval computed by the profile likelihood calculator
// on a Poisson distribution, from only one observed value. Reference values are
// computed via analytic methods: solve equation 2*[ln(LL(xMax)) - ln(LL(x))] = 1.
//
// ModelConfig (implicit) :
//    Observable -> x
//    Parameter of Interest -> mean
//
// Input Parameters:
//    obsValue -> observed value in experiment
//
// 03/2012 - Ioan Gabriel Bucur
//
///////////////////////////////////////////////////////////////////////////////

class TestProfileLikelihoodCalculator2 : public RooUnitTest {
private:
   Int_t fObsValue;

public:
   TestProfileLikelihoodCalculator2(
      TFile* refFile,
      Bool_t writeRef,
      Int_t verbose,
      Int_t obsValue = 5
   ) :
      RooUnitTest("ProfileLikelihoodCalculator Interval - Poisson Simple Model", refFile, writeRef, verbose),
      fObsValue(obsValue)
   {};

   // Basic checks for the parameters passed to the test
   // In case of invalid parameters, a warning is printed and the test is skipped
   Bool_t isTestAvailable() {
      if (fObsValue < 0 || fObsValue > 1000) {
         Warning("isTestAvailable", "Observed value must be in the range [0,1000]. Skipping test...");
         return kFALSE;
      }
      return kTRUE;
   }

   Bool_t testCode() {

      // the compared values / objects must have the same name in write / compare modes
      const TString lowerLimitString = TString::Format("tplc2_lower_limit_mean_%d", fObsValue);
      const TString upperLimitString = TString::Format("tplc2_upper_limit_mean_%d", fObsValue);

      // write reference values
      if (_write == kTRUE) {

         // Solutions of equation 2*[ln(LL(xMax)) - ln(LL(x))] = 1, where xMax is the point of maximum likelihood
         // For the special case of the Poisson distribution with N = 1, xMax = obsValue
         TString llRatioExpression = TString::Format("2*(x-%d*log(x)-%d+%d*log(%d))", fObsValue, fObsValue, fObsValue, fObsValue);
         // Special case fObsValue = 0 because log(0) not computable, the limit of n * log(n), n->0 must be taken
         if (fObsValue == 0) llRatioExpression = TString::Format("2*x");

         TF1 *llRatio = new TF1("llRatio", llRatioExpression, 1e-100, fObsValue); // lowerLimit < obsValue
         Double_t lowerLimit = llRatio->GetX(1);
         llRatio->SetRange(fObsValue, 1000); // upperLimit > obsValue
         Double_t upperLimit = llRatio->GetX(1);

         // Compare the limits obtained via ProfileLikelihoodCalculator with the likelihood ratio analytic computations
         regValue(lowerLimit, lowerLimitString);
         regValue(upperLimit, upperLimitString);

         // Cleanup branch objects
         delete llRatio;

         // compare with reference values
      } else {

         // Set a 68% confidence level for the interval
         const Double_t confidenceLevel = 2 * normal_cdf(1) - 1.0;

         // Create Poisson model and dataset
         RooWorkspace* w = new RooWorkspace("w", kTRUE);
         w->factory(TString::Format("Poisson::poiss(x[%d,0,1000], mean[0,1000])", fObsValue));
         RooDataSet *data = new RooDataSet("data", "data", *w->var("x"));
         data->add(*w->var("x"));

         // Calculate likelihood interval using the ProfileLikelihoodCalculator
         ProfileLikelihoodCalculator *plc = new ProfileLikelihoodCalculator(*data, *w->pdf("poiss"), *w->var("mean"));
         plc->SetConfidenceLevel(confidenceLevel);
         LikelihoodInterval *interval = plc->GetInterval();

         // Register externally computed limits in the reference file
         regValue(interval->LowerLimit(*w->var("mean")), lowerLimitString);
         regValue(interval->UpperLimit(*w->var("mean")), upperLimitString);

         // Cleanup branch objects
         delete plc;
         delete interval;
         delete data;
         delete w;
      }

      return kTRUE ;
   }
};


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
//    built in stressRooStats_models.cxx
//
// Input Parameters:
//    obsValueX -> observed value "x" when measuring sig + bkg1
//    obsValueY -> observed value "y" when measuring 2*sig*1.2^beta + bkg2
//    confidenceLevel -> Confidence Level of the interval we are calculating
//
// 04/2012 - Ioan Gabriel Bucur
//
///////////////////////////////////////////////////////////////////////////////

class TestProfileLikelihoodCalculator3 : public RooUnitTest {
private:
   Int_t fObsValueX;
   Int_t fObsValueY;
   Double_t fConfidenceLevel;

public:
   TestProfileLikelihoodCalculator3(
      TFile* refFile,
      Bool_t writeRef,
      Int_t verbose,
      Int_t obsValueX = 15,
      Int_t obsValueY = 30,
      Double_t confidenceLevel = 2 * normal_cdf(1) - 1
   ) :
      RooUnitTest("ProfileLikelihoodCalculator Interval - Poisson Product Model", refFile, writeRef, verbose),
      fObsValueX(obsValueX),
      fObsValueY(obsValueY),
      fConfidenceLevel(confidenceLevel)
   {};

   // Basic checks for the parameters passed to the test
   // In case of invalid parameters, a warning is printed and the test is skipped
   Bool_t isTestAvailable() {
      if (fObsValueX < 0 || fObsValueX > 40) {
         Warning("isTestAvailable", "Observed value X=s+b must be in the range [0,40]. Skipping test...");
         return kFALSE;
      }
      if (fObsValueY < 0 || fObsValueY > 120) {
         Warning("isTestAvailable", "Observed value Y=2*s*1.2^beta+b must be in the range [0,120]. Skipping test...");
         return kFALSE;
      }
      if (fConfidenceLevel <= 0.0 || fConfidenceLevel >= 1.0) {
         Warning("isTestAvailable", "Confidence level must be in the range (0,1). Skipping test...");
         return kFALSE;
      }
      return kTRUE;
   }

   Bool_t testCode() {

      // Create workspace and model
      RooWorkspace *w = new RooWorkspace("w", kTRUE);
      buildPoissonProductModel(w);
      ModelConfig *model = (ModelConfig *)w->obj("S+B");

      // add observed values to data set
      w->var("x")->setVal(fObsValueX);
      w->var("y")->setVal(fObsValueY);
      w->data("data")->add(*model->GetObservables());

      // build likelihood interval with ProfileLikelihoodCalculator
      ProfileLikelihoodCalculator *plc = new ProfileLikelihoodCalculator(*w->data("data"), *model);
      plc->SetConfidenceLevel(fConfidenceLevel);
      LikelihoodInterval *interval = plc->GetInterval();

      regValue(
         interval->LowerLimit(*w->var("sig")),
         TString::Format("tplc3_lower_limit_sig_%d_%d_%lf", fObsValueX, fObsValueY, fConfidenceLevel)
      );
      regValue(
         interval->UpperLimit(*w->var("sig")),
         TString::Format("tplc3_upper_limit_sig_%d_%d_%lf", fObsValueX, fObsValueY, fConfidenceLevel)
      );

      // cleanup
      delete interval;
      delete plc;
      delete w;

      return kTRUE ;
   }
};


///////////////////////////////////////////////////////////////////////////////
//
// PROFILE LIKELIHOOD CALCULATOR HYPOTHESIS TEST - ON / OFF MODEL
//
// Perform a hypothesis test using the ProfileLikelihoodCalculator on the
// on/off model. Reference values and test values are both computed with the
// ProfileLikelihoodCalculator. As such, this test can only confirm if the
// ProfileLikelihoodCalculator has the same behaviour accross different
// computing platforms or RooStats revisions.
//
// ModelConfig (explicit) : Poisson On / Off Model
//    built in stressRooStats_models.cxx
//
// For a detailed description of the on/off model, see the paper: "Evaluation
// of three methods for calculating statistical significance when incorporating
// a systematic uncertainty into a test of the background-only hypothesis for
// a Poisson process" by Robert D. Cousins, James T. Linnemann, Jordan Tucker
//
// 04/2012 - Ioan Gabriel Bucur
//
///////////////////////////////////////////////////////////////////////////////

class TestProfileLikelihoodCalculator4 : public RooUnitTest {
public:
   TestProfileLikelihoodCalculator4(
      TFile* refFile,
      Bool_t writeRef,
      Int_t verbose
   ) :
      RooUnitTest("ProfileLikelihoodCalculator Hypothesis Test", refFile, writeRef, verbose)
   {};

   // Override test value tolerance
   // A larger tolerance is needed since the values in the Cousins paper are given with 1e-2 precision
   Double_t vtol() {
      return 1e-2;
   }

   Bool_t testCode() {

      // For testing purposes, we consider four special cases for which the values are known from
      // the Cousins et al. paper mentioned above. The inputs for each of these cases are (using
      // the notations from the paper): n_on, n_off and Z_PL. We provide a certain fixed input set
      // for each case.
      const Int_t numberTestSets = 3;
      const Int_t numberOnEvents[numberTestSets] = {4, 50, 67};
      const Int_t numberOffEvents[numberTestSets] = {5, 55, 15};
      const Double_t tau[numberTestSets] = {5.0, 2.0, 0.5};
      const Double_t significance[numberTestSets] =  {1.95, 3.02, 3.04};

      for (Int_t i = 0; i < numberTestSets; i++) {

         TString stringSignificance = TString::Format("tplc4_significance_%d_%d_%lf", numberOnEvents[i], numberOffEvents[i], tau[i]);

         if (_write == kTRUE) {

            // register reference values from Cousins et al. paper
            regValue(significance[i], stringSignificance);

         } else {

            // build workspace and model
            RooWorkspace *w = new RooWorkspace("w", kTRUE);
            buildOnOffModel(w);
            ModelConfig *sbModel = (ModelConfig *)w->obj("S+B");
            ModelConfig *bModel = (ModelConfig *)w->obj("B");

            // add observable values to data set
            w->var("n_on")->setVal(numberOnEvents[i]);
            w->var("n_off")->setVal(numberOffEvents[i]);
            w->var("tau")->setVal(tau[i]);
            w->var("tau")->setConstant();
            w->data("data")->add(*sbModel->GetObservables());

            // set snapshots
            w->var("sig")->setVal(numberOnEvents[i] - numberOffEvents[i] / tau[i]);
            sbModel->SetSnapshot(*sbModel->GetParametersOfInterest());
            w->var("sig")->setVal(0);
            bModel->SetSnapshot(*bModel->GetParametersOfInterest());

            // get significance using the ProfileLikelihoodCalculator
            ProfileLikelihoodCalculator *plc = new ProfileLikelihoodCalculator(*w->data("data"), *sbModel);
            plc->SetNullParameters(*bModel->GetSnapshot());
            plc->SetAlternateParameters(*sbModel->GetSnapshot());
            regValue(plc->GetHypoTest()->Significance(), stringSignificance);

            // cleanup
            delete plc;
            delete w;
         }
      }

      return kTRUE ;
   }
};

//
// END OF PART ONE
//
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
//_____________________________________________________________________________





//_____________________________________________________________________________
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
//
// PART TWO:
//    BAYESIAN CALCULATOR UNIT TESTS
//

#include "RooStats/BayesianCalculator.h"
#include "RooCFunction1Binding.h" // for prior building purposes

///////////////////////////////////////////////////////////////////////////////
//
// BAYESIAN CENTRAL INTERVAL - SIMPLE MODEL
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
// Therefore, the reference interval limits will be computed analytically.
//
// ModelConfig (implicit) :
//    Observable -> x
//    Parameter of Interest -> mean
//
// Input Parameters:
//    obsValue -> observed value in experiment
//    confidenceLevel -> Confidence Level of the interval we are calculating
//
// 04/2012 - Ioan Gabriel Bucur
//
///////////////////////////////////////////////////////////////////////////////

class TestBayesianCalculator1 : public RooUnitTest {
private:
   Int_t fObsValue;
   Double_t fConfidenceLevel;
   static Double_t priorInv(Double_t mean) {
      return 1.0 / mean;
   }
   static Double_t priorInvSqrt(Double_t mean) {
      return 1.0 / sqrt(mean);
   }

public:
   TestBayesianCalculator1(
      TFile* refFile,
      Bool_t writeRef,
      Int_t verbose,
      Int_t obsValue = 3,
      Double_t confidenceLevel = 2 * normal_cdf(1) - 1
   ) :
      RooUnitTest("BayesianCalculator Central Interval - Poisson Simple Model", refFile, writeRef, verbose),
      fObsValue(obsValue),
      fConfidenceLevel(confidenceLevel)
   {};

   // Basic checks for the parameters passed to the test
   // In case of invalid parameters, a warning is printed and the test is skipped
   Bool_t isTestAvailable() {
      if (fObsValue < 0 || fObsValue > 100) {
         Warning("isTestAvailable", "Observed value must be in the range [0,100]. Skipping test...");
         return kFALSE;
      }
      if (fConfidenceLevel <= 0.0 || fConfidenceLevel >= 1.0) {
         Warning("isTestAvailable", "Confidence level must be in the range (0,1). Skipping test...");
         return kFALSE;
      }
      return kTRUE;
   }

   Bool_t testCode() {

      // Set the confidence level for a 68.3% CL central interval
      const Double_t gammaShape = 2; // shape of the gamma distribution prior (gamma = alpha)
      const Double_t gammaRate = 1; // rate = 1/scale of the gamma distribution prior (beta = 1/theta)
      const Int_t numberScans = 10000; // tested to be sufficient for the scan of the Bayesian posterior

      // names of tested variables must be the same in write / comparison modes
      const TString lowerLimitString = TString::Format("tbc1_lower_limit_unif_%d_%lf", fObsValue, fConfidenceLevel);
      const TString upperLimitString = TString::Format("tbc1_upper_limit_unif_%d_%lf", fObsValue, fConfidenceLevel);
      const TString lowerLimitInvString = TString::Format("tbc1_lower_limit_inv_%d_%lf", fObsValue, fConfidenceLevel);
      const TString upperLimitInvString = TString::Format("tbc1_upper_limit_inv_%d_%lf", fObsValue, fConfidenceLevel);
      const TString lowerLimitInvSqrtString = TString::Format("tbc1_lower_limit_inv_sqrt_%d_%lf", fObsValue, fConfidenceLevel);
      const TString upperLimitInvSqrtString = TString::Format("tbc1_upper_limit_inv_sqrt_%d_%lf", fObsValue, fConfidenceLevel);
      const TString lowerLimitGammaString = TString::Format("tbc1_lower_limit_gamma_%d_%lf", fObsValue, fConfidenceLevel);
      const TString upperLimitGammaString = TString::Format("tbc1_upper_limit_gamma_%d_%lf", fObsValue, fConfidenceLevel);

      if (_write == kTRUE) {

         Double_t lowerLimit = gamma_quantile((1.0 - fConfidenceLevel) / 2, fObsValue + 1, 1); // integrate to 16%
         Double_t upperLimit = gamma_quantile_c((1.0 - fConfidenceLevel) / 2, fObsValue + 1, 1); // integrate to 84%
         Double_t lowerLimitInv = gamma_quantile((1.0 - fConfidenceLevel) / 2, fObsValue, 1);
         Double_t upperLimitInv = gamma_quantile_c((1.0 - fConfidenceLevel) / 2, fObsValue, 1);
         Double_t lowerLimitInvSqrt = gamma_quantile((1.0 - fConfidenceLevel) / 2, fObsValue + 0.5, 1);
         Double_t upperLimitInvSqrt = gamma_quantile_c((1.0 - fConfidenceLevel) / 2, fObsValue + 0.5, 1);
         Double_t lowerLimitGamma = gamma_quantile((1.0 - fConfidenceLevel) / 2, fObsValue + gammaShape, 1.0 / (1 + gammaRate));
         Double_t upperLimitGamma = gamma_quantile_c((1.0 - fConfidenceLevel) / 2, fObsValue + gammaShape, 1.0 / (1 + gammaRate));

         // Compare the limits obtained via BayesianCalculator with quantile values
         regValue(lowerLimit, lowerLimitString);
         regValue(upperLimit, upperLimitString);
         regValue(lowerLimitInv, lowerLimitInvString);
         regValue(upperLimitInv, upperLimitInvString);
         regValue(lowerLimitInvSqrt, lowerLimitInvSqrtString);
         regValue(upperLimitInvSqrt, upperLimitInvSqrtString);
         regValue(lowerLimitGamma, lowerLimitGammaString);
         regValue(upperLimitGamma, upperLimitGammaString);

      } else {

         // Create Poisson model
         RooWorkspace* w = new RooWorkspace("w", kTRUE);
         w->factory("Poisson::poiss(x[0,100], mean[1e-6,100])");
         // TODO: see why it does not work so well for boundary observed values {0, 100}

         // create prior pdfs
         w->factory("Uniform::prior(mean)");
         w->import(*(new RooCFunction1PdfBinding<Double_t, Double_t>("priorInv", "priorInv", &priorInv, *w->var("mean"))));
         w->import(*(new RooCFunction1PdfBinding<Double_t, Double_t>("priorInvSqrt", "priorInvSqrt", priorInvSqrt, *w->var("mean"))));
         w->factory(TString::Format("Gamma::priorGamma(mean, %lf, %lf, 0)", gammaShape, gammaRate));

         // build argument sets and data set
         w->defineSet("obs", "x");
         w->defineSet("poi", "mean");
         w->var("x")->setVal(fObsValue);
         RooDataSet *data = new RooDataSet("data", "data", *w->set("obs"));
         data->add(*w->set("obs"));

         // NOTE: RooIntegrator1D is too slow and gives poor results
         RooAbsReal::defaultIntegratorConfig()->method1D().setLabel("RooAdaptiveGaussKronrodIntegrator1D");

         // Uniform prior on mean
         BayesianCalculator *bc = new BayesianCalculator(*data, *w->pdf("poiss"), *w->set("poi"), *w->pdf("prior"), NULL);
         bc->SetConfidenceLevel(fConfidenceLevel);
         bc->SetScanOfPosterior(numberScans);
         SimpleInterval *interval = bc->GetInterval();
         regValue(interval->LowerLimit(), lowerLimitString);
         regValue(interval->UpperLimit(), upperLimitString);

         delete bc;
         delete interval;

         // Inverse of mean prior
         bc = new BayesianCalculator(*data, *w->pdf("poiss"), *w->set("poi"), *w->pdf("priorInv"), NULL);
         bc->SetConfidenceLevel(fConfidenceLevel);
         bc->SetScanOfPosterior(numberScans);
         interval = bc->GetInterval();
         regValue(interval->LowerLimit(), lowerLimitInvString);
         regValue(interval->UpperLimit(), upperLimitInvString);

         delete bc;
         delete interval;

         // Square root of inverse of mean prior
         bc = new BayesianCalculator(*data, *w->pdf("poiss"), *w->set("poi"), *w->pdf("priorInvSqrt"), NULL);
         bc->SetConfidenceLevel(fConfidenceLevel);
         bc->SetScanOfPosterior(numberScans);
         interval = bc->GetInterval();
         regValue(interval->LowerLimit(), lowerLimitInvSqrtString);
         regValue(interval->UpperLimit(), upperLimitInvSqrtString);

         delete bc;
         delete interval;

         // Gamma distribution prior
         bc = new BayesianCalculator(*data, *w->pdf("poiss"), *w->set("poi"), *w->pdf("priorGamma"), NULL);
         bc->SetConfidenceLevel(fConfidenceLevel);
         bc->SetScanOfPosterior(numberScans);
         interval = bc->GetInterval();
         regValue(interval->LowerLimit(), lowerLimitGammaString);
         regValue(interval->UpperLimit(), upperLimitGammaString);

         // Cleanup branch objects
         delete bc;
         delete interval;
         delete data;
         delete w;
      }

      return kTRUE ;
   }

};


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
// ModelConfig (implicit) :
//    Observable -> x
//    Parameter of Interest -> mean
//
// 04/2012 - Ioan Gabriel Bucur
//
///////////////////////////////////////////////////////////////////////////////

class TestBayesianCalculator2 : public RooUnitTest {
public:
   TestBayesianCalculator2(
      TFile* refFile,
      Bool_t writeRef,
      Int_t verbose
   ) :
      RooUnitTest("BayesianCalculator Shortest Interval - Poisson Simple Model", refFile, writeRef, verbose)
   {};

   // the references values in the paper have a precision of only two decimal points
   // in such a situation, it is natural that we increase the value tolerance
   Double_t vtol() {
      return 1e-2;
   }

   Bool_t testCode() {

      // Put the confidence level so that we obtain a 68% confidence interval
      const Double_t confidenceLevel = 2 * normal_cdf(1) - 1;
      const Int_t obsValue = 3; // observed experiment value
      const Int_t numberScans = 10000; // sufficient number of scans

      // names of tested variables must be the same in write / comparison modes
      const TString lowerLimitString = "tbc2_lower_limit_unif";
      const TString upperLimitString = "tbc2_upper_limit_unif";
      const TString lowerLimitInvString = "tbc2_lower_limit_inv";
      const TString upperLimitInvString = "tbc2_upper_limit_inv";

      if (_write == kTRUE) {

         // Compare the limits obtained via BayesianCalculator with given reference values
         regValue(1.55, lowerLimitString);
         regValue(5.15, upperLimitString);
         regValue(0.86, lowerLimitInvString);
         regValue(3.85, upperLimitInvString);

      } else {

         // Create Poisson model
         RooWorkspace* w = new RooWorkspace("w", kTRUE);
         w->factory("Poisson::poiss(x[0,100], mean[1e-6,100])");
         w->factory("Uniform::prior(mean)");
         w->factory("EXPR::priorInv('1/mean', mean)");

         // build argument sets and data set
         w->defineSet("poi", "mean");
         w->defineSet("obs", "x");
         w->var("x")->setVal(obsValue);
         RooDataSet *data = new RooDataSet("data", "data", *w->set("obs"));
         data->add(*w->set("obs"));

         // NOTE: RooIntegrator1D is too slow and gives poor results
         RooAbsReal::defaultIntegratorConfig()->method1D().setLabel("RooAdaptiveGaussKronrodIntegrator1D");

         // Uniform prior on mean
         BayesianCalculator *bc = new BayesianCalculator(*data, *w->pdf("poiss"), *w->set("poi"), *w->pdf("prior"), NULL);
         bc->SetConfidenceLevel(confidenceLevel);
         bc->SetShortestInterval();
         bc->SetScanOfPosterior(numberScans);
         SimpleInterval *interval = bc->GetInterval();
         regValue(interval->LowerLimit(), lowerLimitString);
         regValue(interval->UpperLimit(), upperLimitString);

         delete bc;
         delete interval;

         // Inverse of mean prior
         bc = new BayesianCalculator(*data, *w->pdf("poiss"), *w->set("poi"), *w->pdf("priorInv"), NULL);
         bc->SetConfidenceLevel(confidenceLevel);
         bc->SetShortestInterval();
         bc->SetScanOfPosterior(numberScans);
         interval = bc->GetInterval();
         regValue(interval->LowerLimit(), lowerLimitInvString);
         regValue(interval->UpperLimit(), upperLimitInvString);

         // Cleanup branch objects
         delete bc;
         delete interval;
         delete data;
         delete w;
      }

      return kTRUE ;
   }

};


///////////////////////////////////////////////////////////////////////////////
//
// BAYESIAN CENTRAL INTERVAL - POISSON PRODUCT MODEL
//
// Test the validity of the central interval computed by the BayesianCalculator
// on a complex Poisson model distribution. Reference values and test values
// are both computed with the BayesianCalculator. As such, this test can only
// confirm if the BayesianCalculator has the same behaviour across different
// computing platforms or RooStats revisions. A uniform prior PDF is used for the
// parameter of interest ("sig").
//
// ModelConfig (explicit) : Poisson Product Model
//    built in stressRooStats_models.cxx
//
// Input Parameters:
//    obsValueX -> observed value "x" when measuring sig + bkg1
//    obsValueY -> observed value "y" when measuring 2*sig*1.2^beta + bkg2
//    confidenceLevel -> Confidence Level of the interval we are calculating
//
// 04/2012 - Ioan Gabriel Bucur
//
///////////////////////////////////////////////////////////////////////////////

class TestBayesianCalculator3 : public RooUnitTest {
private:
   Int_t fObsValueX;
   Int_t fObsValueY;
   Double_t fConfidenceLevel;

public:
   TestBayesianCalculator3(
      TFile* refFile,
      Bool_t writeRef,
      Int_t verbose,
      Int_t obsValueX = 15,
      Int_t obsValueY = 30,
      Double_t confidenceLevel = 2 * normal_cdf(1) - 1
   ) :
      RooUnitTest("BayesianCalculator Central Interval - Poisson Product Model", refFile, writeRef, verbose),
      fObsValueX(obsValueX),
      fObsValueY(obsValueY),
      fConfidenceLevel(confidenceLevel)
   {};

   // Basic checks for the parameters passed to the test
   // In case of invalid parameters, a warning is printed and the test is skipped
   Bool_t isTestAvailable() {
      if (fObsValueX < 0 || fObsValueX > 40) {
         Warning("isTestAvailable", "Observed value X=s+b must be in the range [0,40]. Skipping test...");
         return kFALSE;
      }
      if (fObsValueY < 0 || fObsValueY > 120) {
         Warning("isTestAvailable", "Observed value Y=2*s*1.2^beta+b must be in the range [0,120]. Skipping test...");
         return kFALSE;
      }
      if (fConfidenceLevel <= 0.0 || fConfidenceLevel >= 1.0) {
         Warning("isTestAvailable", "Confidence level must be in the range (0,1). Skipping test...");
         return kFALSE;
      }
      return kTRUE;
   }

   Bool_t testCode() {

      const Int_t numberScans = 10; // sufficient number of scans

      // Create workspace and model
      RooWorkspace *w = new RooWorkspace("w", kTRUE);
      buildPoissonProductModel(w);
      ModelConfig *model = (ModelConfig *)w->obj("S+B");

      // add observed values to data set
      w->var("x")->setVal(fObsValueX);
      w->var("y")->setVal(fObsValueY);
      w->data("data")->add(*model->GetObservables());

      // NOTE: Roo1DIntegrator is too slow and gives poor results
      RooAbsReal::defaultIntegratorConfig()->method1D().setLabel("RooAdaptiveGaussKronrodIntegrator1D");

      // Create BayesianCalculator and
      BayesianCalculator *bc = new BayesianCalculator(*w->data("data"), *model);
      bc->SetConfidenceLevel(fConfidenceLevel);
      bc->SetScanOfPosterior(numberScans);

      // Obtain confidence interval by scanning the posterior function in the given number of points
      SimpleInterval *interval = bc->GetInterval();
      regValue(
         interval->LowerLimit(),
         TString::Format("tbc3_lower_limit_sig_%d_%d_%lf", fObsValueX, fObsValueY, fConfidenceLevel)
      );
      regValue(
         interval->UpperLimit(),
         TString::Format("tbc3_upper_limit_sig_%d_%d_%lf", fObsValueX, fObsValueY, fConfidenceLevel)
      );

      // Cleanup
      delete bc;
      delete interval;
      delete w;

      return kTRUE ;
   }
};


//
// END OF PART TWO
//
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
//_____________________________________________________________________________





//_____________________________________________________________________________
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
//
// PART THREE:
//    MARKOV CHAIN MONTE CARLO CALCULATOR UNIT TESTS
//

#include "RooStats/MCMCCalculator.h"
#include "RooStats/SequentialProposal.h"

///////////////////////////////////////////////////////////////////////////////
//
// MCMC INTERVAL CALCULATOR - POISSON PRODUCT MODEL
//
// Test the validity of the confidence interval computed by the MCMCCalculator
// on a complex Poisson model distribution. Reference values and test values
// are both computed with the MCMCCalculator. As such, this test can only
// confirm if the BayesianCalculator has the same behaviour across different
// computing platforms or RooStats revisions.
//
// ModelConfig (explicit) : Poisson Product Model
//    built in stressRooStats_models.cxx
//
// Input Parameters:
//    obsValueX -> observed value "x" when measuring sig + bkg1
//    obsValueY -> observed value "y" when measuring 2*sig*1.2^beta + bkg2
//    confidenceLevel -> Confidence Level of the interval we are calculating
//
// 04/2012 - Ioan Gabriel Bucur
//
///////////////////////////////////////////////////////////////////////////////

class TestMCMCCalculator : public RooUnitTest {
private:
   Int_t fObsValueX;
   Int_t fObsValueY;
   Double_t fConfidenceLevel;

public:
   TestMCMCCalculator(
      TFile* refFile,
      Bool_t writeRef,
      Int_t verbose,
      Int_t obsValueX = 15,
      Int_t obsValueY = 30,
      Double_t confidenceLevel = 2 * normal_cdf(1) - 1
   ) :
      RooUnitTest("MCMCCalculator Interval - Poisson Product Model", refFile, writeRef, verbose),
      fObsValueX(obsValueX),
      fObsValueY(obsValueY),
      fConfidenceLevel(confidenceLevel)
   {};

   // Basic checks for the parameters passed to the test
   // In case of invalid parameters, a warning is printed and the test is skipped
   Bool_t isTestAvailable() {
      if (fObsValueX < 0 || fObsValueX > 40) {
         Warning("isTestAvailable", "Observed value X=s+b must be in the range [0,40]. Skipping test...");
         return kFALSE;
      }
      if (fObsValueY < 0 || fObsValueY > 120) {
         Warning("isTestAvailable", "Observed value Y=2*s*1.2^beta+b must be in the range [0,120]. Skipping test...");
         return kFALSE;
      }
      if (fConfidenceLevel <= 0.0 || fConfidenceLevel >= 1.0) {
         Warning("isTestAvailable", "Confidence level must be in the range (0,1). Skipping test...");
         return kFALSE;
      }
      return kTRUE;
   }

   Bool_t testCode() {

      // Create workspace and model
      RooWorkspace *w = new RooWorkspace("w", kTRUE);
      buildPoissonProductModel(w);
      ModelConfig *model = (ModelConfig *)w->obj("S+B");

      // add observed values to data set
      w->var("x")->setVal(fObsValueX);
      w->var("y")->setVal(fObsValueY);
      w->data("data")->add(*model->GetObservables());

      // NOTE: Roo1DIntegrator is too slow and gives poor results
      RooAbsReal::defaultIntegratorConfig()->method1D().setLabel("RooAdaptiveGaussKronrodIntegrator1D");

      // create and configure MCMC calculator
      SequentialProposal *sp = new SequentialProposal(0.1);
      MCMCCalculator *mcmcc = new MCMCCalculator(*w->data("data"), *model);
      mcmcc->SetProposalFunction(*sp);
      mcmcc->SetNumIters(100000); // Metropolis-Hastings algorithm iterations
      mcmcc->SetNumBurnInSteps(50); // first 50 steps to be ignored as burn-in
      mcmcc->SetConfidenceLevel(fConfidenceLevel);

      // calculate the confidence interval
      MCMCInterval *interval = mcmcc->GetInterval();
      regValue(
         interval->LowerLimit(*w->var("sig")),
         TString::Format("mcmcc_lower_limit_sig_%d_%d_%lf", fObsValueX, fObsValueY, fConfidenceLevel)
      );
      regValue(
         interval->UpperLimit(*w->var("sig")),
         TString::Format("mcmcc_upper_limit_sig_%d_%d_%lf", fObsValueX, fObsValueY, fConfidenceLevel)
      );

      // cleanup
      delete interval;
      delete mcmcc;
      delete sp;
      delete w;

      return kTRUE ;
   }
};


//
// END OF PART THREE
//
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
//_____________________________________________________________________________





//_____________________________________________________________________________
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
//
// PART FOUR:
//    HYPOTHESIS TEST CALCULATOR UNIT TESTS
//

// Hypo Test Calculators
#include "RooStats/HypoTestCalculatorGeneric.h"
#include "RooStats/FrequentistCalculator.h"
#include "RooStats/HybridCalculator.h"
#include "RooStats/AsymptoticCalculator.h"
// Test Statistics
#include "RooStats/ProfileLikelihoodTestStat.h"
#include "RooStats/RatioOfProfiledLikelihoodsTestStat.h"
#include "RooStats/SimpleLikelihoodRatioTestStat.h"
#include "RooStats/ProfileLikelihoodCalculator.h"
#include "RooStats/MaxLikelihoodEstimateTestStat.h"
#include "RooStats/NumEventsTestStat.h"

/////////////////////////////////////////////////////////////////////////
//
// ZBI - ON / OFF MODEL
//
// Evaluate the functionality of the top level functions in RooStats
// called NumberCountingUtils::BinomialWithTauObsZ. This function
// computes the significance of a hypothesis test via a frequentist
// solution. This significance, called ZBi, is detailed in the article
// "Evaluation of three methods for calculating statistical significance
// when incorporating a systematic uncertainty into a test of the
// background-only hypothesis for a Poisson process" by Robert D. Cousins,
// James T. Linnemann, Jordan Tucker. The reference values are taken
// from the paper, as well as the On / Off model on which the test is
// evaluated.
//
// ModelConfig (implicit) : Poisson On / Off Model
//    built in stressRooStats_models.cxx
//    implicit in NumberCountingUtils::BinomialWithTauObsZ
//
// 05/2012 - Wouter Verkerke, Lorenzo Moneta, Ioan Gabriel Bucur
//
/////////////////////////////////////////////////////////////////////////


class TestZBi : public RooUnitTest {
public:

   TestZBi(
      TFile* refFile,
      Bool_t writeRef,
      Int_t verbose
   ) :
      RooUnitTest("ZBi Significance - On / Off Model", refFile, writeRef, verbose)
   {};

   // Override test value tolerance
   // A larger tolerance is needed since the values in the Cousins paper are given with 1e-2 precision
   Double_t vtol() {
      return 1e-2;
   }

   Bool_t testCode() {

      // For testing purposes, we consider four special cases for which the values are known from
      // the Cousins et al. paper mentioned above. The inputs for each of these cases are (using
      // the notations from the paper): n_on, n_off and Z_PL. We provide a certain fixed input set
      // for each case.
      const Int_t numberTestSets = 4;
      const Int_t numberOnEvents[numberTestSets] = {4, 50, 67, 200};
      const Int_t numberOffEvents[numberTestSets] = {5, 55, 15, 10};
      const Double_t tau[numberTestSets] = {5.0, 2.0, 0.5, 0.1};
      const Double_t significance[numberTestSets] =  {1.66, 2.93, 2.89, 2.2};

      for (Int_t i = 0; i < numberTestSets; i++) {

         TString stringSignificance = TString::Format("tzbi_significance_%d_%d_%lf", numberOnEvents[i], numberOffEvents[i], tau[i]);

         if (_write == kTRUE) {

            // register reference values from Cousins et al. paper
            regValue(significance[i], stringSignificance);

         } else {

            // call top level function
            regValue(
               NumberCountingUtils::BinomialWithTauObsZ(numberOnEvents[i], numberOffEvents[i], tau[i]),
               stringSignificance
            );

         }
      }

      return kTRUE ;
   }
};


///////////////////////////////////////////////////////////////////////////////
//
// ASYMPTOTIC CALCULATOR VS PROFILE LIKELIHOOD CALCULATOR HYPOTHESIS TEST
//
// This test evaluates the functionality of the AsymptoticCalculator by
// comparing the significance given from a hypothesis test on the on/off model
// with the significance given by the ProfileLikelihoodCalculator. The validity
// of the PLC hypothesis test is evaluated in TestProfileLikelihoodCalculator4.
// If working properly, the two methods should yield identical results.
//
// ModelConfig (explicit) : Poisson On / Off Model
//    built in stressRooStats_models.cxx
//
// Input Parameters:
//    obsValueOn -> observed value "n_on" of sig + bkg
//    obsValueOff -> observed value "n_off" of tau * bkg
//    tau -> parameter of the model (constant with regard to integration)
//
// 05/2012 - Ioan Gabriel Bucur
//
///////////////////////////////////////////////////////////////////////////////

class TestHypoTestCalculator1 : public RooUnitTest {
private:
   Int_t fObsValueOn;
   Int_t fObsValueOff;
   Double_t fTau;

public:
   TestHypoTestCalculator1(
      TFile* refFile,
      Bool_t writeRef,
      Int_t verbose,
      Int_t obsValueOn = 150,
      Int_t obsValueOff = 100,
      Double_t tau = 1.0
   ) :
      RooUnitTest("AsymptoticCalculator vs ProfileLikelihoodCalculator Significance - On / Off Model", refFile, writeRef, verbose),
      fObsValueOn(obsValueOn),
      fObsValueOff(obsValueOff),
      fTau(tau)
   {};

   Bool_t testCode() {

      // names of tested variables must be the same in write / comparison modes
      TString significanceString = TString::Format("thtc1_significance_%d_%d_%lf", fObsValueOn, fObsValueOff, fTau);

      // build workspace and model
      RooWorkspace* w = new RooWorkspace("w", kTRUE);
      buildOnOffModel(w);
      ModelConfig *sbModel = (ModelConfig *)w->obj("S+B");
      ModelConfig *bModel = (ModelConfig *)w->obj("B");

      // add observable values to data set and fix other parameters
      w->var("n_on")->setVal(fObsValueOn);
      w->var("n_off")->setVal(fObsValueOff);
      w->var("tau")->setVal(fTau);
      w->var("tau")->setConstant();
      w->data("data")->add(*sbModel->GetObservables());
      w->var("bkg")->setVal(fObsValueOff / fTau);

      // Make snapshots
      w->var("sig")->setVal(fObsValueOn - fObsValueOff / fTau);
      sbModel->SetSnapshot(*sbModel->GetParametersOfInterest());
      w->var("sig")->setVal(0.0);
      bModel->SetSnapshot(*bModel->GetParametersOfInterest());

      // Do hypothesis test with ProfileLikelihoodCalculator
      if (_write == kTRUE) {

         ProfileLikelihoodCalculator *plc = new ProfileLikelihoodCalculator(*w->data("data"), *sbModel);
         plc->SetNullParameters(*bModel->GetSnapshot());
         regValue(plc->GetHypoTest()->Significance(), significanceString);

         // cleanup branch
         delete plc;

      } else { // Do hypothesis test with AsymptoticCalculator

         AsymptoticCalculator::SetPrintLevel(0); // disable superfluous messaging
         AsymptoticCalculator *atc = new AsymptoticCalculator(*w->data("data"), *sbModel, *bModel);
         atc->SetOneSidedDiscovery(kTRUE);
         regValue(atc->GetHypoTest()->Significance(), significanceString);

         // cleanup branch
         delete atc;
      }

      // cleanup
      delete w;

      return kTRUE ;
   }
} ;


//
// END OF PART FOUR
//
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
//_____________________________________________________________________________





//_____________________________________________________________________________
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
//
// PART FIVE:
//    HYPOTHESIS TEST INVERTER UNIT TESTS
//

#include "RooStats/HypoTestInverter.h"
#include "RooStats/HypoTestInverterResult.h"
#include "RooStats/ToyMCSampler.h"
#include "RooStats/HypoTestInverterPlot.h"
#include "RooStats/SamplingDistPlot.h"

///////////////////////////////////////////////////////////////////////////////
//
// HYPOTESTINVERTER INTERVAL - POISSON PRODUCT MODEL
//
// Test the validity of the confidence interval computed by the HypoTestInverter
// on a complex Poisson model distribution. Reference values and test values
// are both computed with the MCMCCalculator. As such, this test can only
// confirm if the BayesianCalculator has the same behaviour across different
// computing platforms or RooStats revisions.
//
// ModelConfig (explicit) : Poisson Product Model
//    built in stressRooStats_models.cxx
//
// Input Parameters:
//    obsValueX -> observed value "x" when measuring sig + bkg1
//    obsValueY -> observed value "y" when measuring 2*sig*1.2^beta + bkg2
//    confidenceLevel -> Confidence Level of the interval we are calculating
//
// 04/2012 - Ioan Gabriel Bucur
//
///////////////////////////////////////////////////////////////////////////////


class TestHypoTestInverter1 : public RooUnitTest {
private:
   HypoTestInverter::ECalculatorType fCalculatorType;
   ETestStatType fTestStatType;
   Int_t fObsValueX;
   Int_t fObsValueY;
   Double_t fConfidenceLevel;

public:
   TestHypoTestInverter1(
      TFile* refFile,
      Bool_t writeRef,
      Int_t verbose,
      HypoTestInverter::ECalculatorType calculatorType,
      ETestStatType testStatType,
      Int_t obsValueX = 15,
      Int_t obsValueY = 30,
      Double_t confidenceLevel = 2 * normal_cdf(1) - 1
   ) :
      RooUnitTest(TString::Format("HypoTestInverter Interval - Poisson Product Model - %s - %s",
                                  kECalculatorTypeString[calculatorType], kETestStatTypeString[testStatType]), refFile, writeRef, verbose),
      fCalculatorType(calculatorType),
      fTestStatType(testStatType),
      fObsValueX(obsValueX),
      fObsValueY(obsValueY),
      fConfidenceLevel(confidenceLevel)
   {};

   // Basic checks for the parameters passed to the test
   // In case of invalid parameters, a warning is printed and the test is skipped
   Bool_t isTestAvailable() {
      if (fObsValueX < 0 || fObsValueX > 40) {
         Warning("isTestAvailable", "Observed value X=s+b must be in the range [0,40]. Skipping test...");
         return kFALSE;
      }
      if (fObsValueY < 0 || fObsValueY > 120) {
         Warning("isTestAvailable", "Observed value Y=2*s*1.2^beta+b must be in the range [0,120]. Skipping test...");
         return kFALSE;
      }
      if (fConfidenceLevel <= 0.0 || fConfidenceLevel >= 1.0) {
         Warning("isTestAvailable", "Confidence level must be in the range (0,1). Skipping test...");
         return kFALSE;
      }
      return kTRUE;
   }

   Bool_t testCode() {

      // Create workspace and model
      RooWorkspace *w = new RooWorkspace("w", kTRUE);
      buildPoissonProductModel(w);
      ModelConfig *sbModel = (ModelConfig *)w->obj("S+B");
      ModelConfig *bModel = (ModelConfig *)w->obj("B");

      // add observed values to data set
      w->var("x")->setVal(fObsValueX);
      w->var("y")->setVal(fObsValueY);
      w->data("data")->add(*sbModel->GetObservables());

      // set snapshots
      w->var("sig")->setVal(fObsValueX - w->var("bkg1")->getValV());
      sbModel->SetSnapshot(*sbModel->GetParametersOfInterest());
      w->var("sig")->setVal(0);
      bModel->SetSnapshot(*bModel->GetParametersOfInterest());

      //TODO: check how to eliminate this code, maybe 0 should be default print level for AsymptoticCalculator
      if (fCalculatorType == HypoTestInverter::kAsymptotic) {
         AsymptoticCalculator::SetPrintLevel(0); // print only minimal output
      }

      // configure HypoTestInverter
      HypoTestInverter *hti =
         new HypoTestInverter(*w->data("data"), *sbModel, *bModel, NULL, fCalculatorType, 1.0 - fConfidenceLevel);
      hti->SetTestStatistic(*buildTestStatistic(fTestStatType, *sbModel, *bModel));
      hti->SetFixedScan(10, w->var("sig")->getMin(), w->var("sig")->getMax()); // significant speedup

      //TODO: check how to eliminate this code, calculator should autoconfigure itself
      if (fCalculatorType == HypoTestInverter::kHybrid) {
         // force prior nuisance pdf and set toys for speedup
         HybridCalculator *hc = (HybridCalculator *)hti->GetHypoTestCalculator();
         hc->ForcePriorNuisanceNull(*MakeNuisancePdf(*sbModel, "nuis_prior_null"));
         hc->ForcePriorNuisanceAlt(*MakeNuisancePdf(*bModel, "nuis_prior_alt"));
         hc->SetToys(100, 1);
      }

      //TODO: check how to eliminate this code
      if (fCalculatorType == HypoTestInverter::kFrequentist) {
         // set toys for speedup
         FrequentistCalculator *fc = (FrequentistCalculator *)hti->GetHypoTestCalculator();
         fc->SetToys(100, 1);
      }

      // ToyMCSampler configuration
      ToyMCSampler *tmcs = (ToyMCSampler *)hti->GetHypoTestCalculator()->GetTestStatSampler();
      tmcs->SetNEventsPerToy(1); // needed because we don't have an extended pdf
      tmcs->SetAlwaysUseMultiGen(kTRUE); // speedup

      HypoTestInverterResult *interval = hti->GetInterval();
      regValue(interval->LowerLimit(), TString::Format("hti1_lower_limit_sig1_calc_%s_%s_%d_%d_%lf",
                                                       kECalculatorTypeString[fCalculatorType],
                                                       kETestStatTypeString[fTestStatType],
                                                       fObsValueX, fObsValueY, fConfidenceLevel));
      regValue(interval->UpperLimit(), TString::Format("hti1_upper_limit_sig1_calc_%s_%s_%d_%d_%lf",
                                                       kECalculatorTypeString[fCalculatorType],
                                                       kETestStatTypeString[fTestStatType],
                                                       fObsValueX, fObsValueY, fConfidenceLevel));

      if (_verb >= 1) {
         HypoTestInverterPlot *plot = new HypoTestInverterPlot("hti1_scan", "Two-Sided Scan", interval);
         TCanvas *c1 = new TCanvas("hti1_scan", "HTI Scan");
         c1->SetLogy(false);
         plot->Draw("");
         c1->SaveAs(TString::Format("hti1_scan_%s_%s_%d_%d_%lf.pdf",
                                    kECalculatorTypeString[fCalculatorType],
                                    kETestStatTypeString[fTestStatType],
                                    fObsValueX, fObsValueY, fConfidenceLevel));

         if (_verb == 2) {
            const int n = interval->ArraySize();
            if (n > 0 && interval->GetResult(0)->GetNullDistribution()) {
               TCanvas *c2 = new TCanvas("HTI_TestStatistic_Distribution", "HTI Test Statistic Distributions", 2);
               if (n > 1) {
                  int ny = TMath::CeilNint(sqrt(n));
                  int nx = TMath::CeilNint(double(n) / ny);
                  c2->Divide(nx, ny);
               }
               for (int i = 0; i < n; i++) {
                  if (n > 1) c2->cd(i + 1);
                  SamplingDistPlot *pl = plot->MakeTestStatPlot(i);
                  if (pl == NULL) return kTRUE;
                  pl->SetLogYaxis(kTRUE);
                  pl->Draw();
               }
               c2->SaveAs(TString::Format("hti1_teststat_distrib_%s_%s_%d_%d_%lf.pdf",
                                          kECalculatorTypeString[fCalculatorType],
                                          kETestStatTypeString[fTestStatType],
                                          fObsValueX, fObsValueY, fConfidenceLevel));
            }
         }
      }

      // cleanup
      delete interval;
      delete hti;
      delete w;

      return kTRUE ;
   }
};


//
// END OF PART FIVE
//
///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
//_____________________________________________________________________________










// Other tests currently not included in any suite


class TestHypoTestCalculator : public RooUnitTest {
public:
   TestHypoTestCalculator(TFile* refFile, Bool_t writeRef, Int_t verbose) : RooUnitTest("HypoTestCalculator - On / Off Problem", refFile, writeRef, verbose) {};

   Bool_t testCode() {

      const Int_t xValue = 150;
      const Int_t yValue = 100;
      const Double_t tauValue = 1.0;

      if (_write == kTRUE) {

         // register analytical Z_Bi value
         Double_t Z_Bi = NumberCountingUtils::BinomialWithTauObsZ(xValue, yValue, tauValue);
         regValue(Z_Bi, "thtc_significance_hybrid");

      } else {

         // Make model for prototype on/off problem
         // Pois(x | s+b) * Pois(y | tau b )
         RooWorkspace* w = new RooWorkspace("w", kTRUE);
         w->factory(TString::Format("Poisson::on_pdf(x[%d,0,500],sum::splusb(sig[0,0,100],bkg[100,0,300]))", xValue));
         w->factory(TString::Format("Poisson::off_pdf(y[%d,0,500],prod::taub(tau[%lf],bkg))", yValue, tauValue));
         w->factory("PROD::prod_pdf(on_pdf, off_pdf)");

         w->var("x")->setVal(xValue);
         w->var("y")->setVal(yValue);
         w->var("y")->setConstant();
         w->var("tau")->setVal(tauValue);

         // construct the Bayesian-averaged model (eg. a projection pdf)
         // p'(x|s) = \int db p(x|s+b) * [ p(y|b) * prior(b) ]
         w->factory("Uniform::prior(bkg)");
         w->factory("PROJ::averagedModel(PROD::foo(on_pdf|bkg,off_pdf,prior),bkg)") ;

         // define sets of variables obs={x} and poi={sig}
         // x is the only observable in the main measurement and y is treated as a separate measurement,
         // which is used to produce the prior that will be used in the calculation to randomize the nuisance parameters
         w->defineSet("obs", "x");
         w->defineSet("poi", "sig");

         // Add observable value to a data set
         RooDataSet *data = new RooDataSet("data", "data", *w->set("obs"));
         data->add(*w->set("obs"));


         // Build S+B and B models
         ModelConfig *sbModel = new ModelConfig("SB_ModelConfig", w);
         sbModel->SetPdf(*w->pdf("prod_pdf"));
         sbModel->SetObservables(*w->set("obs"));
         sbModel->SetParametersOfInterest(*w->set("poi"));
         w->var("sig")->setVal(xValue - yValue / tauValue); // important !
         sbModel->SetSnapshot(*w->set("poi"));

         ModelConfig *bModel = new ModelConfig("B_ModelConfig", w);
         bModel->SetPdf(*w->pdf("prod_pdf"));
         bModel->SetObservables(*w->set("obs"));
         bModel->SetParametersOfInterest(*w->set("poi"));
         w->var("sig")->setVal(0.0); // important !
         bModel->SetSnapshot(*w->set("poi"));

         // alternate priors
         w->factory("Gaussian::gauss_prior(bkg, y, expr::sqrty('sqrt(y)', y))");
         w->factory("Lognormal::lognorm_prior(bkg, y, expr::kappa('1+1./sqrt(y)',y))");

         // build test statistic
         SimpleLikelihoodRatioTestStat *slrts =  new SimpleLikelihoodRatioTestStat(*bModel->GetPdf(), *sbModel->GetPdf());
         slrts->SetNullParameters(*bModel->GetSnapshot());
         slrts->SetAltParameters(*sbModel->GetSnapshot());
         slrts->SetAlwaysReuseNLL(kTRUE);

         RatioOfProfiledLikelihoodsTestStat *roplts = new RatioOfProfiledLikelihoodsTestStat(*bModel->GetPdf(), *sbModel->GetPdf());
         roplts->SetAlwaysReuseNLL(kTRUE);

         ProfileLikelihoodTestStat *pllts = new ProfileLikelihoodTestStat(*bModel->GetPdf());
         pllts->SetAlwaysReuseNLL(kTRUE);
         pllts->SetOneSidedDiscovery(kTRUE);

         MaxLikelihoodEstimateTestStat *mlets =
            new MaxLikelihoodEstimateTestStat(*sbModel->GetPdf(), *((RooRealVar *)sbModel->GetParametersOfInterest()->first()));

         NumEventsTestStat *nevts = new NumEventsTestStat(*sbModel->GetPdf());




         HybridCalculator *htc = new HybridCalculator(*data, *sbModel, *bModel);
         ToyMCSampler *tmcs = (ToyMCSampler *)htc->GetTestStatSampler();
         tmcs->SetNEventsPerToy(1);
         htc->SetToys(5000, 1000);
         htc->ForcePriorNuisanceAlt(*w->pdf("off_pdf"));
         htc->ForcePriorNuisanceNull(*w->pdf("off_pdf"));

         tmcs->SetTestStatistic(pllts);
         HypoTestResult *htr = htc->GetHypoTest();
         htr->Print();
         cout << "PLLTS " << htr->Significance() << endl;
         tmcs->SetTestStatistic(mlets);
         htr = htc->GetHypoTest();
         htr->Print();
         cout << "MLETS " << htr->Significance() << endl;
         tmcs->SetTestStatistic(nevts);
         htr = htc->GetHypoTest();
         htr->Print();
         cout << "NEVTS " << htr->Significance() << endl;
         tmcs->SetTestStatistic(slrts);
         htr = htc->GetHypoTest();
         htr->Print();
         cout << "SLRTS " << htr->Significance() << endl;
         tmcs->SetTestStatistic(roplts);
         htr = htc->GetHypoTest();
         htr->Print();
         cout << "ROPLTS " << htr->Significance() << endl;


         regValue(htr->Significance(), "thtc_significance_hybrid");

         delete htc;
         delete htr;
         delete w;
         delete data;
      }

      return kTRUE ;
   }
} ;


#include "RooPoisson.h"

class TestHypoTestCalculator2 : public RooUnitTest {
public:
   TestHypoTestCalculator2(
      TFile* refFile,
      Bool_t writeRef,
      Int_t verbose
   ) :
      RooUnitTest("HypoTestCalculator Frequentist - Simultaneous Pdf", refFile, writeRef, verbose)
   {};

   Bool_t testCode() {

      const Int_t fObsValueX = 20;
      const Int_t fObsValueY = 30;
      const Double_t fTau = 1.0;

      if (_write == kTRUE) {

         // register analytical Z_Bi value
         Double_t Z_Bi = NumberCountingUtils::BinomialWithTauObsZ(fObsValueX, fObsValueY, fTau);
         regValue(Z_Bi, "thtc_significance_frequentist");

      } else {

         // Build workspace and models
         RooWorkspace* w = new RooWorkspace("w", kTRUE);
         buildPoissonProductModel(w);
         ModelConfig *sbModel = (ModelConfig *)w->obj("S+B");
         ModelConfig *bModel = (ModelConfig *)w->obj("B");

         sbModel->SetPdf(*w->pdf("sim_pdf"));
         bModel->SetPdf(*w->pdf("sim_pdf"));

         // add observed values to data set
         w->var("x")->setVal(fObsValueX);
         w->var("y")->setVal(fObsValueY);
         w->data("data")->add(*sbModel->GetObservables());

         // combined dataset for simultaneous pdf
         RooDataSet *combData = new RooDataSet(
            "combData",
            "combined data",
            *sbModel->GetObservables(),
            Index(*dynamic_cast<RooCategory *>(w->obj("index"))),
            Import("cat1", *dynamic_cast<RooDataSet *>(w->data("data"))),
            Import("cat2", *dynamic_cast<RooDataSet *>(w->data("data")))
         );

   combData->Print("v");
   for (Int_t i = 0; i < combData->numEntries(); i++) {
      combData->get(i)->Print("v");
   }



         //w->data("combData")->get(0)->Print("v");
        // w->data("combData")->get(1)->Print("v");

/*
         RooRealVar x("x","x",0,40);
         RooRealVar mean("mean", "mean", 2, 0, 10);
         RooRealVar mc("mc", "mc", 5, 0, 10);
         RooPoisson p1("p1","p1",x,mean);
         RooPoisson pc("pc","pc",mc,mean);
         RooProdPdf pdf("pdf","pdf",RooArgList(p1,pc));

         RooDataSet *data1 = p1.generate(x,100);
         RooDataSet *data2 = pc.generate(x,50);
*/


//         w->Print();

         // set snapshots
         w->var("sig")->setVal(fObsValueX - w->var("bkg1")->getValV());
         sbModel->SetSnapshot(*sbModel->GetParametersOfInterest());
         w->var("sig")->setVal(0);
         bModel->SetSnapshot(*bModel->GetParametersOfInterest());

         // build test statistic
         SimpleLikelihoodRatioTestStat *slrts =  new SimpleLikelihoodRatioTestStat(*bModel->GetPdf(), *sbModel->GetPdf());
         slrts->SetNullParameters(*bModel->GetSnapshot());
         slrts->SetAltParameters(*sbModel->GetSnapshot());
         slrts->SetAlwaysReuseNLL(kTRUE);

         RatioOfProfiledLikelihoodsTestStat *roplts = new RatioOfProfiledLikelihoodsTestStat(*bModel->GetPdf(), *sbModel->GetPdf());
         roplts->SetAlwaysReuseNLL(kTRUE);

         ProfileLikelihoodTestStat *pllts = new ProfileLikelihoodTestStat(*bModel->GetPdf());
         pllts->SetOneSidedDiscovery(kTRUE);
         pllts->SetAlwaysReuseNLL(kTRUE);

         MaxLikelihoodEstimateTestStat *mlets =
            new MaxLikelihoodEstimateTestStat(*sbModel->GetPdf(), *((RooRealVar *)sbModel->GetParametersOfInterest()->first()));

         NumEventsTestStat *nevts = new NumEventsTestStat(*sbModel->GetPdf());


         FrequentistCalculator *ftc = new FrequentistCalculator(*combData, *sbModel, *bModel);
         ftc->SetToys(50000, 1000);
         //   ftc->SetConditionalMLEsNull(w->set("nuis"));
         //   ftc->SetConditionalMLEsAlt(w->set("nuis"));
         ToyMCSampler *tmcs = (ToyMCSampler *)ftc->GetTestStatSampler();
         tmcs->SetNEventsPerToy(1); // because the model is in number counting form
         tmcs->SetAlwaysUseMultiGen(kTRUE);

         HypoTestResult *htr;
         ProfileLikelihoodCalculator *plc = new ProfileLikelihoodCalculator(*combData, *sbModel);
         plc->SetNullParameters(*bModel->GetSnapshot());
         htr = plc->GetHypoTest();
         htr->Print();
         cout << "PLC " << htr->Significance() << endl;

         tmcs->SetTestStatistic(slrts);
         htr = ftc->GetHypoTest();
         htr->Print();
         cout << "SLRTS " << htr->Significance() << endl;
         tmcs->SetTestStatistic(pllts);
         htr = ftc->GetHypoTest();
         htr->Print();
         cout << "PLLTS " << htr->Significance() << endl;
         tmcs->SetTestStatistic(mlets);
         htr = ftc->GetHypoTest();
         htr->Print();
         cout << "MLETS " << htr->Significance() << endl;
         tmcs->SetTestStatistic(nevts);
         htr = ftc->GetHypoTest();
         htr->Print();
         cout << "NEVTS " << htr->Significance() << endl;
         tmcs->SetTestStatistic(roplts);
         htr = ftc->GetHypoTest();
         htr->Print();
         cout << "ROPLTS " << htr->Significance() << endl;

         regValue(htr->Significance(), "thtc_significance_frequentist");

         delete ftc;
         delete htr;
         delete w;
      }
      return kTRUE ;
   }
} ;











class TestHypoTestInverter2 : public RooUnitTest {
private:
   HypoTestInverter::ECalculatorType fCalculatorType;

public:
   TestHypoTestInverter2(TFile* refFile, Bool_t writeRef, Int_t verbose, HypoTestInverter::ECalculatorType calculatorType, ETestStatType /* testStatType */) : RooUnitTest(TString::Format("HypoTestInverter Upper Limit - Poisson Model with Signal, Background and Efficiency - Calculator Type %d", calculatorType), refFile, writeRef, verbose) {
      fCalculatorType = calculatorType;
   };

   Bool_t testCode() {

      const Double_t testSize = normal_cdf_c(1) * 2;

      // Create workspace and model
      RooWorkspace *w = new RooWorkspace("w", kTRUE);
      createPoissonEfficiencyModel(w);
      ModelConfig *sbModel = (ModelConfig *)w->obj("S+B");
      ModelConfig *bModel = (ModelConfig *)w->obj("B");

      // calculate upper limit with HypoTestInverter
      HypoTestInverter *hti = new HypoTestInverter(*w->data("data"), *sbModel, *bModel, w->var("x"), fCalculatorType, testSize);
      hti->SetFixedScan(10, 0, 20);
      hti->UseCLs(kTRUE);

      if (fCalculatorType == HypoTestInverter::kAsymptotic && _verb == 0) {
         AsymptoticCalculator::SetPrintLevel(0); // print only minimal output
      }

      if (fCalculatorType == HypoTestInverter::kHybrid) {
         // force prior nuisance pdf
         HybridCalculator *hc = (HybridCalculator *)hti->GetHypoTestCalculator();
         w->factory("PROD::priorbkg(constrb, constre)");
         hc->ForcePriorNuisanceNull(*w->pdf("priorbkg"));
         hc->ForcePriorNuisanceAlt(*w->pdf("priorbkg"));
      }

      // Set up the test statistic
      ProfileLikelihoodTestStat *profll = new ProfileLikelihoodTestStat(*sbModel->GetPdf());
      profll->SetOneSided(kTRUE);

      // needed because we have no extended pdf and the ToyMC Sampler evaluation returns an error
      ToyMCSampler *tmcs = (ToyMCSampler *)hti->GetHypoTestCalculator()->GetTestStatSampler();
      tmcs->SetMaxToys(300);
      tmcs->SetNEventsPerToy(1);
      tmcs->SetTestStatistic(profll);
//      tmcs->SetUseMultiGen(kTRUE); // make ToyMCSampler faster

      // calculate interval and extract observed upper limit and expected upper limit (+- sigma)
      HypoTestInverterResult *interval = hti->GetInterval();
      regValue(interval->UpperLimit(), TString::Format("hti2_upper_limit_sig1_calc_%d", fCalculatorType));
      regValue(interval->GetExpectedUpperLimit(0), TString::Format("hti2_exp_upper_limit_sig_calc_%d", fCalculatorType));
      regValue(interval->GetExpectedUpperLimit(-1), TString::Format("hti2_exp_upper_limit-sigma_sig_calc_%d", fCalculatorType));
      regValue(interval->GetExpectedUpperLimit(1), TString::Format("hti2_exp_upper_limit+sigma_sig_calc_%d", fCalculatorType));


      if (_verb >= 1) {
         HypoTestInverterPlot *plot = new HypoTestInverterPlot("HTI_Result_Plot", "HTI Upper Limit Scan", interval);
         TCanvas *c1 = new TCanvas("HypoTestInverter Scan");
         c1->SetLogy(false);
         plot->Draw("2CL CLb");
         c1->SaveAs(TString::Format("hti2 Upper Limit Scan - Calc %d.pdf", fCalculatorType));

         if (_verb == 2) {
            const int n = interval->ArraySize();
            if (n > 0 && interval->GetResult(0)->GetNullDistribution()) {
               TCanvas *c2 = new TCanvas("Test Statistic Distributions", "", 2);
               if (n > 1) {
                  int ny = TMath::CeilNint(sqrt(n));
                  int nx = TMath::CeilNint(double(n) / ny);
                  c2->Divide(nx, ny);
               }
               for (int i = 0; i < n; i++) {
                  if (n > 1) c2->cd(i + 1);
                  SamplingDistPlot *pl = plot->MakeTestStatPlot(i);
                  if (pl == NULL) return kTRUE;
                  pl->SetLogYaxis(kTRUE);
                  pl->Draw();
               }
               c2->SaveAs(TString::Format("hti2 TestStat Distributions - Calc %d.pdf", fCalculatorType));
            }
         }
      }

      // cleanup
      delete interval;
      delete hti;
      delete w;

      return kTRUE ;
   }
};

#include "RooStats/RatioOfProfiledLikelihoodsTestStat.h"
#include "RooStats/MaxLikelihoodEstimateTestStat.h"
#include "RooStats/NumEventsTestStat.h"

static TestStatistic *buildTestStatistic(const ETestStatType testStatType, const ModelConfig &sbModel, const ModelConfig &bModel)
{

   TestStatistic *testStat = NULL;

   if (testStatType == kSimpleLR) {
      SimpleLikelihoodRatioTestStat *slrts = new SimpleLikelihoodRatioTestStat(*sbModel.GetPdf(), *bModel.GetPdf());
      slrts->SetNullParameters(*sbModel.GetSnapshot());
      slrts->SetAltParameters(*bModel.GetSnapshot());
      slrts->SetAlwaysReuseNLL(kTRUE);
      testStat = slrts;
   } else if (testStatType == kRatioLR)  {
      RatioOfProfiledLikelihoodsTestStat *roplts =
         new RatioOfProfiledLikelihoodsTestStat(*sbModel.GetPdf(), *bModel.GetPdf(), bModel.GetSnapshot());
      roplts->SetAlwaysReuseNLL(kTRUE);
      testStat = roplts;
   } else if (testStatType == kMLE) {
      MaxLikelihoodEstimateTestStat *mlets =
         new MaxLikelihoodEstimateTestStat(*sbModel.GetPdf(), *((RooRealVar *)sbModel.GetParametersOfInterest()->first()));
      testStat = mlets;
   } else if (testStatType == kNObs) {
      NumEventsTestStat *nevtts = new NumEventsTestStat(*sbModel.GetPdf());
      testStat = nevtts;
   } else { // kProfileLR, kProfileLROneSided and kProfileLRSigned
      ProfileLikelihoodTestStat *plts = new ProfileLikelihoodTestStat(*sbModel.GetPdf());
      if (testStatType == kProfileLROneSided) plts->SetOneSided(kTRUE);
      if (testStatType == kProfileLRSigned) plts->SetSigned(kTRUE);
      plts->SetAlwaysReuseNLL(kTRUE);
      testStat = plts;
   }

   assert(testStat != NULL); // sanity check - should never happen

   return testStat; // fgNToys seems like a good choice for now
}




