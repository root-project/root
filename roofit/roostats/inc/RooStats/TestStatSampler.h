// @(#)root/roostats:$Id$
// Author: Kyle Cranmer, Lorenzo Moneta, Gregory Schott, Wouter Verkerke
/*************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOSTATS_TestStatSampler
#define ROOSTATS_TestStatSampler


#include "Rtypes.h"

class RooAbsArg;
class RooAbsData;
class RooArgSet;
class RooAbsPdf;




namespace RooStats {

   class SamplingDistribution;
   class TestStatistic;

/** \class RooStats::TestStatSampler
    \ingroup Roostats

TestStatSampler is an interface class for a tools which produce RooStats
SamplingDistributions. Tools that implement this interface are expected to be
used for coverage studies, the Neyman Construction, etc.

*/

   class TestStatSampler {

   public:
      virtual ~TestStatSampler() {}

      /// Main interface to get a ConfInterval, pure virtual
      virtual SamplingDistribution* GetSamplingDistribution(RooArgSet& paramsOfInterest) = 0;

      /// Main interface to evaluate the test statistic on a dataset
      virtual double EvaluateTestStatistic(RooAbsData& data, RooArgSet& paramsOfInterest) = 0;

      /// Get the TestStatistic
      virtual TestStatistic* GetTestStatistic()  const = 0;

      /// Get the Confidence level for the test
      virtual double ConfidenceLevel()  const = 0;

      /// Common Initialization
      virtual void Initialize(RooAbsArg& testStatistic, RooArgSet& paramsOfInterest, RooArgSet& nuisanceParameters) = 0;

      /// Set the Pdf, add to the workspace if not already there
      virtual void SetPdf(RooAbsPdf&) = 0;

      /// How to randomize the prior. Set to nullptr to deactivate randomization.
      virtual void SetPriorNuisance(RooAbsPdf*) = 0;

      /// specify the values of parameters used when evaluating test statistic
      virtual void SetParametersForTestStat(const RooArgSet& /*nullpoi*/) = 0;

      /// specify the nuisance parameters (eg. the rest of the parameters)
      virtual void SetNuisanceParameters(const RooArgSet&) = 0;
      /// specify the observables in the dataset (needed to evaluate the test statistic)
      virtual void SetObservables(const RooArgSet& ) = 0;
      /// specify the conditional observables
      virtual void SetGlobalObservables(const RooArgSet& ) = 0;

      /// set the size of the test (rate of Type I error) ( Eg. 0.05 for a 95% Confidence Interval)
      virtual void SetTestSize(double size) = 0;
      /// set the confidence level for the interval (eg. 0.95 for a 95% Confidence Interval)
      virtual void SetConfidenceLevel(double cl) = 0;

      /// Set the TestStatistic (want the argument to be a function of the data & parameter points
      virtual void SetTestStatistic(TestStatistic* testStatistic) = 0;

      /// Set the name of the sampling distribution used for plotting
      virtual void SetSamplingDistName(const char* name) = 0;


   protected:
      ClassDef(TestStatSampler,1)   // Interface for tools setting limits (producing confidence intervals)
   };
}


#endif
