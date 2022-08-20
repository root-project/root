// @(#)root/roostats:$Id$
// Author: Kyle Cranmer, Lorenzo Moneta, Gregory Schott, Wouter Verkerke
/*************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOSTATS_CombinedCalculator
#define ROOSTATS_CombinedCalculator


#include "RooStats/IntervalCalculator.h"

#include "RooStats/HypoTestCalculator.h"

#include "RooStats/ModelConfig.h"

#include "RooAbsPdf.h"

#include "RooAbsData.h"

#include "RooArgSet.h"

// #ifndef ROO_WORKSPACE
// #include "RooWorkspace.h"
// #endif

namespace RooStats {

/** \class CombinedCalculator
   \ingroup Roostats

CombinedCalculator is an interface class for a tools which can produce both RooStats
HypoTestResults and ConfIntervals. The interface currently assumes that any such
calculator can be configured by specifying:

  - a model common model (eg. a family of specific models which includes both the null and alternate),
  - a data set,
  - a set of parameters of which specify the null (including values and const/non-const status),
  - a set of parameters of which specify the alternate (including values and const/non-const status),
  - a set of parameters of nuisance parameters (including values and const/non-const status).

The interface allows one to pass the model, data, and parameters via a workspace
and then specify them with names. The interface also allows one to pass the model,
data, and parameters without a workspace (which is created internally).

After configuring the calculator, one only needs to ask GetHypoTest() (which will
return a HypoTestResult pointer) or GetInterval() (which will return an ConfInterval pointer).

The concrete implementations of this interface should deal with the details of how
the nuisance parameters are dealt with (eg. integration vs. profiling) and which test-statistic is used (perhaps this should be added to the interface).

The motivation for this interface is that we hope to be able to specify the problem
in a common way for several concrete calculators.

*/


   class CombinedCalculator : public IntervalCalculator, public HypoTestCalculator {

   public:

      CombinedCalculator() :
         fSize(0.),
         fPdf(0),
         fData(0)
      {}

      CombinedCalculator(RooAbsData& data, RooAbsPdf& pdf, const RooArgSet& paramsOfInterest,
                         double size = 0.05, const RooArgSet* nullParams = 0, const RooArgSet* altParams = 0, const RooArgSet* nuisParams = 0) :

         fPdf(&pdf),
         fData(&data),
         fPOI(paramsOfInterest)
      {
         if (nullParams) fNullParams.add(*nullParams);
         if (altParams) fAlternateParams.add(*altParams);
         if (nuisParams) fNuisParams.add(*nuisParams);
         SetTestSize(size);
      }

      /// constructor from data and model configuration
      CombinedCalculator(RooAbsData& data, const ModelConfig& model,
                         double size = 0.05) :
         fPdf(0),
         fData(&data)
      {
         SetModel(model);
         SetTestSize(size);
      }

      /// destructor.
      ~CombinedCalculator() override { }

      /// Main interface to get a ConfInterval, pure virtual
      ConfInterval* GetInterval() const override = 0;
      /// main interface to get a HypoTestResult, pure virtual
      HypoTestResult* GetHypoTest() const override = 0;

      /// set the size of the test (rate of Type I error) ( Eg. 0.05 for a 95% Confidence Interval)
      void SetTestSize(double size) override {fSize = size;}
      /// set the confidence level for the interval (eg. 0.95 for a 95% Confidence Interval)
      void SetConfidenceLevel(double cl) override {fSize = 1.-cl;}
      /// Get the size of the test (eg. rate of Type I error)
      double Size() const override {return fSize;}
      /// Get the Confidence level for the test
      double ConfidenceLevel()  const override {return 1.-fSize;}

      /// Set the DataSet, add to the workspace if not already there
      void SetData(RooAbsData & data) override {
         fData = &data;
      }

      /// set the model (in this case can set only the parameters for the null hypothesis)
      void SetModel(const ModelConfig & model) override {
         fPdf = model.GetPdf();
         if (model.GetParametersOfInterest()) SetParameters(*model.GetParametersOfInterest());
         if (model.GetSnapshot()) SetNullParameters(*model.GetSnapshot());
         if (model.GetNuisanceParameters()) SetNuisanceParameters(*model.GetNuisanceParameters());
         if (model.GetConditionalObservables()) SetConditionalObservables(*model.GetConditionalObservables());
         if (model.GetGlobalObservables()) SetGlobalObservables(*model.GetGlobalObservables());
      }

      void SetNullModel( const ModelConfig &) override {  // to be understood what to do
      }
      void SetAlternateModel(const ModelConfig &) override {  // to be understood what to do
      }

      /* specific setting - keep for convenience-  some of them could be removed */

      /// Set the Pdf
      virtual void SetPdf(RooAbsPdf& pdf) { fPdf = &pdf; }

      /// specify the parameters of interest in the interval
      virtual void SetParameters(const RooArgSet& set) { fPOI.removeAll(); fPOI.add(set); }

       /// specify the nuisance parameters (eg. the rest of the parameters)
      virtual void SetNuisanceParameters(const RooArgSet& set) {fNuisParams.removeAll(); fNuisParams.add(set);}

      /// set parameter values for the null if using a common PDF
      virtual void SetNullParameters(const RooArgSet& set) {fNullParams.removeAll(); fNullParams.add(set);}

      /// set parameter values for the alternate if using a common PDF
      virtual void SetAlternateParameters(const RooArgSet& set) {fAlternateParams.removeAll(); fAlternateParams.add(set);}

      /// set conditional observables needed for computing the NLL
      virtual void SetConditionalObservables(const RooArgSet& set) {fConditionalObs.removeAll(); fConditionalObs.add(set);}

       /// set global observables needed for computing the NLL
      virtual void SetGlobalObservables(const RooArgSet& set) {fGlobalObs.removeAll(); fGlobalObs.add(set);}


   protected:

      RooAbsPdf * GetPdf() const { return fPdf; }
      RooAbsData * GetData() const { return fData; }

      double fSize; ///< size of the test (eg. specified rate of Type I error)

      RooAbsPdf  * fPdf;
      RooAbsData * fData;
      RooArgSet fPOI;             ///< RooArgSet specifying parameters of interest for interval
      RooArgSet fNullParams;      ///< RooArgSet specifying null parameters for hypothesis test
      RooArgSet fAlternateParams; ///< RooArgSet specifying alternate parameters for hypothesis test
      RooArgSet fNuisParams;      ///< RooArgSet specifying nuisance parameters for interval
      RooArgSet fConditionalObs;  ///< RooArgSet specifying the conditional observables
      RooArgSet fGlobalObs;       ///< RooArgSet specifying the global observables


      ClassDefOverride(CombinedCalculator,2) // A base class that is for tools that can be both HypoTestCalculators and IntervalCalculators

   };
}


#endif
