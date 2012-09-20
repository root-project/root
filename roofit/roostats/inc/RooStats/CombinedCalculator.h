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


#ifndef ROOSTATS_IntervalCalculator
#include "RooStats/IntervalCalculator.h"
#endif

#ifndef ROOSTATS_HypoTestCalculator
#include "RooStats/HypoTestCalculator.h"
#endif

#ifndef ROOSTATS_ModelConfig
#include "RooStats/ModelConfig.h"
#endif

#ifndef ROO_ABS_PDF
#include "RooAbsPdf.h"
#endif

#ifndef ROO_ABS_DATA
#include "RooAbsData.h"
#endif

#ifndef ROO_ARG_SET
#include "RooArgSet.h"
#endif

// #ifndef ROO_WORKSPACE
// #include "RooWorkspace.h"
// #endif


//_________________________________________________
/*
BEGIN_HTML
<p>
CombinedCalculator is an interface class for a tools which can produce both RooStats HypoTestResults and ConfIntervals.  
The interface currently assumes that any such calculator can be configured by specifying:
<ul>
 <li>a model common model (eg. a family of specific models which includes both the null and alternate),</li>
 <li>a data set, </li>
 <li>a set of parameters of which specify the null (including values and const/non-const status), </li>
 <li>a set of parameters of which specify the alternate (including values and const/non-const status),</li>
 <li>a set of parameters of nuisance parameters  (including values and const/non-const status).</li>
</ul>
The interface allows one to pass the model, data, and parameters via a workspace and then specify them with names.
The interface also allows one to pass the model, data, and parameters without a workspace (which is created internally).
</p>
<p>
After configuring the calculator, one only needs to ask GetHypoTest() (which will return a HypoTestResult pointer) or GetInterval() (which will return an ConfInterval pointer).
</p>
<p>
The concrete implementations of this interface should deal with the details of how the nuisance parameters are
dealt with (eg. integration vs. profiling) and which test-statistic is used (perhaps this should be added to the interface).
</p>
<p>
The motivation for this interface is that we hope to be able to specify the problem in a common way for several concrete calculators.
</p>
END_HTML
*/
//

namespace RooStats {

   class CombinedCalculator : public IntervalCalculator, public HypoTestCalculator {

   public:

      CombinedCalculator() : 
         fSize(0.),
         fPdf(0),
         fData(0)
      {}

      CombinedCalculator(RooAbsData& data, RooAbsPdf& pdf, const RooArgSet& paramsOfInterest, 
                         Double_t size = 0.05, const RooArgSet* nullParams = 0, const RooArgSet* altParams = 0, const RooArgSet* nuisParams = 0) : 

         fPdf(&pdf),
         fData(&data),
         fPOI(paramsOfInterest)
      {
         if (nullParams) fNullParams.add(*nullParams); 
         if (altParams) fAlternateParams.add(*altParams); 
         if (nuisParams) fNuisParams.add(*nuisParams); 
         SetTestSize(size);
      }

      // constructor from data and model configuration
      CombinedCalculator(RooAbsData& data, const ModelConfig& model,
                         Double_t size = 0.05) : 
         fPdf(0),
         fData(&data)
      {
         SetModel(model);
         SetTestSize(size);
      }


      // destructor.
      virtual ~CombinedCalculator() { }


    
      // Main interface to get a ConfInterval, pure virtual
      virtual ConfInterval* GetInterval() const = 0; 
      // main interface to get a HypoTestResult, pure virtual
      virtual HypoTestResult* GetHypoTest() const = 0;   

      // set the size of the test (rate of Type I error) ( Eg. 0.05 for a 95% Confidence Interval)
      virtual void SetTestSize(Double_t size) {fSize = size;}
      // set the confidence level for the interval (eg. 0.95 for a 95% Confidence Interval)
      virtual void SetConfidenceLevel(Double_t cl) {fSize = 1.-cl;}
      // Get the size of the test (eg. rate of Type I error)
      virtual Double_t Size() const {return fSize;}
      // Get the Confidence level for the test
      virtual Double_t ConfidenceLevel()  const {return 1.-fSize;}
    
      // Set the DataSet, add to the the workspace if not already there
      virtual void SetData(RooAbsData & data) {    
         fData = &data;
      }

      // set the model (in this case can set only the parameters for the null hypothesis)
      virtual void SetModel(const ModelConfig & model) { 
         fPdf = model.GetPdf();
         if (model.GetParametersOfInterest()) SetParameters(*model.GetParametersOfInterest()); 
         if (model.GetSnapshot()) SetNullParameters(*model.GetSnapshot());
         if (model.GetNuisanceParameters()) SetNuisanceParameters(*model.GetNuisanceParameters()); 
         if (model.GetConditionalObservables()) SetConditionalObservables(*model.GetConditionalObservables()); 
      }
      
      virtual void SetNullModel( const ModelConfig &) {  // to be understood what to do 
      }
      virtual void SetAlternateModel(const ModelConfig &) {  // to be understood what to do 
      }

      /* specific setting - keep for convenience-  some of them could be removed */

      // Set the Pdf 
      virtual void SetPdf(RooAbsPdf& pdf) { fPdf = &pdf; }

      // specify the parameters of interest in the interval
      virtual void SetParameters(const RooArgSet& set) { fPOI.removeAll(); fPOI.add(set); }

       // specify the nuisance parameters (eg. the rest of the parameters)
      virtual void SetNuisanceParameters(const RooArgSet& set) {fNuisParams.removeAll(); fNuisParams.add(set);}
    
      // set parameter values for the null if using a common PDF
      virtual void SetNullParameters(const RooArgSet& set) {fNullParams.removeAll(); fNullParams.add(set);}

      // set parameter values for the alternate if using a common PDF
      virtual void SetAlternateParameters(const RooArgSet& set) {fAlternateParams.removeAll(); fAlternateParams.add(set);}

      // set conditional observables needed for computing the NLL 
      virtual void SetConditionalObservables(const RooArgSet& set) {fConditionalObs.removeAll(); fConditionalObs.add(set);}

         
   protected:

      RooAbsPdf * GetPdf() const { return fPdf; }
      RooAbsData * GetData() const { return fData; }

      Double_t fSize; // size of the test (eg. specified rate of Type I error)

      RooAbsPdf  * fPdf; 
      RooAbsData * fData; 
      RooArgSet fPOI; // RooArgSet specifying  parameters of interest for interval
      RooArgSet fNullParams; // RooArgSet specifying null parameters for hypothesis test
      RooArgSet fAlternateParams; // RooArgSet specifying alternate parameters for hypothesis test       // Is it used ????
      RooArgSet fNuisParams;// RooArgSet specifying  nuisance parameters for interval
      RooArgSet fConditionalObs; // RooArgSet specifying the conditional observables


      ClassDef(CombinedCalculator,1) // A base class that is for tools that can be both HypoTestCalculators and IntervalCalculators
    
   };
}


#endif
