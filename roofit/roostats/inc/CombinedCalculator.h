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

#ifndef ROO_ABS_PDF
#include "RooAbsPdf.h"
#endif

#ifndef ROO_ABS_DATA
#include "RooAbsData.h"
#endif

#ifndef ROO_ARG_SET
#include "RooArgSet.h"
#endif

#ifndef ROO_WORKSPACE
#include "RooWorkspace.h"
#endif


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
      CombinedCalculator(){
         // default constructor
         fWS = 0;
         fNullParams = 0;
         fAlternateParams = 0;
         fPOI = 0;
         fNuisParams = 0;
         fOwnsWorkspace = false;
      }

      CombinedCalculator(RooWorkspace& ws, RooAbsData& data, RooAbsPdf& pdf, RooArgSet& paramsOfInterest, 
                         Double_t size = 0.05, RooArgSet* nullParams = 0, RooArgSet* altParams = 0){
         // alternate constructor
         SetWorkspace(ws);
         SetData(data);
         SetPdf(pdf);
         SetParameters(paramsOfInterest);
         SetTestSize(size);
         if(nullParams ) 
            SetNullParameters(*nullParams);
         else
            SetNullParameters(paramsOfInterest);
         if (altParams) SetAlternateParameters(*altParams);
         fOwnsWorkspace = false;
      }

      CombinedCalculator(RooAbsData& data, RooAbsPdf& pdf, RooArgSet& paramsOfInterest, 
                         Double_t size = 0.05, RooArgSet* nullParams = 0, RooArgSet* altParams = 0){
         // alternate constructor
         fWS = new RooWorkspace();
         fOwnsWorkspace = true;
         SetData(data);
         SetPdf(pdf);
         SetParameters(paramsOfInterest);
         SetTestSize(size);
         if(nullParams ) 
            SetNullParameters(*nullParams);
         else
            SetNullParameters(paramsOfInterest);
         if (altParams) SetAlternateParameters(*altParams);
      }

      virtual ~CombinedCalculator() {
         // destructor.
         if( fOwnsWorkspace && fWS) delete fWS;
         // commented out b/c currently the calculator does not own these.  Change if we clone.
         //      if (fWS) delete fWS;
         //      if (fNullParams) delete fNullParams;
         //      if (fAlternateParams) delete fAlternateParams;
         //      if (fPOI) delete fPOI;
         //      if (fNuisParams) delete fNuisParams;
      }

    
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
    

      // set a workspace that owns all the necessary components for the analysis
      virtual void SetWorkspace(RooWorkspace & ws) {
         if (!fWS)
            fWS = &ws;
         else{
            RooMsgService::instance().setGlobalKillBelow(RooFit::ERROR) ;
            fWS->merge(ws);
            RooMsgService::instance().setGlobalKillBelow(RooFit::DEBUG) ;
         }

      }

      // Set the DataSet, add to the the workspace if not already there
      virtual void SetData(RooAbsData & data) {      
         if (!fWS) {
            fWS = new RooWorkspace();
            fOwnsWorkspace = true; 
         }
         if (! fWS->data( data.GetName() ) ){
            RooMsgService::instance().setGlobalKillBelow(RooFit::ERROR) ;
            fWS->import(data);
            RooMsgService::instance().setGlobalKillBelow(RooFit::DEBUG) ;
         }
         SetData( data.GetName() );

      };

      // Set the Pdf, add to the the workspace if not already there
      virtual void SetPdf(RooAbsPdf& pdf) {
         if (!fWS) 
            fWS = new RooWorkspace();
         if (! fWS->pdf( pdf.GetName() ) ){
            RooMsgService::instance().setGlobalKillBelow(RooFit::ERROR) ;
            fWS->import(pdf);
            RooMsgService::instance().setGlobalKillBelow(RooFit::DEBUG) ;
         }
         SetPdf( pdf.GetName() );
      }

      // Set the Pdf, add to the the workspace if not already there
      virtual void SetCommonPdf(RooAbsPdf& pdf) { SetPdf(pdf);}
      // Set the Pdf, add to the the workspace if not already there
      virtual void SetNullPdf(RooAbsPdf& pdf) { SetPdf(pdf);}
      // Set the Pdf, add to the the workspace if not already there
      virtual void SetAlternatePdf(RooAbsPdf& pdf) { SetPdf(pdf);}

      // specify the name of the PDF in the workspace to be used
      virtual void SetPdf(const char* name) {fPdfName = name;}
      // specify the name of the dataset in the workspace to be used
      virtual void SetData(const char* name){fDataName = name;}
      // specify the parameters of interest in the interval
      virtual void SetParameters(RooArgSet& set) {fPOI = &set;}
      // specify the nuisance parameters (eg. the rest of the parameters)
      virtual void SetNuisanceParameters(RooArgSet& set) {fNuisParams = &set;}
    
      // from HypoTestCalculator
      // set the PDF for the null hypothesis.  Needs to be the common one
      virtual void SetNullPdf(const char* name) {SetPdf(name);}
      // set the PDF for the alternate hypothesis. Needs to be the common one
      virtual void SetAlternatePdf(const char* name) {SetPdf(name);}
      // set a common PDF for both the null and alternate hypotheses
      virtual void SetCommonPdf(const char* name) {SetPdf(name);}
      // set parameter values for the null if using a common PDF
      virtual void SetNullParameters(RooArgSet& set) {fNullParams = &set;}
      // set parameter values for the alternate if using a common PDF
      virtual void SetAlternateParameters(RooArgSet& set) {fAlternateParams = &set;}
    

   protected:

      Double_t fSize; // size of the test (eg. specified rate of Type I error)
      RooWorkspace* fWS; // a workspace that owns all the components to be used by the calculator
      const char* fPdfName; // name of  common PDF in workspace
      const char* fDataName; // name of data set in workspace
      RooArgSet* fNullParams; // RooArgSet specifying null parameters for hypothesis test
      RooArgSet* fAlternateParams; // RooArgSet specifying alternate parameters for hypothesis test
      RooArgSet* fPOI; // RooArgSet specifying  parameters of interest for interval
      RooArgSet* fNuisParams;// RooArgSet specifying  nuisance parameters for interval
      Bool_t fOwnsWorkspace;


      ClassDef(CombinedCalculator,1) // A base class that is for tools that can be both HypoTestCalculators and IntervalCalculators
    
   };
}


#endif
