// @(#)root/roostats:$Id$
// Author: Kyle Cranmer, Lorenzo Moneta, Gregory Schott, Wouter Verkerke
// Additions and modifications by Mario Pelliccioni
/*************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOSTATS_ToyMCSamplerOld
#define ROOSTATS_ToyMCSamplerOld

//_________________________________________________
/*
BEGIN_HTML
<p>
ToyMCSamplerOld is a simple implementation of the TestStatSampler interface.
It generates Toy Monte Carlo for a given parameter point, and evaluates a 
test statistic that the user specifies (passed via the RooStats::TestStatistic interface).

Development notes: We need to provide a nice way for the user to:
<ul>
  <li>specify the number of toy experiments (needed to probe a given confidence level)</li>
  <li>specify if the number of events per toy experiment should be fixed (conditioning) or floating (unconditional)</li>
  <li>specify if any auxiliary observations should be fixed (conditioning) or floating (unconditional)</li>
  <li>specify if nuisance paramters should be part of the toy MC: eg: integrated out (Bayesian marginalization)</li>
</ul>

All of these should be made fairly explicit in the interface.
</p>
END_HTML
*/
//

#ifndef ROOT_Rtypes
#include "Rtypes.h"
#endif

#include <vector>
#include <string>
#include <sstream>

#include "RooStats/TestStatSampler.h"
#include "RooStats/SamplingDistribution.h"
#include "RooStats/TestStatistic.h"
#include "RooStats/RooStatsUtils.h"

#include "RooWorkspace.h"
#include "RooMsgService.h"
#include "RooAbsPdf.h"
#include "TRandom.h"

#include "RooDataSet.h"

namespace RooStats {

    class ToyMCSamplerOld : public TestStatSampler {


  public:
    ToyMCSamplerOld(TestStatistic &ts) {
      fTestStat = &ts;
      fWS = new RooWorkspace();
      fOwnsWorkspace = true;
      fDataName = "";
      fPdfName = "";
      fNullPOI = 0;
      fNuisParams=0;
      fObservables=0;
      fExtended = kTRUE;
      fRand = new TRandom();
      fCounter=0;
      fVarName = fTestStat->GetVarName();
      fLastDataSet = 0;
      fNevents = 0; 
      fNtoys = 0; 
      fSize = 0.05; 
    }

    virtual ~ToyMCSamplerOld() {
      if(fOwnsWorkspace) delete fWS;
      if(fRand) delete fRand;
      if(fLastDataSet) delete fLastDataSet;
    }
    
    // Extended interface to append to sampling distribution more samples
    virtual SamplingDistribution* AppendSamplingDistribution(RooArgSet& allParameters, 
							     SamplingDistribution* last, 
							     Int_t additionalMC) {

      Int_t tmp = fNtoys;
      fNtoys = additionalMC;
      SamplingDistribution* newSamples = GetSamplingDistribution(allParameters);
      fNtoys = tmp;

      if(last){
	last->Add(newSamples);
	delete newSamples;
	return last;
      }

      return newSamples;
    }

    // Main interface to get a SamplingDistribution
    virtual SamplingDistribution* GetSamplingDistribution(RooArgSet& allParameters) {
      std::vector<Double_t> testStatVec;
      //       cout << " about to generate sampling dist " << endl;

      RooFit::MsgLevel msglevel = RooMsgService::instance().globalKillBelow();
      RooMsgService::instance().setGlobalKillBelow(RooFit::ERROR) ;

      for(Int_t i=0; i<fNtoys; ++i){
	//cout << " on toy number " << i << endl;
	//	RooAbsData* toydata = (RooAbsData*)GenerateToyData(allParameters);
	//	testStatVec.push_back( fTestStat->Evaluate(*toydata, allParameters) );
	//	delete toydata;

	RooDataSet* toydata = (RooDataSet*)GenerateToyData(allParameters);

	// note, evaluation always done at fNullPOI
	testStatVec.push_back( fTestStat->Evaluate(*toydata, *fNullPOI) );

	// want to clean up memory, but delete toydata causes problem with 
	// nll->setData(data, noclone) because pointer to last data set is no longer valid
	//	delete toydata; 

	// instead, delete previous data set
	if(fLastDataSet) delete fLastDataSet;
	fLastDataSet = toydata;
      }
     
       RooMsgService::instance().setGlobalKillBelow(msglevel);

      //      cout << " generated sampling dist " << endl;
      return new SamplingDistribution( "temp",//MakeName(allParameters).c_str(),
				       "Sampling Distribution of Test Statistic", testStatVec, fVarName );
    } 

     virtual RooAbsData* GenerateToyData(RooArgSet& allParameters) const {
       // This method generates a toy dataset for the given parameter point.


       //       cout << "fNevents = " << fNevents << endl;
       RooAbsPdf* pdf = fWS->pdf(fPdfName);
       if(!fObservables){
	 cout << "Observables not specified in ToyMCSamplerOld, will try to determine.  "
	      << "Will ignore all constant parameters, parameters of interest, and nuisance parameters." << endl;
	 RooArgSet* observables = pdf->getVariables();
	 RemoveConstantParameters(observables); // observables might be set constant, this is just a guess


	 if(fNullPOI) observables->remove(*fNullPOI, kFALSE, kTRUE);
	 if(fNuisParams) observables->remove(*fNuisParams, kFALSE, kTRUE);
	 cout << "will use the following as observables when generating data" << endl;
	 observables->Print();
	 fObservables=observables;
       } /*else {
	   cout << "obs set: will use the following as observables when generating data" << endl;
	   fObservables->Print();
	   }*/

       //fluctuate the number of events if fExtended is on.  
       // This is a bit slippery for number counting expts. where entry in data and

       /*
       // model is number of events, and so number of entries in data always =1.
       Int_t nEvents = fNevents;
       if(fExtended) {
	 if( pdf->expectedEvents(*fObservables) > 0){
	   // if PDF knows expected events use it instead
	   nEvents = fRand->Poisson(pdf->expectedEvents(*fObservables));
	 } else{
	   nEvents = fRand->Poisson(fNevents);
	 }
       }
       */

       // Set the parameters to desired values for generating toys
       RooArgSet* parameters = pdf->getParameters(fObservables);
       RooStats::SetParameters(&allParameters, parameters);

       /*       
	 cout << "expected events = " <<  pdf->expectedEvents(*observables) 
	    << "fExtended = " << fExtended
	    << "fNevents = " << fNevents << " fNevents " 
	    << "generating" << nEvents << " events " << endl;
       */
       
       RooFit::MsgLevel level = RooMsgService::instance().globalKillBelow();
       RooMsgService::instance().setGlobalKillBelow(RooFit::ERROR) ;

       //       cout << "nEvents = " << nEvents << endl;
       RooAbsData* data = NULL;
       if(fExtended) {
	 data = (RooAbsData*)pdf->generate(*fObservables, RooFit::Extended());
       } else {
	 data = (RooAbsData*)pdf->generate(*fObservables, fNevents);
     }

       RooMsgService::instance().setGlobalKillBelow(level) ;
       delete parameters;
       return data;
     }

     // helper method to create meaningful names for sampling dist
     string MakeName(RooArgSet& /*params*/){
       std::ostringstream str;
       str<<"SamplingDist_"<< fCounter;
       fCounter++;
       std::string buf = str.str(); 
       return buf ;       
     }

      // Main interface to evaluate the test statistic on a dataset
     virtual Double_t EvaluateTestStatistic(RooAbsData& data, RooArgSet& allParameters) {
       return fTestStat->Evaluate(data, allParameters);
     }

      // Get the TestStatistic
      virtual TestStatistic* GetTestStatistic()  const {
	 return fTestStat;
      }  
    
      // Get the Confidence level for the test
      virtual Double_t ConfidenceLevel()  const {return 1.-fSize;}  

      // Common Initialization
      virtual void Initialize(RooAbsArg& /*testStatistic*/, 
			      RooArgSet& /*paramsOfInterest*/, 
			      RooArgSet& /*nuisanceParameters*/) {}

      //set the parameters for the toyMC generation
      virtual void SetNToys(const Int_t ntoy) {
        fNtoys = ntoy;
      }

      virtual void SetNEventsPerToy(const Int_t nevents) {
        fNevents = nevents;
      }


      virtual void SetExtended(const Bool_t isExtended) {
        fExtended = isExtended;
      }

      // Set the DataSet, add to the the workspace if not already there
      virtual void SetData(RooAbsData& data) {
	if(&data){
	  fWS->import(data);
	  fDataName = data.GetName();
	  fWS->Print();
	}
      }
      // Set the Pdf, add to the the workspace if not already there
      virtual void SetPdf(RooAbsPdf& pdf) { 
	if(&pdf){
	  fWS->import(pdf);
	  fPdfName = pdf.GetName();
	}
      }

      // specify the name of the dataset in the workspace to be used
      virtual void SetData(const char* name) {fDataName = name;}
      // specify the name of the PDF in the workspace to be used
      virtual void SetPdf(const char* name) {fPdfName = name;}
      // How to randomize the prior. Set to NULL to deactivate randomization.
      virtual void SetPriorNuisance(RooAbsPdf* /*not supported*/) {}

      // specify the values of parameters used when evaluating test statistic
      virtual void SetParametersForTestStat(const RooArgSet& nullpoi) {fNullPOI = (RooArgSet*)nullpoi.snapshot();}
      // specify the nuisance parameters (eg. the rest of the parameters)
      virtual void SetNuisanceParameters(const RooArgSet& set) {fNuisParams = &set;}
      // specify the observables in the dataset (needed to evaluate the test statistic)
      virtual void SetObservables(const RooArgSet& set) {fObservables = &set;}
      // specify the conditional observables
      virtual void SetGlobalObservables(const RooArgSet& ) {}

      // set the size of the test (rate of Type I error) ( Eg. 0.05 for a 95% Confidence Interval)
      virtual void SetTestSize(Double_t size) {fSize = size;}
      // set the confidence level for the interval (eg. 0.95 for a 95% Confidence Interval)
      virtual void SetConfidenceLevel(Double_t cl) {fSize = 1.-cl;}

      // Set the TestStatistic (want the argument to be a function of the data & parameter points
      virtual void SetTestStatistic(TestStatistic* testStat)   {
	fTestStat = testStat;
      }  

      // Set the name of the sampling distribution used for plotting
      void SetSamplingDistName(const char* name) { if(name) fSamplingDistName = name; }

      
   private:
      Double_t fSize;
      RooWorkspace* fWS; // a workspace that owns all the components to be used by the calculator
      Bool_t fOwnsWorkspace; // flag if this object owns its workspace
      string fSamplingDistName; // name of the model
      const char* fPdfName; // name of  common PDF in workspace
      const char* fDataName; // name of data set in workspace
      RooArgSet* fNullPOI; // the values of parameters used when evaluating test statistic
      const RooArgSet* fNuisParams;// RooArgSet specifying  nuisance parameters for interval
      mutable const RooArgSet* fObservables; // RooArgSet specifying the observables in the dataset (needed to evaluate the test statistic)
      TestStatistic* fTestStat; // pointer to the test statistic that is being sampled
      Int_t fNtoys; // number of toys to generate
      Int_t fNevents; // number of events per toy (may be ignored depending on settings)
      Bool_t fExtended; // if nEvents should fluctuate
      TRandom* fRand; // random generator
      TString fVarName; // name of test statistic

      Int_t fCounter; // counter for naming sampling dist objects

      RooDataSet* fLastDataSet; // work around for memory issues in nllvar->setData(data, noclone)

   protected:
      ClassDef(ToyMCSamplerOld,1)   // A simple implementation of the TestStatSampler interface
	};
}


#endif
