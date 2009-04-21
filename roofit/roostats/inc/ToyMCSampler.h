// @(#)root/roostats:$Id: ToyMCSampler.h 26805 2009-01-13 17:45:57Z cranmer $
// Author: Kyle Cranmer, Lorenzo Moneta, Gregory Schott, Wouter Verkerke
// Additions and modifications by Mario Pelliccioni
/*************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOSTATS_ToyMCSampler
#define ROOSTATS_ToyMCSampler

//_________________________________________________
/*
BEGIN_HTML
<p>
ToyMCSampler is a simple implementation of the TestStatSampler interface.
It generates Toy Monte Carlo for a given parameter point, and evaluates a 
test statistic that the user specifies (passed via the RooStats::TestStatistic interface).

We need to provide a nice way for the user to:
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
#include <sstream>

#include "RooStats/TestStatSampler.h"
#include "RooStats/SamplingDistribution.h"
#include "RooStats/TestStatistic.h"
#include "RooStats/RooStatsUtils.h"

#include "RooGlobalFunc.h"
#include "RooWorkspace.h"
#include "TRandom.h"

#include "RooDataSet.h"

namespace RooStats {

    class ToyMCSampler : public TestStatSampler {


  public:
    ToyMCSampler(TestStatistic &ts) {
      fTestStat = &ts;
      fWS = new RooWorkspace();
      fOwnsWorkspace = true;
      fDataName = "";
      fPdfName = "";
      fPOI = 0;
      fNuisParams=0;
      fExtended = kTRUE;
      fRand = new TRandom();
      fCounter=0;
    }

    virtual ~ToyMCSampler() {
      if(fOwnsWorkspace) delete fWS;
      if(fRand) delete fRand;
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

      RooMsgService::instance().setGlobalKillBelow(RooMsgService::ERROR) ;
      RooMsgService::instance().setGlobalKillBelow(RooMsgService::DEBUG) ;

      for(Int_t i=0; i<fNtoys; ++i){
	//cout << " on toy number " << i << endl;
	//	RooAbsData* toydata = (RooAbsData*)GenerateToyData(allParameters);
	//	testStatVec.push_back( fTestStat->Evaluate(*toydata, allParameters) );
	//	delete toydata;

	RooDataSet* toydata = (RooDataSet*)GenerateToyData(allParameters);
	testStatVec.push_back( fTestStat->Evaluate(*toydata, allParameters) );
	delete toydata;
      }

      //      cout << " generated sampling dist " << endl;
      return new SamplingDistribution( MakeName(allParameters),
				      "Sampling Distribution of Test Statistic", testStatVec );
    } 

     virtual RooAbsData* GenerateToyData(RooArgSet& allParameters) const {

       //       cout << "fNevents = " << fNevents << endl;
       RooAbsPdf* pdf = fWS->pdf(fPdfName);
       // need a nicer way to specify observables in the dataset
       RooArgSet* observables = pdf->getVariables();

       // Set the parameters to desired values for generating toys
       RooStats::SetParameters(&allParameters, observables);

       if(fPOI) observables->remove(*fPOI, kFALSE, kTRUE);
       if(fNuisParams) observables->remove(*fNuisParams, kFALSE, kTRUE);

       // observables->Print("verbose");

       //fluctuate the number of events if fExtended is on.  
       // This is a bit slippery for number counting expts. where entry in data and
       // model is number of events, and so number of entries in data always =1.
       Int_t nEvents = fNevents;
       if(fExtended) {
	 if( pdf->expectedEvents(*observables) > 0){
	   // if PDF knows expected events use it instead
	   nEvents = fRand->Poisson(pdf->expectedEvents(*observables));
	 } else{
	   nEvents = fRand->Poisson(fNevents);
	 }
       }

       /*       
	 cout << "expected events = " <<  pdf->expectedEvents(*observables) 
	    << "fExtended = " << fExtended
	    << "fNevents = " << fNevents << " fNevents " 
	    << "generating" << nEvents << " events " << endl;
       */
       

       RooAbsData* data = (RooAbsData*)pdf->generate(*observables, nEvents);
       delete observables;
       //       delete pdf;
       return data;
     }

     // helper method to create meaningful names for sampling dist
     const char* MakeName(RooArgSet& /*params*/){
       /*
       std::string name;
       TIter      itr = params.createIterator();
       RooRealVar* myarg;
       while ((myarg = (RooRealVar *)itr.Next())) { 
	 name += myarg->GetName();
	 std::stringstream str;
	 str<<"_"<< myarg->getVal() << "__";

	 name += str.str();
       }
       */

       std::stringstream str;
       str<<"SamplingDist_"<< fCounter;
       fCounter++;
       return str.str().c_str();
       
     }

      // Main interface to evaluate the test statistic on a dataset
     virtual Double_t EvaluateTestStatistic(RooAbsData& data, RooArgSet& allParameters) {
       return fTestStat->Evaluate(data, allParameters);
     }

      // Get the TestStatistic
      virtual const RooAbsArg* GetTestStatistic()  const {
	 return fTestStat->GetTestStatistic();}  
    
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

      // specify the parameters of interest in the interval
      virtual void SetParameters(RooArgSet& set) {fPOI = &set;}
      // specify the nuisance parameters (eg. the rest of the parameters)
      virtual void SetNuisanceParameters(RooArgSet& set) {fNuisParams = &set;}

      // set the size of the test (rate of Type I error) ( Eg. 0.05 for a 95% Confidence Interval)
      virtual void SetTestSize(Double_t size) {fSize = size;}
      // set the confidence level for the interval (eg. 0.95 for a 95% Confidence Interval)
      virtual void SetConfidenceLevel(Double_t cl) {fSize = 1.-cl;}

      // Set the TestStatistic (want the argument to be a function of the data & parameter points
      virtual void SetTestStatistic(RooAbsArg&)  const {}  


      
   private:
      Double_t fSize;
      RooWorkspace* fWS; // a workspace that owns all the components to be used by the calculator
      Bool_t fOwnsWorkspace; // flag if this object owns its workspace
      const char* fPdfName; // name of  common PDF in workspace
      const char* fDataName; // name of data set in workspace
      RooArgSet* fPOI; // RooArgSet specifying  parameters of interest for interval
      RooArgSet* fNuisParams;// RooArgSet specifying  nuisance parameters for interval
      TestStatistic* fTestStat;
      Int_t fNtoys;
      Int_t fNevents;
      Bool_t fExtended;
      TRandom* fRand;
      
      Int_t fCounter;

   protected:
      ClassDef(ToyMCSampler,1)   // A simple implementation of the TestStatSampler interface
	};
}


#endif
