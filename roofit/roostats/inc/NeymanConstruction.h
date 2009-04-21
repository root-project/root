// @(#)root/roostats:$Id: NeymanConstruction.h 26805 2009-01-13 17:45:57Z cranmer $
// Author: Kyle Cranmer, Lorenzo Moneta, Gregory Schott, Wouter Verkerke
/*************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOSTATS_NeymanConstruction
#define ROOSTATS_NeymanConstruction


#ifndef ROOT_Rtypes
#include "Rtypes.h"
#endif

#ifndef ROOSTATS_IntervalCalculator
#include "RooStats/IntervalCalculator.h"
#endif

#include "RooStats/TestStatSampler.h"
#include "RooStats/ConfidenceBelt.h"

#include "RooTreeData.h"
#include "RooWorkspace.h"
#include "RooAbsPdf.h"
#include "RooArgSet.h"
#include "TList.h"

class RooAbsData; 

namespace RooStats {

   class ConfInterval; 

 class NeymanConstruction : public IntervalCalculator {

   public:
     NeymanConstruction();
     virtual ~NeymanConstruction() {}
    
      // Main interface to get a ConfInterval (will be a PointSetInterval)
      virtual ConfInterval* GetInterval() const;
      virtual ConfInterval* GetIntervalUsingList() const;

      // Interface extended with I/O support
      virtual ConfInterval* GetInterval(const char* asciiFilePat) const;

      // Actually generate teh sampling distribution
      virtual TList*        GenSamplingDistribution(const char* asciiFilePat = 0) const; 
      virtual ConfInterval* Run(TList *SamplingList) const;

      // in addition to interface we also need:
      // Set the TestStatSampler (eg. ToyMC or FFT, includes choice of TestStatistic)
      void SetTestStatSampler(TestStatSampler& distCreator) {fTestStatSampler = &distCreator;}
      // fLeftSideTailFraction*fSize defines lower edge of acceptance region.
      // Unified limits use 0, central limits use 0.5, 
      // for upper/lower limits it is 0/1 depends on sign of test statistic w.r.t. parameter
      void SetLeftSideTailFraction(Double_t leftSideFraction = 0.) {fLeftSideFraction = leftSideFraction;} 

      // User-defined set of points to test
      void SetParameterPointsToTest(RooTreeData& pointsToTest) {
	fPointsToTest = &pointsToTest;
        fConfBelt = new ConfidenceBelt("ConfBelt",pointsToTest);
      }
      // This class can make regularly spaced scans based on range stored in RooRealVars.
      // Choose number of steps for a rastor scan (common for each dimension)
      //      void SetNumSteps(Int_t);
      // This class can make regularly spaced scans based on range stored in RooRealVars.
      // Choose number of steps for a rastor scan (specific for each dimension)
      //      void SetNumSteps(map<RooAbsArg, Int_t>)

      // Get the size of the test (eg. rate of Type I error)
      virtual Double_t Size() const {return fSize;}
      // Get the Confidence level for the test
      virtual Double_t ConfidenceLevel()  const {return 1.-fSize;}  
      // set a workspace that owns all the necessary components for the analysis
      virtual void SetWorkspace(RooWorkspace& ws) {fWS = &ws;}

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


      ConfidenceBelt* GetConfidenceBelt() {return fConfBelt;}


      void UseAdaptiveSampling(bool flag=true){fAdaptiveSampling=flag;}

      
   private:

      Double_t fSize; // size of the test (eg. specified rate of Type I error)
      RooWorkspace* fWS; // a workspace that owns all the components to be used by the calculator
      Bool_t fOwnsWorkspace; // flag if this object owns its workspace
      const char* fPdfName; // name of  common PDF in workspace
      const char* fDataName; // name of data set in workspace
      RooArgSet* fPOI; // RooArgSet specifying  parameters of interest for interval
      RooArgSet* fNuisParams;// RooArgSet specifying  nuisance parameters for interval
      TestStatSampler* fTestStatSampler;
      RooTreeData* fPointsToTest;
      Double_t fLeftSideFraction;
      ConfidenceBelt* fConfBelt;
      bool fAdaptiveSampling; // controls use of adaptive sampling algorithm

   protected:
      ClassDef(NeymanConstruction,1)   // Interface for tools setting limits (producing confidence intervals)
   };
}


#endif
