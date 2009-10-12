// @(#)root/roostats:$Id: SimpleInterval.h 30478 2009-09-25 19:42:07Z schott $
// Author: Kyle Cranmer, Lorenzo Moneta, Gregory Schott, Wouter Verkerke
/*************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/**
   HypoTestInvertor class

**/

// include other header files

#include "RooAbsPdf.h"
#include "RooAbsData.h"
#include "RooRealVar.h"

#include "RooStats/HybridCalculator.h"
#include "RooStats/HybridResult.h"

// include header file of this class 
#include "RooStats/HypoTestInvertor.h"


ClassImp(RooStats::HypoTestInvertor)

using namespace RooStats;


HypoTestInvertor::HypoTestInvertor( const char* name,
				    const char* title ) :
   TNamed( TString(name), TString(title) ), 
   fCalculator0(0),
   fScannedVariable(0),
   fResults(0)
{
  // default constructor (doesn't do anything)
}


HypoTestInvertor::HypoTestInvertor( const char* name,
				    const char* title,
				    HypoTestCalculator* myhc0,
				    RooRealVar* scannedVariable ) :
  TNamed( TString(name), TString(title) ),
  fCalculator0(myhc0),
  fScannedVariable(scannedVariable)
{
  // constructor

  //if (myhc0->ClassName()!="HybridCalculator") std::cout << "NOT SUPPORTED\n";

  // create a new HypoTestInvertorResult to hold all computed results
  TString results_name = this->GetName();
  results_name += "_results";
  fResults = new HypoTestInvertorResult(results_name,"HypoTestInvertorResult",scannedVariable,ConfidenceLevel());

}


HypoTestInvertor::~HypoTestInvertor()
{
  // destructor
  
  // delete the HypoTestInvertorResult
   if (fResults) delete fResults;
}


bool HypoTestInvertor::RunAutoScan( double xMin, double xMax, double epsilon )
{
  double target = Size();
  bool status;

  double xLow = xMin;
  status = RunOnePoint(xMin);
  if ( !status ) return false;
  double yLow = fResults->GetYValue(fResults->Size()-1);

  double xHigh = xMax;
  status = RunOnePoint(xMax);
  if ( !status ) return false;
  double yHigh = fResults->GetYValue(fResults->Size()-1);

  // after the initial points, check the point in the middle of the last two 
  // closest points, until the value is in the range target+/-epsilon
  // the final value becomes the inverted upper limit value

  bool stopNow = false;
  while ( !stopNow || fResults->Size()==30 ) {
    double xMiddle = xLow+0.5*(xHigh-xLow);
    status = RunOnePoint(xMiddle);
    if ( !status ) return false;
    double yMiddle = fResults->GetYValue(fResults->Size()-1);
    if (fabs(yMiddle-target)<epsilon) stopNow = true;
    else if ( yMiddle>target && yHigh<target ) { xLow = xMiddle; yLow = yMiddle; }
    else if ( yMiddle<target && yHigh<target ) { xHigh = xMiddle; yHigh = yMiddle; }
    else if ( yMiddle<target && yHigh>target ) { xLow = xMiddle; yLow = yMiddle; }
    else if ( yMiddle>target && yHigh>target ) { xHigh = xMiddle; yHigh = yMiddle; }
  }

  std::cout << "Converged in " << fResults->Size() << " iterations\n";

  return true;
}


bool HypoTestInvertor::RunFixedScan( int nBins, double xMin, double xMax )
{
  // safety checks
  if ( nBins<=0 ) {
    std::cout << "Please provide nBins>0\n";
    return false;
  }
  if ( nBins==1 && xMin!=xMax ) {
    std::cout << "nBins==1 -> I will run for xMin (" << xMin << ")\n";
  }
  if ( xMin==xMax && nBins>1 ) { 
    std::cout << "xMin==xMax -> I will enforce nBins==1\n";
    nBins = 1;
  }
  if ( xMin>xMax ) {
    std::cout << "Please provide xMin (" << xMin << ") smaller that xMax (" << xMax << ")\n";
    return false;
  } 
  
  for (int i=0; i<nBins; i++) {
    double thisX = xMin+i*(xMax-xMin)/(nBins-1);
    bool status = RunOnePoint(thisX);
    
    // check if failed status
    if ( status==false ) {
      std::cout << "Loop interupted because of failed status\n";
      return false;
    }
  }

  return true;
}


bool HypoTestInvertor::RunOnePoint( double thisX )
{
  // check if thisX is in the range specified for fScannedVariable
  if ( thisX<fScannedVariable->getMin() || thisX>fScannedVariable->getMax() ) {
    std::cout << "I will not run because the specified value in not in the range of the variable being scanned\n";
    return false;
  }

  std::cout << "Running for " << fScannedVariable->GetName() << " = " << thisX << endl;
  
  double oldValue = fScannedVariable->getVal();

  fScannedVariable->setVal(thisX);

  // create a clone of the HypoTestCalculator
  //HypoTestCalculator* calculator = (HypoTestCalculator*) ((HybridCalculator*) fCalculator0)->Clone(); // MAYBE WE DON'T NEED
  HypoTestCalculator* calculator = fCalculator0;
  
  // compute the results
  HypoTestResult* myHybridResult =  calculator->GetHypoTest(); 

  // fill the results in the HypoTestInvertorResult
  fResults->fXValues.push_back(thisX);
  fResults->fYObjects.Add(myHybridResult); // TO DO vector of HypoTestResult

  //delete calculator;

  //std::cout << "DONE\n";

  fScannedVariable->setVal(oldValue);

  return true;
}
