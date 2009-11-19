// @(#)root/roostats:$Id$
// Author: Kyle Cranmer, Lorenzo Moneta, Gregory Schott, Wouter Verkerke
/*************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/**
   HypoTestInverter class

   New contributions to this class have been written by Matthias Wolf (advanced AutoRun algorithm)
**/

// include other header files

#include "RooAbsPdf.h"
#include "RooAbsData.h"
#include "RooRealVar.h"

#include "RooStats/HybridCalculator.h"
#include "RooStats/HybridResult.h"

// include header file of this class 
#include "RooStats/HypoTestInverter.h"


ClassImp(RooStats::HypoTestInverter)

using namespace RooStats;


HypoTestInverter::HypoTestInverter( ) :
   fCalculator0(0),
   fScannedVariable(0),
   fResults(0),
   fUseCLs(false),
   fSize(0)
{
  // default constructor (doesn't do anything) 
}


HypoTestInverter::HypoTestInverter( const char* name,
				    HypoTestCalculator* myhc0,
				    RooRealVar* scannedVariable, double size ) :
  TNamed( name, name ),
  fCalculator0(myhc0),
  fScannedVariable(scannedVariable), 
  fResults(0),
  fUseCLs(false),
  fSize(size)
{
  // constructor
  if (name==0) SetName("HypoTestInverter");

  //if (myhc0->ClassName()!="HybridCalculator") std::cout << "NOT SUPPORTED\n";

}


HypoTestInverter::~HypoTestInverter()
{
  // destructor
  
  // delete the HypoTestInverterResult
  if (fResults) delete fResults;
}

void  HypoTestInverter::CreateResults() { 
  // create a new HypoTestInverterResult to hold all computed results
   if (fResults == 0) {
      TString results_name = this->GetName();
      results_name += "_results";
      fResults = new HypoTestInverterResult(results_name,*fScannedVariable,ConfidenceLevel());
      fResults->SetTitle("HypoTestInverter Result");
   }
   fResults->UseCLs(fUseCLs);
}


bool HypoTestInverter::RunAutoScan( double xMin, double xMax, double target, double epsilon, int numAlgorithm  )
{

  CreateResults();
  if ( target==Size()/2 ) fResults->fInterpolateLowerLimit = false;
  if ( target==(1-Size()/2) ) fResults->fInterpolateUpperLimit = false;

  if (numAlgorithm==0) {

    // Search for the value of the searched variable where the CL is within
    // 1 sigma of the desired level and sigma smaller than epsilon.  This is done by consecutively replacing
    // the worst value of the interval with one that has been interpolated
    // exponentially.

    double nSigma = 1;

    double leftX = xMin;
    double rightX = xMax;
    if (!RunOnePoint(leftX)) return false;
    double leftCL = fResults->GetYValue(fResults->ArraySize()-1);
    double leftCLError = fResults->GetYError(fResults->ArraySize()-1);
    if (!RunOnePoint(rightX)) return false;
    double rightCL = fResults->GetYValue(fResults->ArraySize()-1);
    double rightCLError = fResults->GetYError(fResults->ArraySize()-1);
    double centerCL;
    double centerCLError;

    do {
      if (!leftCL) leftCL = DBL_EPSILON;
      if (!rightCL) rightCL = DBL_EPSILON;

      double a = (log(leftCL) - log(rightCL)) / (leftX - rightX);
      double b = leftCL / exp(a * leftX);
      double x = (log(target) - log(b)) / a;
      if (isnan(x)) {
	std::cout << "ERROR: Failed the auto run for finding target\n";
	return false;
      }

      if (!RunOnePoint(x)) return false;
      centerCL = fResults->GetYValue(fResults->ArraySize()-1);
      centerCLError = fResults->GetYError(fResults->ArraySize()-1);

      // Test if the interval points are on different sides, then replace the
      // one on the "right" side with the center
      if ( (leftCL > target) == (rightCL < target) ) {
	if ( (centerCL > target) == (leftCL > target) ) {
	  leftX = x;
	  leftCL = centerCL;
	  leftCLError = centerCLError;
	} else {
	  rightX = x;
	  rightCL = centerCL;
	  rightCLError = centerCLError;
	}
	// Otherwise replace the point farest away from target (measured in
	// sigmas)
      } else if ( (fabs(leftCL - target) / leftCLError) >
		  (fabs(rightCL - target) / rightCLError) ) {
	leftX = x;
	leftCL = centerCL;
	leftCLError = centerCLError;
      } else {
	rightX = x;
	rightCL = centerCL;
	rightCLError = centerCLError;
      }
      if ( fabs(centerCL-target) < nSigma*centerCLError && centerCLError > epsilon  ) {
// 	do {
	// add statistics to the number of toys to gain precision

	  int nToys = ((HybridCalculator*)fCalculator0)->GetNumberOfToys(); // current number of toys
	  int nToysTarget = (int) TMath::Max(nToys*1.5, 1.2*nToys*pow(centerCLError/epsilon,2)); // estimated number of toys until the target precision is reached
	  ((HybridCalculator*)fCalculator0)->SetNumberOfToys(nToysTarget);
      
	  std::cout << "Increasing the number of toys to: " << nToysTarget << " (CL error was: " << centerCLError << ")\n";

// 	  centerCL = fResults->GetYValue(fResults->ArraySize()-1);
// 	  centerCLError = fResults->GetYValue(fResults->ArraySize()-1);

//  	} while ( fabs(centerCL-target) < nSigma*centerCLError && centerCLError > epsilon )
       }

    } while ( fabs(centerCL-target) > nSigma*centerCLError || centerCLError > epsilon );
    std::cout << "Converged in " << fResults->ArraySize() << " iterations\n";
    return true;
  } else if ( numAlgorithm==1 ) {
    // Newton search

    double xLow = xMin;
    bool status = RunOnePoint(xMin);
    if ( !status ) return false;
    double yLow = fResults->GetYValue(fResults->ArraySize()-1);

    double xHigh = xMax;
    status = RunOnePoint(xMax);
    if ( !status ) return false;
    double yHigh = fResults->GetYValue(fResults->ArraySize()-1);

    // after the initial points, check the point in the middle of the last two 
    // closest points, until the value is in the range target+/-epsilon
    // the final value becomes the inverted upper limit value

    bool stopNow = false;
    while ( !stopNow || fResults->ArraySize()==30 ) {
      double xMiddle = xLow+0.5*(xHigh-xLow);
      status = RunOnePoint(xMiddle);
      if ( !status ) return false;
      double yMiddle = fResults->GetYValue(fResults->ArraySize()-1);
      if (fabs(yMiddle-target)<epsilon) stopNow = true;
      else if ( yMiddle>target && yHigh<target ) { xLow = xMiddle; yLow = yMiddle; }
      else if ( yMiddle<target && yHigh<target ) { xHigh = xMiddle; yHigh = yMiddle; }
      else if ( yMiddle<target && yHigh>target ) { xLow = xMiddle; yLow = yMiddle; }
      else if ( yMiddle>target && yHigh>target ) { xHigh = xMiddle; yHigh = yMiddle; }
    }
    std::cout << "Converged in " << fResults->ArraySize() << " iterations\n";
    return true;
  } else {
    std::cout << "not valid algorithm option specified\n";
    return false;
  }
}


bool HypoTestInverter::RunFixedScan( int nBins, double xMin, double xMax )
{
   CreateResults();
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


bool HypoTestInverter::RunOnePoint( double thisX )
{
   CreateResults();

  // check if thisX is in the range specified for fScannedVariable
  if ( thisX<fScannedVariable->getMin() ) {
    std::cout << "Out of range: using the lower bound on the scanned variable rather than " << thisX<< "\n";
    thisX = fScannedVariable->getMin();
  }
  if ( thisX>fScannedVariable->getMax() ) {
    std::cout << "Out of range: using the upper bound on the scanned variable rather than " << thisX<< "\n";
    thisX = fScannedVariable->getMax();
  }

  double oldValue = fScannedVariable->getVal();

  fScannedVariable->setVal(thisX);
  std::cout << "Running for " << fScannedVariable->GetName() << " = " << thisX << endl;

  // compute the results
  HypoTestResult* myHybridResult = fCalculator0->GetHypoTest(); 

  // fill the results in the HypoTestInverterResult
  fResults->fXValues.push_back(thisX);
  fResults->fYObjects.Add(myHybridResult);

  //delete calculator;

  fScannedVariable->setVal(oldValue);

  return true;
}
