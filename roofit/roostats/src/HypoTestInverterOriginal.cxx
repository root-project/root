// @(#)root/roostats:$Id$
// Author: Kyle Cranmer, Lorenzo Moneta, Gregory Schott, Wouter Verkerke
/*************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//_________________________________________________________________
/**
   HypoTestInverterOriginal class for performing an hypothesis test inversion by scanning the hypothesis test results of the 
  HybridCalculator  for various values of the parameter of interest. By looking at the confidence level curve of 
 the result  an upper limit, where it intersects the desired confidence level, can be derived.
 The class implements the RooStats::IntervalCalculator interface and returns an  RooStats::HypoTestInverterResult class.
 The result is a SimpleInterval, which via the method UpperLimit returns to the user the upper limit value.

The  HypoTestInverterOriginal implements various option for performing the scan. HypoTestInverterOriginal::RunFixedScan will scan using a fixed grid the parameter of interest. HypoTestInverterOriginal::RunAutoScan will perform an automatic scan to find optimally the curve and it will stop until the desired precision is obtained.
The confidence level value at a given point can be done via  HypoTestInverterOriginal::RunOnePoint.
The class can scan the CLs+b values or alternativly CLs (if the method HypoTestInverterOriginal::UseCLs has been called).


   New contributions to this class have been written by Matthias Wolf (advanced AutoRun algorithm)
**/

// include other header files

#include "RooAbsPdf.h"
#include "RooAbsData.h"
#include "RooRealVar.h"
#include "TMath.h"

#include "RooStats/HybridCalculatorOriginal.h"
#include "RooStats/HybridResult.h"

// include header file of this class 
#include "RooStats/HypoTestInverterOriginal.h"


ClassImp(RooStats::HypoTestInverterOriginal)

using namespace RooStats;
using namespace std;


HypoTestInverterOriginal::HypoTestInverterOriginal( ) :
   fCalculator0(0),
   fScannedVariable(0),
   fResults(0),
   fUseCLs(false),
   fSize(0)
{
  // default constructor (doesn't do anything) 
}


HypoTestInverterOriginal::HypoTestInverterOriginal( HypoTestCalculator& myhc0,
				    RooRealVar& scannedVariable, double size ) :
   TNamed( ),
   fCalculator0(&myhc0),
   fScannedVariable(&scannedVariable), 
   fResults(0),
   fUseCLs(false),
   fSize(size)
{
   // constructor from a reference to an HypoTestCalculator 
   // (it must be an HybridCalculator type) and a RooRealVar for the variable
   SetName("HypoTestInverterOriginal");


   HybridCalculatorOriginal * hc = dynamic_cast<HybridCalculatorOriginal *> (fCalculator0);
   if (hc == 0) { 
      Fatal("HypoTestInverterOriginal","Using non HybridCalculatorOriginal class IS NOT SUPPORTED");
   }

}


HypoTestInverterOriginal::~HypoTestInverterOriginal()
{
  // destructor
  
  // delete the HypoTestInverterResult
  if (fResults) delete fResults;
}

void  HypoTestInverterOriginal::CreateResults() { 
  // create a new HypoTestInverterResult to hold all computed results
   if (fResults == 0) {
      TString results_name = this->GetName();
      results_name += "_results";
      fResults = new HypoTestInverterResult(results_name,*fScannedVariable,ConfidenceLevel());
      fResults->SetTitle("HypoTestInverterOriginal Result");
   }
   fResults->UseCLs(fUseCLs);
}


bool HypoTestInverterOriginal::RunAutoScan( double xMin, double xMax, double target, double epsilon, unsigned int numAlgorithm  )
{
  /// Search for the value of the parameter of interest (vary the
  /// hypothesis being tested) in the specified range [xMin,xMax]
  /// until the confidence level is compatible with the target value
  /// within one time the estimated error (and the estimated error
  /// should also become smaller than the specified parameter epsilon)

  // various sanity checks on the input parameters
  if ( xMin>=xMax || xMin< fScannedVariable->getMin() || xMax>fScannedVariable->getMax() ) {
    std::cout << "Error: problem with the specified range\n";
    return false;
  }
  if ( target<=0 || target>=1 ) {
    std::cout << "Error: problem with target value\n";
    return false;
  }
  if ( epsilon>0.5-fabs(0.5-target) ) {
    std::cout << "Error: problem with error value\n";
    return false;
  }
  if ( numAlgorithm!=0 && numAlgorithm!=1 ) {
    std::cout << "Error: invalid interpolation algorithm\n";
    return false;
  }

  CreateResults();

  // if ( TMath::AreEqualRel(target,1-Size()/2,DBL_EPSILON) ) {  // to uncomment for ROOT 5.26
  if ( fabs(1-target/(1-Size()/2))<DBL_EPSILON ) {
    fResults->fInterpolateLowerLimit = false;
    std::cout << "Target matches lower limit: de-activate interpolation in HypoTestInverterResult\n";
  }
  // if ( TMath::AreEqualRel(target,Size()/2,DBL_EPSILON) ) {  // to uncomment for ROOT 5.26
  if ( fabs(1-target/((Size()/2)))<DBL_EPSILON ) {
    fResults->fInterpolateUpperLimit = false;
    std::cout << "Target matches upper limit: de-activate interpolation in HypoTestInverterResult\n";
  }

  // parameters of the algorithm that are hard-coded
  const double nSigma = 1; // number of times the estimated error the final p-value should be from the target

  // backup some values to be restored at the end 
  const unsigned int nToys_backup = ((HybridCalculatorOriginal*)fCalculator0)->GetNumberOfToys();

  // check the 2 hypothesis tests specified as extrema in the constructor
  double leftX = xMin;
  if (!RunOnePoint(leftX)) return false;
  double leftCL = fResults->GetYValue(fResults->ArraySize()-1);
  double leftCLError = fResults->GetYError(fResults->ArraySize()-1);
 
  double rightX = xMax;
  if (!RunOnePoint(rightX)) return false;
  double rightCL = fResults->GetYValue(fResults->ArraySize()-1);
  double rightCLError = fResults->GetYError(fResults->ArraySize()-1);
  
  if ( rightCL>target && leftCL>target ) {
    std::cout << "The confidence level at both boundaries are both too large ( " << leftCL << " and " <<  rightCL << std::endl << "Run again with other boundaries or larger toy-MC statistics\n";
    return false;
  }
  if ( rightCL<target && leftCL<target ) {
    std::cout << "The confidence level at both boundaries are both too small ( " << leftCL << " and " <<  rightCL << std::endl << "Run again with other boundaries or larger toy-MC statistics\n";
    return false;
  }

  unsigned int nIteration = 2;  // number of iteration performed by the algorithm
  bool quitThisLoop = false;  // flag to interrupt the search and quit cleanly

  double centerCL = 0;
  double centerCLError = 0;

  // search for the value of the searched variable where the CL is
  // within 1 sigma of the desired level and sigma smaller than
  // epsilon.
  do {
    double x = 0;

    // safety checks
    if (leftCL==rightCL) {
      std::cout << "This cannot (and should not) happen... quit\n";
      quitThisLoop = true;
    } else if (leftX==rightX) {
      std::cout << "This cannot (and should not) happen... quit\n";
      quitThisLoop = true;
    } else {

      // apply chosen type of interpolation algorithm
      if (numAlgorithm==0) {
	// exponential interpolation

	// add safety checks
	if (!leftCL) leftCL = DBL_EPSILON;
	if (!rightCL) rightCL = DBL_EPSILON;

	double a = (log(leftCL) - log(rightCL)) / (leftX - rightX);
	double b = leftCL / exp(a * leftX);
	x = (log(target) - log(b)) / a;

	// to do: do not allow next iteration outside the xMin,xMax interval
	if (x<xMin || x>xMax || TMath::IsNaN(x)) {
	  std::cout << "Extrapolated value out of range or nan: exits\n";
	  quitThisLoop = true;
	}
      } else if (numAlgorithm==1) {
	// linear interpolation
	
	double a = (leftCL-rightCL)/(leftX-rightX);
	double b = leftCL-a*leftX;
	x = (target-b)/a;

	if (x<xMin || x>xMax || TMath::IsNaN(x)) {
	  std::cout << "Extrapolated value out of range or nan: exits\n";
	  quitThisLoop = true;
	}
      }  // end of interpolation algorithms
    }

    if ( x==leftX || x==rightX ) {
      std::cout << "Error: exit because interpolated value equals to a previous iteration\n";
      quitThisLoop = true;
    }

    // perform another hypothesis-test for value x
    bool success = false;
    if (!quitThisLoop) success = RunOnePoint(x);

    if (success) {

      nIteration++;  // succeeded, increase the iteration counter
      centerCL = fResults->GetYValue(fResults->ArraySize()-1);
      centerCLError = fResults->GetYError(fResults->ArraySize()-1);

      // replace either the left or right point by this new point
    
      // test if the interval points are on different sides, then
      // replace the one on the correct side with the center
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

      // if a point is found compatible with the target CL but with too
      // large error, increase the number of toyMC
      if ( fabs(centerCL-target) < nSigma*centerCLError && centerCLError > epsilon  ) {
	do {

	  int nToys = ((HybridCalculatorOriginal*)fCalculator0)->GetNumberOfToys();  // current number of toys
	  int nToysTarget = (int) TMath::Max(nToys*1.5, 1.2*nToys*pow(centerCLError/epsilon,2));  // estimated number of toys until the target precision is reached
	  
	  std::cout << "Increasing the number of toys to: " << nToysTarget << std::endl;
	  
	  // run again the same point with more toyMC (run the complement number of toys)
	  ((HybridCalculatorOriginal*)fCalculator0)->SetNumberOfToys(nToysTarget-nToys);
	  
	  if (!RunOnePoint(x)) quitThisLoop=true;
	  nIteration++;  // succeeded, increase the iteration counter
	  centerCL = fResults->GetYValue(fResults->ArraySize()-1);
	  centerCLError = fResults->GetYError(fResults->ArraySize()-1);
	  
	  // set the number of toys to reach the target 
	  ((HybridCalculatorOriginal*)fCalculator0)->SetNumberOfToys(nToysTarget);
	} while ( fabs(centerCL-target) < nSigma*centerCLError && centerCLError > epsilon && quitThisLoop==false );  // run this block again if it's still compatible with the target and the error still too large
      }
      
      if (leftCL==rightCL) {
	std::cout << "Algorithm failed: left and right CL are equal (no intrapolation possible or more toy-MC statistics needed)\n";
	  quitThisLoop = true;
      }
    } // end running one more iteration

  } while ( ( fabs(centerCL-target) > nSigma*centerCLError || centerCLError > epsilon ) && quitThisLoop==false );  // end of the main 'do' loop 

  // restore some parameters that might have been changed by the algorithm
  ((HybridCalculatorOriginal*)fCalculator0)->SetNumberOfToys(nToys_backup);
  
  if ( quitThisLoop==true ) {
    // abort and return 'false' to indicate fail status
    std::cout << "Aborted the search because something happened\n";
    return false;
  }

  std::cout << "Converged in " << fResults->ArraySize() << " iterations\n";

  // finished: return 'true' for success status
  return true;
}


bool HypoTestInverterOriginal::RunFixedScan( int nBins, double xMin, double xMax )
{
   // Run a Fixed scan in npoints between min and max

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
      std::cout << "Loop interrupted because of failed status\n";
      return false;
    }
  }

  return true;
}


bool HypoTestInverterOriginal::RunOnePoint( double thisX )
{
   // run only one point 

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
   
   double lastXtested;
   if ( fResults->ArraySize()!=0 ) lastXtested = fResults->GetXValue(fResults->ArraySize()-1);
   else lastXtested = -999;

   if ( lastXtested==thisX ) {
     
     std::cout << "Merge with previous result\n";
     HybridResult* latestResult = (HybridResult*) fResults->GetResult(fResults->ArraySize()-1);
     latestResult->Add((HybridResult*)myHybridResult);
     delete myHybridResult;

   } else {
     
     // fill the results in the HypoTestInverterResult array
     fResults->fXValues.push_back(thisX);
     fResults->fYObjects.Add(myHybridResult);
   }


   std::cout << "computed: " << fResults->GetYValue(fResults->ArraySize()-1) << endl;

   fScannedVariable->setVal(oldValue);
   
   return true;
}
