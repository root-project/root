// @(#)root/roostats:$Id$
// Author: Kyle Cranmer, Lorenzo Moneta, Gregory Schott, Wouter Verkerke
/*************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOSTATS_HypoTestInverterOriginal
#define ROOSTATS_HypoTestInverterOriginal



#ifndef ROOSTATS_IntervalCalculator
#include "RooStats/IntervalCalculator.h"
#endif

#ifndef ROOSTATS_HypoTestCalculator
#include "RooStats/HypoTestCalculator.h"
#endif

#ifndef  ROOSTATS_HypoTestInverterResult
#include "RooStats/HypoTestInverterResult.h"
#endif

class RooRealVar;


namespace RooStats {

   /**
     
   This class is now depratcated and to be replaced by the HypoTestInverter. 
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

  class HypoTestInverterOriginal : public IntervalCalculator, public TNamed {

  public:

    // default constructor (used only for I/O)
    HypoTestInverterOriginal();


    // constructor
    HypoTestInverterOriginal( HypoTestCalculator& myhc0,
		      RooRealVar& scannedVariable, 
                      double size = 0.05) ;

     

    virtual HypoTestInverterResult* GetInterval() const { return fResults; } ;

    bool RunAutoScan( double xMin, double xMax, double target, double epsilon=0.005, unsigned int numAlgorithm=0 );

    bool RunFixedScan( int nBins, double xMin, double xMax );

    bool RunOnePoint( double thisX );

    void UseCLs( bool on = true) { fUseCLs = on; if (fResults) fResults->UseCLs(on);   }

    virtual void  SetData(RooAbsData &) { } // not needed

    virtual void SetModel(const ModelConfig &) { } // not needed 

    // set the size of the test (rate of Type I error) ( Eg. 0.05 for a 95% Confidence Interval)
     virtual void SetTestSize(Double_t size) {fSize = size; if (fResults) fResults->SetTestSize(size); }
    // set the confidence level for the interval (eg. 0.95 for a 95% Confidence Interval)
    virtual void SetConfidenceLevel(Double_t cl) {fSize = 1.-cl;  if (fResults) fResults->SetConfidenceLevel(cl); }
    // Get the size of the test (eg. rate of Type I error)
    virtual Double_t Size() const {return fSize;}
    // Get the Confidence level for the test
    virtual Double_t ConfidenceLevel()  const {return 1.-fSize;}
 
    // destructor
    virtual ~HypoTestInverterOriginal() ;

  private:

    void CreateResults(); 

    HypoTestCalculator* fCalculator0;   // pointer to the calculator passed in the constructor
    RooRealVar* fScannedVariable;     // pointer to the constrained variable
    HypoTestInverterResult* fResults;

    bool fUseCLs;
    double fSize;

  protected:

    ClassDef(HypoTestInverterOriginal,1)  // HypoTestInverterOriginal class

  };
}

#endif
