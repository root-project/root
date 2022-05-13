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



#include "RooStats/IntervalCalculator.h"

#include "RooStats/HypoTestCalculator.h"

#include "RooStats/HypoTestInverterResult.h"

class RooRealVar;


namespace RooStats {

  class HypoTestInverterOriginal : public IntervalCalculator, public TNamed {

  public:

    // default constructor (used only for I/O)
    HypoTestInverterOriginal();


    // constructor
    HypoTestInverterOriginal( HypoTestCalculator& myhc0,
            RooRealVar& scannedVariable,
                      double size = 0.05) ;



    HypoTestInverterResult* GetInterval() const override { return fResults; } ;

    bool RunAutoScan( double xMin, double xMax, double target, double epsilon=0.005, unsigned int numAlgorithm=0 );

    bool RunFixedScan( int nBins, double xMin, double xMax );

    bool RunOnePoint( double thisX );

    void UseCLs( bool on = true) { fUseCLs = on; if (fResults) fResults->UseCLs(on);   }

    void  SetData(RooAbsData &) override { } // not needed

    void SetModel(const ModelConfig &) override { } // not needed

    /// set the size of the test (rate of Type I error) ( Eg. 0.05 for a 95% Confidence Interval)
     void SetTestSize(double size) override {fSize = size; if (fResults) fResults->SetTestSize(size); }
    /// set the confidence level for the interval (eg. 0.95 for a 95% Confidence Interval)
    void SetConfidenceLevel(double cl) override {fSize = 1.-cl;  if (fResults) fResults->SetConfidenceLevel(cl); }
    /// Get the size of the test (eg. rate of Type I error)
    double Size() const override {return fSize;}
    /// Get the Confidence level for the test
    double ConfidenceLevel()  const override {return 1.-fSize;}

    /// destructor
    ~HypoTestInverterOriginal() override ;

  private:

    void CreateResults();

    HypoTestCalculator* fCalculator0; ///< pointer to the calculator passed in the constructor
    RooRealVar* fScannedVariable;     ///< pointer to the constrained variable
    HypoTestInverterResult* fResults;

    bool fUseCLs;
    double fSize;

  protected:

    ClassDefOverride(HypoTestInverterOriginal,1)  // HypoTestInverterOriginal class

  };
}

#endif
