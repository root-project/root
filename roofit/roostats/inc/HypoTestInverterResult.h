// @(#)root/roostats:$Id$
// Author: Kyle Cranmer, Lorenzo Moneta, Gregory Schott, Wouter Verkerke
/*************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOSTATS_HypoTestInverterResult
#define ROOSTATS_HypoTestInverterResult



#ifndef ROOSTATS_SimpleInterval
#include "RooStats/SimpleInterval.h"
#endif

class RooRealVar;

namespace RooStats {

  class HypoTestInverterResult : public SimpleInterval {

  public:

    // default constructor
    explicit HypoTestInverterResult(const char* name = 0);

    // constructor
    HypoTestInverterResult( const char* name, 
			    const RooRealVar& scannedVariable,
			    double cl ) ;

    // destructor
    virtual ~HypoTestInverterResult();

    // function to return the yValue

    double GetXValue( int index ) const ;

    double GetYValue( int index ) const ;
    double GetYError( int index ) const ;

    int Size() const { return fXValues.size(); };

    // set the size of the test (rate of Type I error) ( Eg. 0.05 for a 95% Confidence Interval)
    virtual void SetTestSize(Double_t size) {fConfidenceLevel = 1.-size;  }
    // set the confidence level for the interval (eg. 0.95 for a 95% Confidence Interval)
    virtual void SetConfidenceLevel(Double_t cl) {fConfidenceLevel = cl;  }

    void UseCLs(bool on = true) { fUseCLs = on; }  

    Double_t LowerLimit() { CalculateLimits(); return fLowerLimit; }
    Double_t UpperLimit() { CalculateLimits(); return fUpperLimit; }

    Double_t UpperLimitEstimatedError();

  private:

    void CalculateLimits() ;

  protected:

    bool fUseCLs; 
    bool fInterpolate;
    Double_t fUpperLimitError;

    std::vector<double> fXValues;

    TList fYObjects;

    friend class HypoTestInverter;

    ClassDef(HypoTestInverterResult,1)  // HypoTestInverterResult class      
  };
}

#endif
