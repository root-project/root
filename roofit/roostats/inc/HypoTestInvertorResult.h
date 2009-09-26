// @(#)root/roostats:$Id: SimpleInterval.h 30478 2009-09-25 19:42:07Z schott $
// Author: Kyle Cranmer, Lorenzo Moneta, Gregory Schott, Wouter Verkerke
/*************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOSTATS_HypoTestInvertorResult
#define ROOSTATS_HypoTestInvertorResult



#ifndef ROOSTATS_SimpleInterval
#include "RooStats/SimpleInterval.h"
#endif

class RooRealVar;

namespace RooStats {

  class HypoTestInvertorResult : public SimpleInterval /*, public TNamed*/ {

  public:

    // constructor
    HypoTestInvertorResult( const char* name, 
			    const char* title,
			    RooRealVar* scannedVariable,
			    double cl ) ;

    // destructor
    virtual ~HypoTestInvertorResult() ; // TO DO DELETE ALL yOBJECTS

    // function to return the yValue

    //SimpleInterval* GetInterval() const ; 

    double GetXValue( int index ) const ;

    double GetYValue( int index ) const ;

    int Size() const { return fXValues.size(); };


    Double_t LowerLimit() { CalculateLimits(); return fLowerLimit; }
    Double_t UpperLimit() { CalculateLimits(); return fUpperLimit; }

  private:

    void CalculateLimits() ;

/*     Double_t fLowerLimit;         // lower limit on the constrained variable */
/*     Double_t fUpperLimit;         // upper limit on the constrained variable */
/*     RooAbsArg* fScannedVariable;  // pointer to the constrained variable */
    
  protected:

     std::vector<double> fXValues;

     TList fYObjects;

     friend class HypoTestInvertor;

     ClassDef(HypoTestInvertorResult,1)  // HypoTestInvertorResult class

  };
}

#endif
