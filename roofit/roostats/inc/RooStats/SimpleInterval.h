// @(#)root/roostats:$Id$
// Author: Kyle Cranmer, Lorenzo Moneta, Gregory Schott, Wouter Verkerke
/*************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef RooStats_SimpleInterval
#define RooStats_SimpleInterval

#ifndef ROO_ARG_SET
#include "RooArgSet.h"
#endif
#ifndef RooStats_ConfInterval
#include "RooStats/ConfInterval.h"
#endif

class RooRealVar; 

namespace RooStats {
 class SimpleInterval : public ConfInterval {

  public:
    // default constructors
    explicit SimpleInterval(const char* name = 0);

    SimpleInterval(const SimpleInterval& other, const char* name);

    SimpleInterval& operator=(const SimpleInterval& other) ;

    // constructor from name, the Parameter of interest and lower/upper bound values 
    SimpleInterval(const char* name, const RooRealVar & var, Double_t lower, Double_t upper, Double_t cl);

    // destructor
    virtual ~SimpleInterval();
       
    // check if parameter is in the interval
    virtual Bool_t IsInInterval(const RooArgSet&) const;

    // set the confidence level for the interval. Simple interval is defined at construction time so this function 
    // has no effect
    virtual void SetConfidenceLevel(Double_t ) {}

    // return the confidence interval 
    virtual Double_t ConfidenceLevel() const {return fConfidenceLevel;}
 
    // return the interval lower limit 
    virtual Double_t LowerLimit() {return fLowerLimit;}
    // return the interval upper limit
    virtual Double_t UpperLimit() {return fUpperLimit;}
    
    // return a cloned list with the parameter of interest
    virtual RooArgSet* GetParameters() const;

    // check if parameters are correct (i.e. they are the POI of this interval)
    Bool_t CheckParameters(const RooArgSet&) const ;


    
  protected:

    ClassDef(SimpleInterval,1)  // Concrete implementation of ConfInterval for simple 1-D intervals in the form [a,b]

    RooArgSet fParameters; // set containing the parameter of interest
    Double_t  fLowerLimit; // lower interval limit
    Double_t  fUpperLimit; // upper interval limit
    Double_t  fConfidenceLevel; // confidence level
      
  };
}

#endif
