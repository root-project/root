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


namespace RooStats {
 class SimpleInterval : public ConfInterval {
  protected:
    RooArgSet* fParameters; // parameter of interest
    Double_t fLowerLimit; // lower limit
    Double_t fUpperLimit; // upper limit
    Double_t fConfidenceLevel; // confidence level

  public:
    // constructors,destructors
    SimpleInterval();
    SimpleInterval(const char* name);
    SimpleInterval(const char* name, const char* title);
    SimpleInterval(const char* name, RooAbsArg* var, Double_t, Double_t);
    SimpleInterval(const char* name, const char* title, RooAbsArg* var, Double_t, Double_t);
    virtual ~SimpleInterval();
        
    virtual Bool_t IsInInterval(const RooArgSet&);
    virtual void SetConfidenceLevel(Double_t cl) {fConfidenceLevel = cl;}
    virtual Double_t ConfidenceLevel() const {return fConfidenceLevel;}
 
    // Method to return lower limit
    virtual Double_t LowerLimit() {return fLowerLimit;}
    // Method to return upper limit
    virtual Double_t UpperLimit() {return fUpperLimit;}
    
    // do we want it to return list of parameters
    virtual RooArgSet* GetParameters() const;

    // check if parameters are correct. (dummy implementation to start)
    Bool_t CheckParameters(const RooArgSet&) const ;


    
  protected:
    ClassDef(SimpleInterval,1)  // Concrete implementation of ConfInterval for simple 1-D intervals in the form [a,b]
      
  };
}

#endif
