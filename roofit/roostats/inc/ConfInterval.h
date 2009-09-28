// @(#)root/roostats:$Id$
// Author: Kyle Cranmer, Lorenzo Moneta, Gregory Schott, Wouter Verkerke
/*************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOSTATS_ConfInterval
#define ROOSTATS_ConfInterval

#ifndef ROOT_TNamed
#include "TNamed.h"
#endif

#ifndef ROO_ARG_SET
#include "RooArgSet.h"
#endif

//_________________________________________________________________
//
// BEGIN_HTML
// ConfInterval is an interface class for a generic interval in the RooStats framework.
// Any tool inheriting from IntervalCalculator can return a ConfInterval.
// There are many types of intervals, they may be a simple range [a,b] in 1 dimension,
// or they may be disconnected regions in multiple dimensions.
// So the common interface is simply to ask the interval if a given point "IsInInterval".
// The Interval also knows what confidence level it was constructed at and the space of 
// parameters for which it was constructed.
// Note, one could use the same class for a Bayesian "credible interval".
// END_HTML
//
//


namespace RooStats {

   class ConfInterval : public TNamed {

   public:

      ConfInterval() : TNamed() {} 
      ConfInterval(const char* name) :  TNamed(name,name) {}
      ConfInterval(const char* name, const char* title) : TNamed(name,title) {} 
      virtual ~ConfInterval() {}
    
      //pure virtual?  where does =0 go with const?
      virtual Bool_t IsInInterval(const RooArgSet&) = 0; 
    
      // used to set confidence level.  Keep pure virtual
      virtual void SetConfidenceLevel(Double_t cl) = 0;
      // return confidence level
      virtual Double_t ConfidenceLevel() const = 0;
      // 
      // if so does this implement it?
      // private fSize;
    

      // do we want it to return list of parameters
      virtual RooArgSet* GetParameters() const = 0;

      // check if parameters are correct. (dummy implementation to start)
      virtual Bool_t CheckParameters(const RooArgSet&) const = 0;


   protected:
      ClassDef(ConfInterval,1) // Interface for Confidence Intervals

   };
}


#endif
