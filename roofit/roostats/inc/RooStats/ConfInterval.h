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

#include "TNamed.h"

#include "RooArgSet.h"

namespace RooStats {


/** \class ConfInterval
    \ingroup Roostats

ConfInterval is an interface class for a generic interval in the RooStats framework.
Any tool inheriting from IntervalCalculator can return a ConfInterval.
There are many types of intervals, they may be a simple range [a,b] in 1 dimension,
or they may be disconnected regions in multiple dimensions.
So the common interface is simply to ask the interval if a given point "IsInInterval".
The Interval also knows what confidence level it was constructed at and the space of
parameters for which it was constructed.
Note, one could use the same class for a Bayesian "credible interval".

*/

   class ConfInterval : public TNamed {

   public:

      /// constructor given name and title
      explicit ConfInterval(const char *name = nullptr) : TNamed(name,name) {}

      /// destructor
      ~ConfInterval() override {}

      /// operator=
      ConfInterval& operator=(const ConfInterval& other) {
         if (&other==this) { return *this; }
         TNamed::operator=(other);
         return *this;
      }

      /// check if given point is in the interval
      virtual bool IsInInterval(const RooArgSet&) const = 0;

      /// used to set confidence level.  Keep pure virtual
      virtual void SetConfidenceLevel(double cl) = 0;

      /// return confidence level
      virtual double ConfidenceLevel() const = 0;

      /// return list of parameters of interest defining this interval (return a new cloned list)
      virtual RooArgSet* GetParameters() const = 0;

      /// check if parameters are correct (i.e. they are the POI of this interval)
      virtual bool CheckParameters(const RooArgSet&) const = 0;


   protected:

      ClassDefOverride(ConfInterval,1) // Interface for Confidence Intervals

   };
}


#endif
