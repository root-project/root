// @(#)root/roostats:$Id$
// Author: Kyle Cranmer, Lorenzo Moneta, Gregory Schott, Wouter Verkerke
/*************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/*****************************************************************************
 * Project: RooStats
 * Package: RooFit/RooStats
 * @(#)root/roofit/roostats:$Id$
 * Authors:
 *   Kyle Cranmer, Lorenzo Moneta, Gregory Schott, Wouter Verkerke
 *
 *****************************************************************************/

/** \class RooStats::SimpleInterval
    \ingroup Roostats

SimpleInterval is a concrete implementation of the ConfInterval interface.
It implements simple 1-dimensional intervals in a range [a,b].
In addition, you can ask it for the upper- or lower-bound.
*/

#include "RooStats/SimpleInterval.h"
#include "RooAbsReal.h"
#include "RooRealVar.h"
#include <string>


using namespace std;

ClassImp(RooStats::SimpleInterval); ;

using namespace RooStats;


////////////////////////////////////////////////////////////////////////////////
/// Default constructor

SimpleInterval::SimpleInterval(const char* name) :
   ConfInterval(name),  fLowerLimit(0), fUpperLimit(0), fConfidenceLevel(0)
{
}


////////////////////////////////////////////////////////////////////////////////
///fParameters.add( other.fParameters );

SimpleInterval::SimpleInterval(const SimpleInterval& other, const char* name)
 : ConfInterval(name)
 , fParameters(other.fParameters)
 , fLowerLimit(other.fLowerLimit)
 , fUpperLimit(other.fUpperLimit)
 , fConfidenceLevel(other.fConfidenceLevel)
{
}


////////////////////////////////////////////////////////////////////////////////

SimpleInterval&
SimpleInterval::operator=(const SimpleInterval& other)
{
  if (&other==this) {
    return *this ;
  }

  ConfInterval::operator = (other);

  //fParameters      = other.fParameters;
  fParameters.removeAll();
  fParameters.add(other.fParameters);
  fLowerLimit      = other.fLowerLimit;
  fUpperLimit      = other.fUpperLimit;
  fConfidenceLevel = other.fConfidenceLevel;

  return *this ;
}

////////////////////////////////////////////////////////////////////////////////
/// Alternate constructor

SimpleInterval::SimpleInterval(const char* name, const RooRealVar & var, Double_t lower, Double_t upper, Double_t cl) :
   ConfInterval(name), fParameters(var), fLowerLimit(lower), fUpperLimit(upper), fConfidenceLevel(cl)
{
}

////////////////////////////////////////////////////////////////////////////////
/// Destructor

SimpleInterval::~SimpleInterval()
{
}

////////////////////////////////////////////////////////////////////////////////
/// Method to determine if a parameter point is in the interval

bool SimpleInterval::IsInInterval(const RooArgSet &parameterPoint) const
{
   if( !this->CheckParameters(parameterPoint) )
      return false;

   if(parameterPoint.getSize() != 1 )
      return false;

   RooAbsReal* point = dynamic_cast<RooAbsReal*> (parameterPoint.first());
   if (point == 0)
      return false;

   if ( point->getVal() > fUpperLimit || point->getVal() < fLowerLimit)
      return false;


   return true;
}

////////////////////////////////////////////////////////////////////////////////
/// return cloned list of parameters

RooArgSet* SimpleInterval::GetParameters() const
{
   return new RooArgSet(fParameters);
}

////////////////////////////////////////////////////////////////////////////////

bool SimpleInterval::CheckParameters(const RooArgSet &parameterPoint) const
{
   if (parameterPoint.getSize() != fParameters.getSize() ) {
      std::cout << "size is wrong, parameters don't match" << std::endl;
      return false;
   }
   if ( ! parameterPoint.equals( fParameters ) ) {
      std::cout << "size is ok, but parameters don't match" << std::endl;
      return false;
   }
   return true;
}
