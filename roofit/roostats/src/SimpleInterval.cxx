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



//_________________________________________________________
//
// BEGIN_HTML
// SimpleInterval is a concrete implementation of the ConfInterval interface.  
// It implements simple 1-dimensional intervals in a range [a,b].
// In addition, you can ask it for the upper- or lower-bound.
// END_HTML
//
//

#ifndef RooStats_SimpleInterval
#include "RooStats/SimpleInterval.h"
#endif
#include "RooAbsReal.h"
#include <string>


ClassImp(RooStats::SimpleInterval) ;

using namespace RooStats;

//____________________________________________________________________
SimpleInterval::SimpleInterval() : fLowerLimit(0), fUpperLimit(0)
{
   // Default constructor
}

//____________________________________________________________________
SimpleInterval::SimpleInterval(const char* name) :
   ConfInterval(name,name), fLowerLimit(0), fUpperLimit(0)
{
   // Alternate constructor
}

//____________________________________________________________________
SimpleInterval::SimpleInterval(const char* name, const char* title) :
   ConfInterval(name,title), fLowerLimit(0), fUpperLimit(0)
{
   // Alternate constructor
}

//____________________________________________________________________
SimpleInterval::SimpleInterval(const char* name, RooAbsArg* var, Double_t lower, Double_t upper) :
   ConfInterval(name,name), fLowerLimit(lower), fUpperLimit(upper)
{
   // Alternate constructor
   fParameters = new RooArgSet((std::string(name)+"_parameters").c_str());
   fParameters->addClone( *var );
}

//____________________________________________________________________
SimpleInterval::SimpleInterval(const char* name, const char* title, RooAbsArg* var, Double_t lower, Double_t upper):
   ConfInterval(name,title), fLowerLimit(lower), fUpperLimit(upper)
{
   // Alternate constructor
   fParameters = new RooArgSet((std::string(name)+"_parameters").c_str());
   fParameters->addClone( *var );
}


//____________________________________________________________________
SimpleInterval::~SimpleInterval()
{
   // Destructor

}


//____________________________________________________________________
Bool_t SimpleInterval::IsInInterval(RooArgSet &parameterPoint) 
{  

   // Method to determine if a parameter point is in the interval
   if( !this->CheckParameters(parameterPoint) )
      return false; 

   if(parameterPoint.getSize() != 1 )
      return false;

   RooAbsReal* point = dynamic_cast<RooAbsReal*> (parameterPoint.first());
   if (point == 0)
      return false;

   if ( point->getVal() > fUpperLimit || point->getVal() < fUpperLimit)
      return false;


   return true;
}

//____________________________________________________________________
RooArgSet* SimpleInterval::GetParameters() const
{  
   // returns list of parameters
   return (RooArgSet*) fParameters->clone((std::string(fParameters->GetName())+"_clone").c_str());
}

//____________________________________________________________________
Bool_t SimpleInterval::CheckParameters(RooArgSet &parameterPoint) const
{  

   if (parameterPoint.getSize() != fParameters->getSize() ) {
      std::cout << "size is wrong, parameters don't match" << std::endl;
      return false;
   }
   if ( ! parameterPoint.equals( *fParameters ) ) {
      std::cout << "size is ok, but parameters don't match" << std::endl;
      return false;
   }
   return true;
}
