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



//_________________________________________________
/*
BEGIN_HTML
<p>
HypoTestResult is an base class for a results from hypothesis tests.  
Any tool inheriting from HypoTestCalculator can return a HypoTestResult.
As such, it stores a p-value for the null-hypothesis (eg. background-only) 
and an alternate hypothesis (eg. signal+background).  
The p-values can also be transformed into confidence levels (CLb, CLsplusb) in a trivial way.
The ratio of the CLsplusb to CLb is often called CLs, and is considered useful, though it is 
not a probability.
Finally, the p-value of the null can be transformed into a number of equivalent Gaussian sigma using the 
Significance method.
END_HTML
*/
//

#include "RooStats/HypoTestResult.h"
#include "RooAbsReal.h"

#ifndef RooStats_RooStatsUtils
#include "RooStats/RooStatsUtils.h"
#endif


ClassImp(RooStats::HypoTestResult) ;

using namespace RooStats;

//____________________________________________________________________
HypoTestResult::HypoTestResult(const char* name) : 
   TNamed(name,name), fNullPValue(0), fAlternatePValue(0)
{
   // Default constructor
}


//____________________________________________________________________
HypoTestResult::HypoTestResult(const char* name, Double_t nullp, Double_t altp) :
   TNamed(name,name), fNullPValue(nullp), fAlternatePValue(altp)
{
   // Alternate constructor
}


//____________________________________________________________________
HypoTestResult::~HypoTestResult()
{
   // Destructor

}
