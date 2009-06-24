// @(#)root/roostats:$Id: ProposalFunction.cxx 26805 2009-06-17 14:31:02Z kbelasco $
// Authors: Kevin Belasco        17/06/2009
// Authors: Kyle Cranmer         17/06/2009
/*************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//_________________________________________________
/*
BEGIN_HTML
<p>
ProposalFunction is an interface for all proposal functions that would be used with a Markov Chain Monte Carlo algorithm.  
Given a current point in the parameter space it proposes a new point.  
Proposal functions may or may not be symmetric, in the sense that the probability to propose X1 given we are at X2 
need not be the same as the probability to propose X2 given that we are at X1.  In this case, the IsSymmetric method
should return false, and the Metropolis algorithm will need to take into account the proposal density to maintain detailed balance.
</p>
END_HTML
*/
//

#include "RooStats/ProposalFunction.h"

#include "RooRealVar.h"
#include "TIterator.h"
#include "RooArgSet.h"

ClassImp(RooStats::ProposalFunction);

using namespace RooFit;
using namespace RooStats;

// Assuming all values in set are RooRealVars, randomize their values.
void ProposalFunction::randomizeSet(RooArgSet& set)
{
   TIterator* it = set.createIterator();
   RooRealVar* var;

   while ((var = (RooRealVar*)it->Next()) != NULL)
      var->randomize();
}
