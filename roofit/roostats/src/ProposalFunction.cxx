// @(#)root/roostats:$Id: ProposalFunction.cxx 26805 2009-06-17 14:31:02Z kbelasco $
// Author: Kevin Belasco        17/06/2009
/*************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "RooStats/ProposalFunction.h"

#include "RooRealVar.h"
#include "TIterator.h"
#include "RooAbsCollection.h"

ClassImp(RooStats::ProposalFunction);

using namespace RooFit;
using namespace RooStats;

// Assuming all values in coll are RooRealVars, randomize their values.
void ProposalFunction::randomizeCollection(RooAbsCollection& coll)
{
   TIterator* it = coll.createIterator();
   RooRealVar* var;

   while ((var = (RooRealVar*)it->Next()) != NULL)
      var->randomize();
}
