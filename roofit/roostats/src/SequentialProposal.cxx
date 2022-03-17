// @(#)root/roostats:$Id$
// Authors: Giovanni Petrucciani 4/21/2011
/*************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/** \class RooStats::SequentialProposal
    \ingroup Roostats

Class implementing a proposal function that samples the parameter space
by moving only in one coordinate (chosen randomly) at each step
*/

#include "RooStats/SequentialProposal.h"
#include <RooArgSet.h>
#include <iostream>
#include <memory>
#include <TIterator.h>
#include <RooRandom.h>
#include <RooStats/RooStatsUtils.h>

using namespace std;

ClassImp(RooStats::SequentialProposal);

namespace RooStats {

////////////////////////////////////////////////////////////////////////////////

SequentialProposal::SequentialProposal(double divisor) :
    ProposalFunction(),
    fDivisor(1./divisor)
{
}

////////////////////////////////////////////////////////////////////////////////
/// Populate xPrime with a new proposed point

void SequentialProposal::Propose(RooArgSet& xPrime, RooArgSet& x )
{
   RooStats::SetParameters(&x, &xPrime);
   int n = xPrime.getSize();
   int j = int( floor(RooRandom::uniform()*n) );
   int i = 0;
   for (auto *var : static_range_cast<RooRealVar *>(xPrime)) {
      if (i == j) {
        double val = var->getVal(), max = var->getMax(), min = var->getMin(), len = max - min;
        val += RooRandom::gaussian() * len * fDivisor;
        while (val > max) val -= len;
        while (val < min) val += len;
        var->setVal(val);
        //std::cout << "Proposing a step along " << var->GetName() << std::endl;
      }
      ++i;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Return the probability of proposing the point x1 given the starting
/// point x2

Bool_t SequentialProposal::IsSymmetric(RooArgSet& , RooArgSet& ) {
   return true;
}

Double_t SequentialProposal::GetProposalDensity(RooArgSet& ,
                                                RooArgSet& )
{
   return 1.0; // should not be needed
}

}
