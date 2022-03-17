// @(#)root/roostats:$Id$
// Authors: Kevin Belasco        17/06/2009
// Authors: Kyle Cranmer         17/06/2009
/*************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/** \class RooStats::UniformProposal
    \ingroup Roostats

UniformProposal is a concrete implementation of the ProposalFunction interface
for use with a Markov Chain Monte Carlo algorithm.  This proposal function is
a uniformly random distribution over the parameter space.  The proposal
ignores the current point when it proposes a new point.  The proposal
function is symmetric, though it may not cause a MetropolisHastings run to
converge as quickly as other proposal functions.

*/

#include "Rtypes.h"

#include "RooStats/RooStatsUtils.h"
#include "RooStats/UniformProposal.h"
#include "RooArgSet.h"
#include "RooMsgService.h"
#include "RooRealVar.h"
#include "TIterator.h"

using namespace std;

ClassImp(RooStats::UniformProposal);

using namespace RooFit;
using namespace RooStats;

////////////////////////////////////////////////////////////////////////////////
/// Populate xPrime with a new proposed point

void UniformProposal::Propose(RooArgSet& xPrime, RooArgSet& /* x */)
{
   // kbelasco: remember xPrime and x have not been checked for containing
   // only RooRealVars
   RooStats::RandomizeCollection(xPrime);
}

////////////////////////////////////////////////////////////////////////////////
/// Determine whether or not the proposal density is symmetric for
/// points x1 and x2 - that is, whether the probability of reaching x2
/// from x1 is equal to the probability of reaching x1 from x2

Bool_t UniformProposal::IsSymmetric(RooArgSet& /* x1 */ , RooArgSet& /* x2 */)
{
   return true;
}

////////////////////////////////////////////////////////////////////////////////
/// Return the probability of proposing the point x1 given the starting
/// point x2

Double_t UniformProposal::GetProposalDensity(RooArgSet& /* x1 */,
                                              RooArgSet& x2)
{
   // For a uniform proposal, all points have equal probability and the
   // value of the proposal density function is:
   // 1 / (N-dimensional volume of interval)
   Double_t volume = 1.0;
   for (auto const *var : static_range_cast<RooRealVar *> (x2))
      volume *= (var->getMax() - var->getMin());
   return 1.0 / volume;
}
