// @(#)root/roostats:$Id$
// Authors: Giovanni Petrucciani 4/21/2011
/*************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef RooStats_SequentialProposal
#define RooStats_SequentialProposal

#include "Rtypes.h"
#include "RooStats/ProposalFunction.h"
#include "RooArgSet.h"

namespace RooStats {

class SequentialProposal : public ProposalFunction {

   public:
   SequentialProposal() : RooStats::ProposalFunction(), fDivisor(0) {}
      SequentialProposal(double divisor) ;

      /// Populate xPrime with a new proposed point
      void Propose(RooArgSet& xPrime, RooArgSet& x) override;

      /// Determine whether or not the proposal density is symmetric for
      /// points x1 and x2 - that is, whether the probability of reaching x2
      /// from x1 is equal to the probability of reaching x1 from x2
      bool IsSymmetric(RooArgSet& x1, RooArgSet& x2) override ;

      /// Return the probability of proposing the point x1 given the starting
      /// point x2
      double GetProposalDensity(RooArgSet& x1, RooArgSet& x2) override;

      ~SequentialProposal() override {}

      ClassDefOverride(SequentialProposal,1) // A concrete implementation of ProposalFunction, that uniformly samples the parameter space.

    private:

      double fDivisor;
};

}

#endif
