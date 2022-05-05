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

#ifndef ROOSTATS_UniformProposal
#define ROOSTATS_UniformProposal

#include "Rtypes.h"

#include "RooStats/ProposalFunction.h"

#include "RooArgSet.h"
#include "RooMsgService.h"
#include "RooRealVar.h"
#include "TIterator.h"

namespace RooStats {

   class UniformProposal : public ProposalFunction {

   public:
      UniformProposal() : ProposalFunction() {}

      /// Populate xPrime with a new proposed point
      void Propose(RooArgSet& xPrime, RooArgSet& x) override;

      /// Determine whether or not the proposal density is symmetric for
      /// points x1 and x2 - that is, whether the probability of reaching x2
      /// from x1 is equal to the probability of reaching x1 from x2
      bool IsSymmetric(RooArgSet& x1, RooArgSet& x2) override;

      /// Return the probability of proposing the point x1 given the starting
      /// point x2
      Double_t GetProposalDensity(RooArgSet& x1, RooArgSet& x2) override;

      ~UniformProposal() override {}

      ClassDefOverride(UniformProposal,1) // A concrete implementation of ProposalFunction, that uniformly samples the parameter space.
   };
}

#endif
