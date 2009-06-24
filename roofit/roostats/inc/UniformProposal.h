// @(#)root/roostats:$Id: UniformProposal.h 26805 2009-06-17 14:31:02Z kbelasco $
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

#ifndef ROOT_Rtypes
#include "Rtypes.h"
#endif

#ifndef RooStats_ProposalFunction
#include "RooStats/ProposalFunction.h"
#endif

#include "RooArgSet.h"
#include "RooMsgService.h"
#include "RooRealVar.h"
#include "TIterator.h"

namespace RooStats {

   class UniformProposal : public ProposalFunction {

   public:
      UniformProposal() : ProposalFunction() {}

      // Populate xPrime with a new proposed point
      virtual void Propose(RooArgSet& xPrime, RooArgSet& x);

      // Determine whether or not the proposal density is symmetric for
      // points x1 and x2 - that is, whether the probabilty of reaching x2
      // from x1 is equal to the probability of reaching x1 from x2
      virtual Bool_t IsSymmetric(RooArgSet& x1, RooArgSet& x2);

      // Return the probability of proposing the point xPrime given the starting
      // point x
      virtual Double_t GetProposalDensity(RooArgSet& xPrime, RooArgSet& x);

      virtual ~UniformProposal() {}

      ClassDef(UniformProposal,1) // A concrete implementation of ProposalFunction, that uniformly samples the parameter space.
   };
}

#endif
