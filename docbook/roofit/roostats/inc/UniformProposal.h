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

#ifndef ROOT_Rtypes
#include "Rtypes.h"
#endif

#ifndef ROOSTATS_ProposalFunction
#include "RooStats/ProposalFunction.h"
#endif

#ifndef ROO_ARG_SET
#include "RooArgSet.h"
#endif
#ifndef ROO_MSG_SERVICE
#include "RooMsgService.h"
#endif
#ifndef ROO_REAL_VAR
#include "RooRealVar.h"
#endif
#ifndef ROOT_TIterator
#include "TIterator.h"
#endif

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

      // Return the probability of proposing the point x1 given the starting
      // point x2
      virtual Double_t GetProposalDensity(RooArgSet& x1, RooArgSet& x2);

      virtual ~UniformProposal() {}

      ClassDef(UniformProposal,1) // A concrete implementation of ProposalFunction, that uniformly samples the parameter space.
   };
}

#endif
