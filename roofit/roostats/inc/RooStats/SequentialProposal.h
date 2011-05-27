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

#ifndef ROOT_Rtypes
#include "Rtypes.h"
#endif
#ifndef ROOSTATS_ProposalFunction
#include "RooStats/ProposalFunction.h"
#endif
#ifndef ROO_ARG_SET
#include "RooArgSet.h"
#endif

namespace RooStats {


class SequentialProposal : public ProposalFunction {

   public:
   SequentialProposal() : RooStats::ProposalFunction(), fDivisor(0) {}
      SequentialProposal(double divisor) ;

      // Populate xPrime with a new proposed point
      virtual void Propose(RooArgSet& xPrime, RooArgSet& x);

      // Determine whether or not the proposal density is symmetric for
      // points x1 and x2 - that is, whether the probabilty of reaching x2
      // from x1 is equal to the probability of reaching x1 from x2
      virtual Bool_t IsSymmetric(RooArgSet& x1, RooArgSet& x2) ;

      // Return the probability of proposing the point x1 given the starting
      // point x2
      virtual Double_t GetProposalDensity(RooArgSet& x1, RooArgSet& x2);

      virtual ~SequentialProposal() {}

      ClassDef(SequentialProposal,1) // A concrete implementation of ProposalFunction, that uniformly samples the parameter space.

    private:

      double fDivisor;
};

} 

#endif
