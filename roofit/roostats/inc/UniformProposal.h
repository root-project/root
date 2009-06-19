#ifndef ROOSTATS_UniformProposal
#define ROOSTATS_UniformProposal

#ifndef ROOT_Rtypes
#include "Rtypes.h"
#endif

#ifndef RooStats_ProposalFunction
#include "RooStats/ProposalFunction.h"
#endif

#include "RooAbsCollection.h"
#include "RooMsgService.h"
#include "RooRealVar.h"
#include "TIterator.h"

namespace RooStats {

   class UniformProposal : public ProposalFunction {

   public:
      UniformProposal() : ProposalFunction() {}

      // It is a checked runtime error for params to contain anything
      // other than RooRealVar objects.
      UniformProposal(RooAbsCollection& params) : ProposalFunction(params) {}

      // Populate xPrime with a new proposed point
      virtual void Propose(RooAbsCollection& xPrime);

      // Determine whether or not the proposal density is symmetric for
      // points x1 and x2 - that is, whether the probabilty of reaching x2
      // from x1 is equal to the probability of reaching x1 from x2
      virtual Bool_t IsSymmetric(RooAbsCollection& x1, RooAbsCollection& x2);

      // Return the probability of proposing the point xPrime given the starting
      // point x
      virtual Double_t GetProposalDensity(RooAbsCollection& xPrime,
                                          RooAbsCollection& x);

      virtual ~UniformProposal() {}

      // Interface for tools setting limits (producing confidence intervals)
      ClassDef(UniformProposal,1)
   };
}

#endif
