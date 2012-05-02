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

#ifndef ROOSTATS_ProposalFunction
#define ROOSTATS_ProposalFunction

#ifndef ROOT_Rtypes
#include "Rtypes.h"
#endif

#ifndef ROO_ARG_SET
#include "RooArgSet.h"
#endif
#ifndef ROO_MSG_SERVICE
#include "RooMsgService.h"
#endif
#ifndef ROOT_TIterator
#include "TIterator.h"
#endif
#ifndef ROO_REAL_VAR
#include "RooRealVar.h"
#endif


namespace RooStats {

   class ProposalFunction : public TObject {

   public:
      //Default constructor
      ProposalFunction() {}

      virtual ~ProposalFunction() {}

      // Populate xPrime with the new proposed point,
      // possibly based on the current point x
      virtual void Propose(RooArgSet& xPrime, RooArgSet& x) = 0;
      
      // Determine whether or not the proposal density is symmetric for
      // points x1 and x2 - that is, whether the probabilty of reaching x2
      // from x1 is equal to the probability of reaching x1 from x2
      virtual Bool_t IsSymmetric(RooArgSet& x1, RooArgSet& x2) = 0;

      // Return the probability of proposing the point x1 given the starting
      // point x2
      virtual Double_t GetProposalDensity(RooArgSet& x1, RooArgSet& x2) = 0;

      // Check the parameters for which the ProposalFunction will
      // propose values to make sure they are all RooRealVars
      // Return true if all objects are RooRealVars, false otherwise
      virtual bool CheckParameters(RooArgSet& params)
      {
         TIterator* it = params.createIterator();
         TObject* obj;
         while ((obj = it->Next()) != NULL) {
            if (!dynamic_cast<RooRealVar*>(obj)) {
               coutE(Eval) << "Error when checking parameters in"
                           << "ProposalFunction: "
                           << "Object \"" << obj->GetName() << "\" not of type "
                           << "RooRealVar" << std::endl;
               delete it;
               return false;
            }
         }
         delete it;
         // Made it here, so all parameters are RooRealVars
         return true;
      }

   protected:
      ClassDef(ProposalFunction,1) // Interface for the proposal function used with Markov Chain Monte Carlo
   };
}

#endif
