// @(#)root/roostats:$Id: ProposalFunction.h 26805 2009-06-17 14:31:02Z kbelasco $
// Authors: Kevin Belasco        17/06/2009
// Authors: Kyle Cranmer         17/06/2009
/*************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOSTATS_ProposalFunction
#define ROOSTATS_ProposalFunction

#ifndef ROOT_Rtypes
#include "Rtypes.h"
#endif

#include "RooArgSet.h"
#include "RooMsgService.h"
#include "TIterator.h"
#include "RooRealVar.h"

using namespace std;

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

      // Return the probability of reaching the point xPrime given the starting
      // point x
      virtual Double_t GetProposalDensity(RooArgSet& xPrime, RooArgSet& x) = 0;

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
                           << "RooRealVar" << endl;
               return false;
            }
         }
         // Made it here, so all parameters are RooRealVars
         return true;
      }

   protected:
      // Assuming all values in coll are RooRealVars, randomize their values.
      virtual void randomizeSet(RooArgSet& set);

      ClassDef(ProposalFunction,1) // Interface for the proposal function used with Markov Chain Monte Carlo
   };
}

#endif
