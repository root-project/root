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

#include "RooAbsCollection.h"
#include "RooMsgService.h"
#include "TIterator.h"
#include "RooRealVar.h"

using namespace std;

namespace RooStats {

   class ProposalFunction : public TObject {

   public:
      //Default constructor
      ProposalFunction()
      {
         fParams = NULL;
      }

      // It is a checked runtime error for params to contain anything
      // other than RooRealVar objects.
      // Note: This constructor uniformly randomizes the values of params
      ProposalFunction(RooAbsCollection& params)
      {
         SetParameters(params);
      }

      virtual ~ProposalFunction() {}

      // Populate xPrime with the new proposed point
      virtual void Propose(RooAbsCollection& xPrime) = 0;
      
      // Determine whether or not the proposal density is symmetric for
      // points x1 and x2 - that is, whether the probabilty of reaching x2
      // from x1 is equal to the probability of reaching x1 from x2
      virtual Bool_t IsSymmetric(RooAbsCollection& x1,
                                 RooAbsCollection& x2) = 0;

      // Return the probability of reaching the point xPrime given the starting
      // point x
      virtual Double_t GetProposalDensity(RooAbsCollection& xPrime,
                                          RooAbsCollection& x) = 0;

      // Set the parameters for which the ProposalFunction will
      // propose values, and randomize their values.
      virtual void SetParameters(RooAbsCollection& params)
      {
         fParams = &params;
         TIterator* it = params.createIterator();
         TObject* obj;
         while ((obj = it->Next()) != NULL) {
            if (!dynamic_cast<RooRealVar*>(obj)) {
               coutE(Eval) << "Error when setting parameters in"
                           << "ProposalFunction: "
                           << "Object \"" << obj->GetName() << "\" not of type "
                           << "RooRealVar" << endl;
               coutE(Eval) << "Using NULL parameters list" << endl;
               fParams = NULL;
               return;
            }
         }
         // Made it here, so all parameters are RooRealVars
         fParams = &params;
         randomizeCollection(*fParams);
      }

   protected:
      RooAbsCollection* fParams;

      // Assuming all values in coll are RooRealVars, randomize their values.
      virtual void randomizeCollection(RooAbsCollection& coll);

      ClassDef(ProposalFunction,1) // Interface for the proposal function used with Markov Chain Monte Carlo
   };
}

#endif
