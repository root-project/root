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

#ifndef ROOSTATS_PdfProposal
#define ROOSTATS_PdfProposal

#include "Rtypes.h"

#include "RooStats/ProposalFunction.h"

#include "RooArgSet.h"
#include "RooMsgService.h"
#include "RooRealVar.h"
#include "RooDataSet.h"
#include "RooAbsPdf.h"

#include <map>


namespace RooStats {

   class PdfProposal : public ProposalFunction {

   public:
      PdfProposal();
      PdfProposal(RooAbsPdf& pdf);

      /// Populate xPrime with a new proposed point
      void Propose(RooArgSet& xPrime, RooArgSet& x) override;

      /// Determine whether or not the proposal density is symmetric for
      /// points x1 and x2 - that is, whether the probability of reaching x2
      /// from x1 is equal to the probability of reaching x1 from x2
      Bool_t IsSymmetric(RooArgSet& x1, RooArgSet& x2) override;

      /// Return the probability of proposing the point x1 given the starting
      /// point x2
      Double_t GetProposalDensity(RooArgSet& x1, RooArgSet& x2) override;

      /// Set the PDF to be the proposal density function
      virtual void SetPdf(RooAbsPdf& pdf) { fPdf = &pdf; }

      /// Get the PDF is the proposal density function
      virtual const RooAbsPdf* GetPdf() const { return fPdf; }

      /// specify a mapping between a parameter of the proposal function and
      /// a parameter of interest.  this mapping is used to set the value of
      /// proposalParam equal to the value of update to determine the
      /// proposal function.
      /// proposalParam is a parameter of the proposal function that must
      /// be set to the value of update (from the current point) in order to
      /// propose a new point.
      virtual void AddMapping(RooRealVar& proposalParam, RooAbsReal& update);

      virtual void Reset()
      {
         delete fCache;
         fCache = NULL;
         fCachePosition = 0;
         fLastX.removeAll();
      }

      virtual void printMappings()
      {
         std::map<RooRealVar*, RooAbsReal*>::iterator it;
         for (it = fMap.begin(); it != fMap.end(); it++)
            std::cout << it->first->GetName() << " => " << it->second->GetName() << std::endl;
      }

      /// Set how many points to generate each time we propose from a new point
      /// Default (and minimum) is 1
      virtual void SetCacheSize(Int_t size)
      {
         if (size > 0)
            fCacheSize = size;
         else
            coutE(Eval) << "Warning: Requested non-positive cache size: " <<
               size << ". Cache size unchanged." << std::endl;
      }

      /// set whether we own the PDF that serves as the proposal density function
      /// By default, when constructed, PdfProposal does NOT own the PDF.
      virtual void SetOwnsPdf(Bool_t ownsPdf) { fOwnsPdf = ownsPdf; }

      //virtual void SetIsAlwaysSymmetric(Bool_t isAlwaysSymmetric)
      //{ fIsAlwaysSymmetric = isAlwaysSymmetric; }

      ~PdfProposal() override
      {
         delete fCache;
         if (fOwnsPdf)
            delete fPdf;
      }

   protected:
      RooAbsPdf* fPdf; /// the proposal density function
      std::map<RooRealVar*, RooAbsReal*> fMap; /// map of values in pdf to update
      std::map<RooRealVar*, RooAbsReal*>::iterator fIt; /// pdf iterator
      RooArgSet fLastX; /// the last point we were at
      Int_t fCacheSize; /// how many points to generate each time
      Int_t fCachePosition; /// our position in the cached proposal data set
      RooDataSet* fCache; /// the cached proposal data set
      RooArgSet fMaster; /// pointers to master variables needed for updates
      Bool_t fOwnsPdf; /// whether we own the proposal density function
      //Bool_t fIsAlwaysSymmetric; // does Q(x1 | x2) == Q(x2 | x1) for all x1, x2

      /// determine whether these two RooArgSets represent the same point
      virtual Bool_t Equals(RooArgSet& x1, RooArgSet& x2);

      /// Interface for tools setting limits (producing confidence intervals)
      ClassDefOverride(PdfProposal,1)
   };
}

#endif
