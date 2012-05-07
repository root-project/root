// @(#)root/roostats:$Id: FrequentistCalculator.h 37084 2010-11-29 21:37:13Z moneta $
// Author: Sven Kreiss, Kyle Cranmer   Nov 2010
/*************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOSTATS_FrequentistCalculator
#define ROOSTATS_FrequentistCalculator

//_________________________________________________
/*
BEGIN_HTML
<p>
The use of ToyMCSampler as the TestStatSampler is assumed.
</p>
END_HTML
*/
//



#ifndef ROOSTATS_HypoTestCalculatorGeneric
#include "RooStats/HypoTestCalculatorGeneric.h"
#endif

#ifndef ROOSTATS_ToyMCSampler
#include "RooStats/ToyMCSampler.h"
#endif

#ifndef ROOSTATS_DetailedOutputAggregator
#include "RooStats/DetailedOutputAggregator.h"
#endif

#include "RooFitResult.h"

namespace RooStats {

   class FrequentistCalculator : public HypoTestCalculatorGeneric {

   public:
      FrequentistCalculator(
                        const RooAbsData &data,
                        const ModelConfig &altModel,
                        const ModelConfig &nullModel,
                        TestStatSampler* sampler=0
      ) :
         HypoTestCalculatorGeneric(data, altModel, nullModel, sampler),
         fConditionalMLEsNull(NULL),
         fConditionalMLEsAlt(NULL),
         fNToysNull(-1),
         fNToysAlt(-1),
         fNToysNullTail(0),
         fNToysAltTail(0),
	 fFitInfo(NULL),
	 fStoreFitInfo(false)
      {
      }

      ~FrequentistCalculator() {
         if( fConditionalMLEsNull ) delete fConditionalMLEsNull;
	 if( fConditionalMLEsAlt ) delete fConditionalMLEsAlt;
	 if( fFitInfo ) delete fFitInfo;
      }


      // set number of toys
      void SetToys(int toysNull, int toysAlt) { fNToysNull = toysNull; fNToysAlt = toysAlt; }

      // set least number of toys in tails
      void SetNToysInTails(int toysNull, int toysAlt) { fNToysNullTail = toysNull; fNToysAltTail = toysAlt; }

      // set given nuisance parameters to a specific value that will be used instead of their
      // profiled value for Null toys
      void SetConditionalMLEsNull( const RooArgSet* c ) {
         if( fConditionalMLEsNull ) delete fConditionalMLEsNull;
         
         if( c ) fConditionalMLEsNull = (const RooArgSet*)c->snapshot();
         else fConditionalMLEsNull = NULL;
      }

      // set given nuisance parameters to a specific value that will be used instead of their
      // profiled value for Alternate toys
      void SetConditionalMLEsAlt( const RooArgSet* c ) {
         if( fConditionalMLEsAlt ) delete fConditionalMLEsAlt;
         
         if( c ) fConditionalMLEsAlt = (const RooArgSet*)c->snapshot();
         else fConditionalMLEsAlt = NULL;
      }

      void StoreFitInfo(bool val = true) {
	      fStoreFitInfo = val;
      }

      const RooArgSet* GetFitInfo() const {
	      return fFitInfo;
      }

   protected:
      // configure TestStatSampler for the Null run
      int PreNullHook(RooArgSet *parameterPoint, double obsTestStat) const;

      // configure TestStatSampler for the Alt run
      int PreAltHook(RooArgSet *parameterPoint, double obsTestStat) const;

      void PreHook() const;
      void PostHook() const;

   protected:
      // MLE inputs
      const RooArgSet* fConditionalMLEsNull;
      const RooArgSet* fConditionalMLEsAlt;
   
      // different number of toys for null and alt
      int fNToysNull;
      int fNToysAlt;

      // adaptive sampling
      int fNToysNullTail;
      int fNToysAltTail;

   private:
      mutable RooArgSet* fFitInfo;
      bool fStoreFitInfo;

   protected:
      ClassDef(FrequentistCalculator,1)
   };
}

#endif
