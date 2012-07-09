// @(#)root/roostats:$Id$
// Author: Sven Kreiss, Kyle Cranmer   Nov 2010
/*************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOSTATS_HybridCalculator
#define ROOSTATS_HybridCalculator

//_________________________________________________
/*
BEGIN_HTML
<p>
This class extends the HybridCalculator with Importance Sampling. The use
of ToyMCSampler as the TestStatSampler is assumed.
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

namespace RooStats {

   class HybridCalculator : public HypoTestCalculatorGeneric {

   public:
      HybridCalculator(
                        const RooAbsData &data,
                        const ModelConfig &altModel,
                        const ModelConfig &nullModel,
                        TestStatSampler* sampler=0
      ) :
         HypoTestCalculatorGeneric(data, altModel, nullModel, sampler),
         fPriorNuisanceNull(MakeNuisancePdf(nullModel, "PriorNuisanceNull")),
         fPriorNuisanceAlt(MakeNuisancePdf(altModel, "PriorNuisanceAlt")),
         fPriorNuisanceNullExternal(false),
         fPriorNuisanceAltExternal(false),
         fNToysNull(-1),
         fNToysAlt(-1),
         fNToysNullTail(0),
         fNToysAltTail(0)
      {
      }

      ~HybridCalculator() {
         if(fPriorNuisanceNullExternal == false) delete fPriorNuisanceNull;   
         if(fPriorNuisanceAltExternal == false) delete fPriorNuisanceAlt;
      }


      // Override the distribution used for marginalizing nuisance parameters that is inferred from ModelConfig
      virtual void ForcePriorNuisanceNull(RooAbsPdf& priorNuisance) { 
         if(fPriorNuisanceNullExternal == false) delete fPriorNuisanceNull;
         fPriorNuisanceNull = &priorNuisance; fPriorNuisanceNullExternal = true; 
      }
      virtual void ForcePriorNuisanceAlt(RooAbsPdf& priorNuisance) { 
         if(fPriorNuisanceAltExternal == false) delete fPriorNuisanceAlt;
         fPriorNuisanceAlt = &priorNuisance; fPriorNuisanceAltExternal = true; 
      }

      virtual void SetNullModel(const ModelConfig &nullModel) {
         fNullModel = &nullModel;
         if(fPriorNuisanceNullExternal == false) {
            delete fPriorNuisanceNull; 
            fPriorNuisanceNull = MakeNuisancePdf(nullModel, "PriorNuisanceNull");
         }
      }
   
      virtual void SetAlternateModel(const ModelConfig &altModel) {
         fAltModel = &altModel;
         if(fPriorNuisanceAltExternal == false) {
            delete fPriorNuisanceAlt; 
            fPriorNuisanceAlt = MakeNuisancePdf(altModel, "PriorNuisanceAlt");
         }
      }

      // set number of toys
      void SetToys(int toysNull, int toysAlt) { fNToysNull = toysNull; fNToysAlt = toysAlt; }

      // set least number of toys in tails
      void SetNToysInTails(int toysNull, int toysAlt) { fNToysNullTail = toysNull; fNToysAltTail = toysAlt; }

   protected:
      // check whether all input is consistent
      int CheckHook(void) const;

      // configure TestStatSampler for the Null run
      int PreNullHook(RooArgSet* /*parameterPoint*/, double obsTestStat) const;

      // configure TestStatSampler for the Alt run
      int PreAltHook(RooArgSet* /*parameterPoint*/, double obsTestStat) const;

   protected:
      RooAbsPdf *fPriorNuisanceNull;
      RooAbsPdf *fPriorNuisanceAlt;

      // these flags tell us if the nuisance pdfs came from an external resource (via ForcePriorNuisance)
      // or were created internally and should be deleted
      Bool_t fPriorNuisanceNullExternal; 
      Bool_t fPriorNuisanceAltExternal;

      // different number of toys for null and alt
      int fNToysNull;
      int fNToysAlt;

      // adaptive sampling
      int fNToysNullTail;
      int fNToysAltTail;

   protected:
      ClassDef(HybridCalculator,2)
   };
}

#endif
