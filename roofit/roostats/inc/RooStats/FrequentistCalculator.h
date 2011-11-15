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
         fNullImportanceDensity(NULL),
         fNullImportanceSnapshot(NULL),
         fAltImportanceDensity(NULL),
         fAltImportanceSnapshot(NULL),
         fNToysNull(-1),
         fNToysAlt(-1),
         fNToysNullTail(0),
         fNToysAltTail(0)
      {
      }

      ~FrequentistCalculator() {
         if(fNullImportanceSnapshot) delete fNullImportanceSnapshot;
         if(fAltImportanceSnapshot) delete fAltImportanceSnapshot;
      }


      // sets importance density and snapshot (optional)
      void SetNullImportanceDensity(RooAbsPdf *p, const RooArgSet *s = NULL) {
         fNullImportanceDensity = p;
         if(s) fNullImportanceSnapshot = (RooArgSet*)s->snapshot();
         else fNullImportanceSnapshot = NULL;
      }

      // sets importance density and snapshot (optional)
      void SetAltImportanceDensity(RooAbsPdf *p, const RooArgSet *s = NULL) {
         fAltImportanceDensity = p;
         if(s) fAltImportanceSnapshot = (RooArgSet*)s->snapshot();
         else fAltImportanceSnapshot = NULL;
      }

      // set number of toys
      void SetToys(int toysNull, int toysAlt) { fNToysNull = toysNull; fNToysAlt = toysAlt; }

      // set least number of toys in tails
      void SetNToysInTails(int toysNull, int toysAlt) { fNToysNullTail = toysNull; fNToysAltTail = toysAlt; }

   protected:
      // configure TestStatSampler for the Null run
      int PreNullHook(RooArgSet *parameterPoint, double obsTestStat) const;

      // configure TestStatSampler for the Alt run
      int PreAltHook(RooArgSet *parameterPoint, double obsTestStat) const;

   protected:
      // importance sampling
      RooAbsPdf *fNullImportanceDensity;
      const RooArgSet *fNullImportanceSnapshot;
      RooAbsPdf *fAltImportanceDensity;
      const RooArgSet *fAltImportanceSnapshot;

      // different number of toys for null and alt
      int fNToysNull;
      int fNToysAlt;

      // adaptive sampling
      int fNToysNullTail;
      int fNToysAltTail;

   protected:
      ClassDef(FrequentistCalculator,1)
   };
}

#endif
