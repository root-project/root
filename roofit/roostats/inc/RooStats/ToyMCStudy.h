// @(#)root/roostats:$Id$
// Author: Sven Kreiss and Kyle Cranmer    June 2010
/*************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOSTATS_ToyMCStudy
#define ROOSTATS_ToyMCStudy

//_________________________________________________
/*
BEGIN_HTML
<p>
ToyMCStudy is an implementation of RooAbsStudy for toy Monte Carlo sampling.
This class is automatically used by ToyMCSampler when given a ProofConfig.
This is also its intended use case.
</p>
END_HTML
*/
//

#ifndef ROOT_Rtypes
#include "Rtypes.h"
#endif

#include "RooAbsStudy.h"

#include "RooStats/ToyMCSampler.h"
#include "RooStats/SamplingDistribution.h"

#include "RooWorkspace.h"
#include "RooArgSet.h"

#include "RooDataSet.h"
#include "RooLinkedList.h"
#include "RooLinkedListIter.h"

#include <vector>

namespace RooStats {

class ToyMCStudy: public RooAbsStudy {

   public:
      // need to have constructor without arguments for proof
      ToyMCStudy(const char *name = "ToyMCStudy", const char *title = "ToyMCStudy") :
         RooAbsStudy(name, title),
         fToyMCSampler(NULL)
      {
         // In this case, this is the normal output. The SamplingDistribution
         // instances are stored as detailed output.
         storeDetailedOutput(kTRUE);
      }

	   RooAbsStudy* clone(const char* /*newname*/="") const { return new ToyMCStudy(*this) ; }     

      virtual ~ToyMCStudy() {}

      // RooAbsStudy interfaces
      virtual Bool_t initialize(void);
      virtual Bool_t execute(void);
      virtual Bool_t finalize(void);

      RooDataSet* merge();

      void SetToyMCSampler(ToyMCSampler& t) { fToyMCSampler = &t; }
      void SetParamPoint(const RooArgSet& paramPoint) { fParamPoint.add(paramPoint); }

   protected:

      ToyMCSampler *fToyMCSampler;
      RooArgSet fParamPoint;

   protected:
   ClassDef(ToyMCStudy,1); // toy MC study for parallel processing
   
};


class ToyMCPayload : public TNamed {
   
   public:
      
      ToyMCPayload() {
         // proof constructor, do not use
	 fDataSet = NULL;
      }

      ToyMCPayload(RooDataSet* sd)
      {
         fDataSet = sd;
      }
      
      virtual ~ToyMCPayload() {
      }


      RooDataSet* GetSamplingDistributions()
      {
         return fDataSet;
      }
   
   private:
      RooDataSet* fDataSet;
   
   protected:
   ClassDef(ToyMCPayload,1);
};


}


#endif
