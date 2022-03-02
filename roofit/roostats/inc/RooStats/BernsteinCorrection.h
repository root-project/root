// @(#)root/roostats:$Id$
// Author: Kyle Cranmer
/*************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOSTATS_BernsteinCorrection
#define ROOSTATS_BernsteinCorrection


#include "Rtypes.h"

#include "TH1F.h"
#include "RooWorkspace.h"

namespace RooStats {

   class BernsteinCorrection {

   public:
      BernsteinCorrection(double tolerance = 0.05);
      virtual ~BernsteinCorrection() {}

      Int_t ImportCorrectedPdf(RooWorkspace*, const char*,const char*,const char*);
      void SetMaxCorrection(Double_t maxCorr){fMaxCorrection = maxCorr;}
      void SetMaxDegree(Int_t maxDegree){fMaxDegree = maxDegree;}
      void CreateQSamplingDist(RooWorkspace* wks,
                               const char* nominalName,
                               const char* varName,
                               const char* dataName,
                               TH1F*, TH1F*,
                               Int_t degree,
                               Int_t nToys=500);

   private:

      Int_t    fMaxDegree;     ///< maximum polynomial degree correction (default is 10)
      Double_t fMaxCorrection; ///< maximum correction factor at any point (default is 100)
      Double_t fTolerance;     ///< probability to add an unnecessary term


   protected:
      ClassDef(BernsteinCorrection,2) // A utility to add polynomial correction terms to a model to improve the description of data.
   };
}


#endif
