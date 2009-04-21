// @(#)root/roostats:$Id: BernsteinCorrection.h 26805 2009-02-19 10:00:00 pellicci $
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


#ifndef ROOT_Rtypes
#include "Rtypes.h"
#endif

#include "TH1F.h"
#include "RooWorkspace.h"

namespace RooStats {

  class BernsteinCorrection {

  public:
    BernsteinCorrection(double tolerance = 0.05);
    virtual ~BernsteinCorrection() {}

    Int_t ImportCorrectedPdf(RooWorkspace*, const char*,const char*,const char*);
    void SetMaxCorrection(Double_t maxCorr){fMaxCorrection = maxCorr;}
    void CreateQSamplingDist(RooWorkspace* wks, 
			      const char* nominalName, 
			      const char* varName, 
			      const char* dataName,
			      TH1F*, TH1F*,
			      Int_t degree, 
			      Int_t nToys=500);

  private:
    Double_t fMaxCorrection; // maximum correction factor at any point
    Double_t fTolerance; // probability to add an unecessary term

   protected:
      ClassDef(BernsteinCorrection,1) 
   };
}


#endif
