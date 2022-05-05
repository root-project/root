// @(#)root/roostats:$Id$
// Authors: Kevin Belasco        17/06/2009
// Authors: Kyle Cranmer         17/06/2009
/*************************************************************************
 * Project: RooStats                                                     *
 * Package: RooFit/RooStats                                              *
 *************************************************************************
 * Copyright (C) 1995-2008, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOSTATS_MCMCIntervalPlot
#define ROOSTATS_MCMCIntervalPlot

#include "RooPrintable.h"
#include "RooArgSet.h"
#include "TNamed.h"
#include "TH1.h"
#include "RooStats/MCMCInterval.h"
#include "RooNDKeysPdf.h"
#include "RooProduct.h"

namespace RooStats {

   class MCMCIntervalPlot : public TNamed, public RooPrintable {

   public:
      MCMCIntervalPlot();
      MCMCIntervalPlot(MCMCInterval& interval);

      /// Destructor of SamplingDistribution
      ~MCMCIntervalPlot() override;

      void SetMCMCInterval(MCMCInterval& interval);
      void SetLineColor(Color_t color) {fLineColor = color;}
      void SetLineWidth(Int_t width) {fLineWidth = width;}
      void SetShadeColor(Color_t color) {fShadeColor = color;}
      void SetShowBurnIn(bool showBurnIn) { fShowBurnIn = showBurnIn; }

      void Draw(const Option_t* options = NULL) override;

      void DrawChainScatter(RooRealVar& xVar, RooRealVar& yVar);
      void DrawParameterVsTime(RooRealVar& param);
      void DrawNLLVsTime();
      void DrawNLLHist(const Option_t* options = NULL);
      void DrawWeightHist(const Option_t* options = NULL);

   private:

      MCMCInterval *fInterval;
      RooArgSet *fParameters;
      TH1* fPosteriorHist;
      RooNDKeysPdf* fPosteriorKeysPdf;
      RooProduct* fPosteriorKeysProduct;
      TH1* fNLLHist;
      TH1* fWeightHist;
      TH1* fPosteriorHistHistCopy;
      TH1* fPosteriorHistTFCopy;
      Int_t fDimension;
      Color_t fLineColor;
      Color_t fShadeColor;
      Int_t fLineWidth;
      bool fShowBurnIn;
      TGraph* fWalk;
      TGraph* fBurnIn;
      TGraph* fFirst;
      TGraph* fParamGraph;
      TGraph* fNLLGraph;

   protected:
      void DrawPosterior(const Option_t* options = NULL);
      void* DrawPosteriorHist(const Option_t* options = NULL,
            const char* title = NULL, bool scale = true);
      void* DrawPosteriorKeysPdf(const Option_t* options = NULL);
      void* DrawPosteriorKeysProduct(const Option_t* options = NULL);

      void DrawInterval(const Option_t* options = NULL);
      void DrawShortestInterval(const Option_t* options = NULL);
      void DrawHistInterval(const Option_t* options = NULL);
      void DrawKeysPdfInterval(const Option_t* options = NULL);
      void DrawTailFractionInterval(const Option_t* options = NULL);

      ClassDefOverride(MCMCIntervalPlot,1)  // Class containing the results of the MCMCCalculator
   };
}

#endif
