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
#include "RooStats/MCMCInterval.h"
#include "RooNDKeysPdf.h"
#include "RooProduct.h"

#include <TColor.h>
#include <TH1.h>
#include <TNamed.h>

namespace RooStats {

   class MCMCIntervalPlot : public TNamed, public RooPrintable {

   public:
      MCMCIntervalPlot();
      MCMCIntervalPlot(MCMCInterval& interval);

      /// Destructor of SamplingDistribution
      ~MCMCIntervalPlot() override;

      void SetMCMCInterval(MCMCInterval& interval);
      void SetLineColor(TColorNumber color) {fLineColor = color.number();}
      void SetLineWidth(Int_t width) {fLineWidth = width;}
      void SetShadeColor(TColorNumber color) {fShadeColor = color.number();}
      void SetShowBurnIn(bool showBurnIn) { fShowBurnIn = showBurnIn; }

      void Draw(const Option_t* options = nullptr) override;

      void DrawChainScatter(RooRealVar& xVar, RooRealVar& yVar);
      void DrawParameterVsTime(RooRealVar& param);
      void DrawNLLVsTime();
      void DrawNLLHist(const Option_t* options = nullptr);
      void DrawWeightHist(const Option_t* options = nullptr);

   private:
      MCMCInterval *fInterval = nullptr;
      RooArgSet *fParameters = nullptr;
      TH1 *fPosteriorHist = nullptr;
      RooNDKeysPdf *fPosteriorKeysPdf = nullptr;
      RooProduct *fPosteriorKeysProduct = nullptr;
      TH1 *fNLLHist = nullptr;
      TH1 *fWeightHist = nullptr;
      TH1 *fPosteriorHistHistCopy = nullptr;
      TH1 *fPosteriorHistTFCopy = nullptr;
      Int_t fDimension = 0;
      Color_t fLineColor = kBlack;
      Color_t fShadeColor = kGray;
      Int_t fLineWidth = 1;
      bool fShowBurnIn = true;
      TGraph *fWalk = nullptr;
      TGraph *fBurnIn = nullptr;
      TGraph *fFirst = nullptr;
      TGraph *fParamGraph = nullptr;
      TGraph *fNLLGraph = nullptr;

   protected:
      void DrawPosterior(const Option_t* options = nullptr);
      void* DrawPosteriorHist(const Option_t* options = nullptr,
            const char* title = nullptr, bool scale = true);
      void* DrawPosteriorKeysPdf(const Option_t* options = nullptr);
      void* DrawPosteriorKeysProduct(const Option_t* options = nullptr);

      void DrawInterval(const Option_t* options = nullptr);
      void DrawShortestInterval(const Option_t* options = nullptr);
      void DrawHistInterval(const Option_t* options = nullptr);
      void DrawKeysPdfInterval(const Option_t* options = nullptr);
      void DrawTailFractionInterval(const Option_t* options = nullptr);

      ClassDefOverride(MCMCIntervalPlot,1)  // Class containing the results of the MCMCCalculator
   };
}

#endif
