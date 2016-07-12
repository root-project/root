// @(#)root/hist:$Id$
// Author: Rene Brun   10/12/2001

/*************************************************************************
 * Copyright (C) 1995-2001, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TRatioPlot
#define ROOT_TRatioPlot


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// ROOT_TRatioPlot                                                      //
//                                                                      //
// A collection of histograms                                           //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TH1.h"
#include "TPad.h"
#include "TGraphAsymmErrors.h"
#include "TGraph.h"
#include "TAxis.h"
#include "TGAxis.h"
#include "TH1F.h"


class TBrowser;
class TFileMergeInfo;

class TRatioPlot : /*public TNamed,*/ public TPad {

private:
   TRatioPlot& operator=(const TRatioPlot&); // Not implemented
   TRatioPlot(const TRatioPlot &hrp);
   static const Int_t DIVIDE_HIST = 1;
   static const Int_t DIVIDE_GRAPH = 2;

protected:

   TVirtualPad *fParentPad;
   TPad *fUpperPad;
   TPad *fLowerPad;
   TPad *fTopPad;

   TH1 *fH1;
   TH1 *fH2;

   Int_t fDivideMode;
   TString fDivideOption;

   Float_t fSplitFraction = 0.3;

   TGraph *fRatioGraph;
   TAxis *fSharedXAxis;
   TGaxis *fUpperGXaxis;
   TGaxis *fLowerGXaxis;
   TGaxis *fUpperGYaxis;
   TGaxis *fLowerGYaxis;

   Double_t fUpTopMargin = 0.1;
   Double_t fUpBottomMargin = 0.05;
   Double_t fUpBottomMarginNominal = 0.05;

   Double_t fLowTopMargin = 0.05;
   Double_t fLowTopMarginNominal = 0.05;
   Double_t fLowBottomMargin = 0.3;

   Double_t fLeftMargin = 0.1;
   Double_t fRightMargin = 0.1;

   Double_t fSeparationMargin;
   Bool_t fIsUpdating = kFALSE;

   virtual void BuildRatio();
   virtual void SyncAxesRanges();
   virtual void SetupPads();
   virtual void CreateVisualAxes();
   virtual Bool_t SyncPadMargins();
   virtual void SetPadMargins();

public:

   TRatioPlot();
   TRatioPlot(TH1* h1, TH1* h2, const char *name /*=0*/, const char *title /*=0*/, Option_t *divideOption = "");
//   virtual ~TRatioPlot();
   virtual void     Draw(Option_t *chopt="");
   virtual void Browse(TBrowser *b);

   virtual void SetSplitFraction(Float_t sf);
   virtual void Paint(Option_t *opt = "");

   virtual void UnZoomed();


   virtual void RangeAxisChanged();

   ClassDef(TRatioPlot, 1)  //A ratio of histograms
};

#endif
