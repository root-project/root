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
#include "TGaxis.h"
#include "TH1F.h"


class TBrowser;
class TFileMergeInfo;

class TRatioPlot : public TPad {

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

   TAxis *fUpYaxis;
   TAxis *fLowYaxis;

   // store y axis ranges so we can trigger redraw when they change
   Double_t fUpYFirst = -1;
   Double_t fUpYLast = -1;
   Double_t fLowYFirst = -1;
   Double_t fLowYLast = -1;

   // store margins to be able do determine
   // what has changed when user drags
   Float_t fUpTopMargin = 0.1;
   Float_t fUpBottomMargin = 0.05;
   Float_t fUpBottomMarginNominal = 0.05;

   Float_t fLowTopMargin = 0.05;
   Float_t fLowTopMarginNominal = 0.05;
   Float_t fLowBottomMargin = 0.3;

   Float_t fLeftMargin = 0.1;
   Float_t fRightMargin = 0.1;

   Float_t fSeparationMargin;
   Bool_t fIsUpdating = kFALSE;
   Bool_t fIsPadUpdating = kFALSE;
   Bool_t fPainting = kFALSE;

   virtual void BuildRatio();
   virtual void SyncAxesRanges();
   virtual void SetupPads();
   virtual void CreateVisualAxes();
   virtual Bool_t SyncPadMargins();
   virtual void SetPadMargins();

public:

   TRatioPlot();
   TRatioPlot(TH1* h1, TH1* h2, const char *name /*=0*/, const char *title /*=0*/, Option_t *divideOption = "");
   virtual void Draw(Option_t *chopt="");
   virtual void Browse(TBrowser *b);

   // Setters for margins
   void SetUpTopMargin(Float_t margin);
   void SetUpBottomMargin(Float_t margin);
   void SetLowTopMargin(Float_t margin);
   void SetLowBottomMargin(Float_t margin);
   void SetLeftMargin(Float_t margin);
   void SetRightMargin(Float_t margin);
   
   virtual void SetSplitFraction(Float_t sf);
   virtual void Paint(Option_t *opt = "");
   virtual void PaintModified();

   // Slots for signal receiving
   virtual void UnZoomed();
   virtual void RangeAxisChanged();
   virtual void SubPadResized();

   virtual TAxis *GetXaxis() { return fSharedXAxis; }   
   virtual TAxis *GetUpYaxis() { return fUpYaxis; }
   virtual TAxis *GetLowYaxis() { return fLowYaxis; }

   virtual TGraph *GetRatioGraph() { return fRatioGraph; }

   ClassDef(TRatioPlot, 1)  //A ratio of histograms
};

#endif
