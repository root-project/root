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
#include "TGraphErrors.h"
#include "TGraph.h"
#include "TAxis.h"
#include "TGaxis.h"
#include "TH1F.h"
#include "TFitResultPtr.h"

class TBrowser;
class TFileMergeInfo;

class TRatioPlot : public TPad {

private:
   TRatioPlot& operator=(const TRatioPlot&); // Not implemented
   TRatioPlot(const TRatioPlot &hrp);

   enum CalculationMode {
      kDivideHist = 1,
      kDivideGraph = 2,
      kDifference = 3,
      kFitResidual = 4,
      kDifferenceSign = 5
   };

   enum ErrorMode {
      kErrorSymmetric = 1,
      kErrorAsymmetric = 2,
      kErrorFunc = 3
   };


protected:

   TVirtualPad *fParentPad = 0;
   TPad *fUpperPad = 0;
   TPad *fLowerPad = 0;
   TPad *fTopPad = 0;

   TH1 *fH1 = 0;
   TH1 *fH2 = 0;

   Int_t fDisplayMode = 0;
   Int_t fErrorMode = TRatioPlot::ErrorMode::kErrorSymmetric;
   TString fDisplayOption = "";
   TString fOptH1 = "";
   TString fOptH2 = "";
   TString fOptGraph = "";
   


   Float_t fSplitFraction = 0.3;

   TGraph *fRatioGraph = 0;
   TGraphErrors *fConfidenceInterval1 = 0;
   TGraphErrors *fConfidenceInterval2 = 0;
   Color_t fCi1Color = kGreen;
   Color_t fCi2Color = kYellow;


   Double_t fCl1 = 0.6827;
   Double_t fCl2 = 0.9545;

   Double_t fC1 = 1.;
   Double_t fC2 = 1.;

   TFitResult *fFitResult = 0;

   TAxis *fSharedXAxis = 0;
   TGaxis *fUpperGXaxis = 0;
   TGaxis *fLowerGXaxis = 0;
   TGaxis *fUpperGYaxis = 0;
   TGaxis *fLowerGYaxis = 0;
   TGaxis *fUpperGXaxisMirror = 0;
   TGaxis *fLowerGXaxisMirror = 0;
   TGaxis *fUpperGYaxisMirror = 0;
   TGaxis *fLowerGYaxisMirror = 0;

   TAxis *fUpYaxis = 0;
   TAxis *fLowYaxis = 0;

   std::vector<TLine*> fGridlines;
   std::vector<double> fGridlinePositions;
   Bool_t fShowGridlines = kTRUE;


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

   Bool_t fIsUpdating = kFALSE;
   Bool_t fIsPadUpdating = kFALSE;
   Bool_t fPainting = kFALSE;

   virtual void SyncAxesRanges();
   virtual void SetupPads();
   virtual void CreateVisualAxes();
   virtual Bool_t SyncPadMargins();
   virtual void SetPadMargins();
   virtual void CreateGridline();


   virtual Bool_t IsDrawn();

public:

   TRatioPlot();
   virtual ~TRatioPlot();
   TRatioPlot(TH1* h1, TH1* h2, const char *name /*=0*/, const char *title /*=0*/, Option_t *displayOption = "", Option_t *optH1 = "hist", Option_t *optH2 = "E", Option_t *optGraph = "AP", Double_t c1 = 1., Double_t c2 = 1.);

   TRatioPlot(TH1* h1, const char *name, const char *title, Option_t *displayOption = "", Option_t *optH1 = "", /*Option_t *fitOpt = "L",*/ Option_t *optGraph = "LX", TFitResult *fitres = 0);

   virtual void Draw(Option_t *chopt="");
   virtual void Browse(TBrowser *b);
   
   virtual void BuildLowerPlot();
   
   virtual void Paint(Option_t *opt = "");
   virtual void PaintModified();
   
   // Slots for signal receiving
   virtual void UnZoomed();
   virtual void RangeAxisChanged();
   virtual void SubPadResized();

   // Getters
   virtual TAxis *GetXaxis() { return fSharedXAxis; }   
   virtual TAxis *GetUpYaxis() { return fUpYaxis; }
   virtual TAxis *GetLowYaxis() { return fLowYaxis; }

   virtual TGraph *GetLowerRefGraph();
   virtual TAxis *GetLowerRefXaxis();
   virtual TAxis *GetLowerRefYaxis();

   virtual TPad * GetUpperPad() { return fUpperPad; }
   virtual TPad * GetLowerPad() { return fLowerPad; }
   
   // Setters
   virtual void SetFitResult(TFitResultPtr fitres) { fFitResult = fitres.Get(); }
   virtual void SetFitResult(TFitResult *fitres) { fFitResult = fitres; }
   
   // Setters for margins
   void SetUpTopMargin(Float_t margin);
   void SetUpBottomMargin(Float_t margin);
   void SetLowTopMargin(Float_t margin);
   void SetLowBottomMargin(Float_t margin);
   void SetLeftMargin(Float_t margin);
   void SetRightMargin(Float_t margin);
   
   virtual void SetSeparationMargin(Float_t);
   virtual Float_t GetSeparationMargin();
   virtual void SetSplitFraction(Float_t sf);
   virtual void SetConfidenceLevels(Double_t cl1, Double_t cl2);

   virtual void SetLogx(Int_t value = 1); // *TOGGLE*
   virtual void SetLogy(Int_t value = 1); // *TOGGLE*

   virtual void SetGridlines(Double_t *gridlines, Int_t numGridlines); 
   virtual void SetGridlines(std::vector<double> gridlines); 

   virtual void SetConfidenceIntervalColors(Color_t ci1 = kGreen, Color_t ci2 = kYellow);

   ClassDef(TRatioPlot, 1)  //A ratio of histograms
};

#endif
