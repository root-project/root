// @(#)root/gpad:$Id$
// Author: Paul Gessinger   25/08/2016

/*************************************************************************
 * Copyright (C) 1995-2021, Rene Brun and Fons Rademakers.               *
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

#include "TObject.h"

#include "TString.h"

#include "TGraph.h"

class TH1;
class TPad;
class TVirtualPad;
class TGraphAsymmErrors;
class TGraphErrors;
class TAxis;
class TGaxis;
class TLine;
class TFitResultPtr;
class TFitResult;
class THStack;
class TBrowser;

class TRatioPlot : public TObject {

private:
   TRatioPlot& operator=(const TRatioPlot&) = delete;
   TRatioPlot(const TRatioPlot &) = delete;

   enum CalculationMode {
      kDivideHist = 1, ///< Use `TH1::Divide` to create the ratio.
      kDivideGraph = 2, ///< Use `TGraphAsymmErrors::Divide` to create the ratio.
      kDifference = 3, ///< Calculate the difference between the histograms.
      kFitResidual = 4, ///< Calculate the fit residual between the histogram and a fit stored within it.
      kDifferenceSign = 5 ///< Calculate the difference divided by the error.
   };

   enum ErrorMode {
      kErrorSymmetric = 1, ///< Use the regular `TH1::GetBinError` as the error
      kErrorAsymmetric = 2, ///< Use `TH1::GetBinErrorUp` and `TH1::GetBinErrorLow` for the error, depending on y values.
      kErrorFunc = 3 ///< Use the square root of the function value as the error.
   };

   enum HideLabelMode {
      kHideUp = 1, ///< Hide the first label of the upper y axis when there is low space.
      kHideLow = 2, ///< Hide the last label of the lower y axis when there is low space.
      kNoHide = 3, ///< Do not hide labels when there is low space.
      kForceHideUp = 4, ///< Always hide the first label of the upper y axis
      kForceHideLow = 5 ///< Always hide the last label of the lower y axis
   };


protected:

   TVirtualPad *fParentPad = nullptr; ///< Stores the pad the ratio plot was created in
   TPad *fUpperPad = nullptr; ///< The pad which contains the upper plot part
   TPad *fLowerPad = nullptr; ///< The pad which contains the calculated lower plot part
   TPad *fTopPad = nullptr; ///< The Pad that drawn on top on the others to have consistent coordinates

   TH1 *fH1 = nullptr; ///< Stores the primary histogram
   TH1 *fH2 = nullptr; ///< Stores the secondary histogram, if there is one
   TObject *fHistDrawProxy = nullptr; ///< The object which is actually drawn, this might be TH1 or THStack

   Int_t fMode = 0; ///< Stores which calculation is supposed to be performed as specified by user option
   Int_t fErrorMode = TRatioPlot::ErrorMode::kErrorSymmetric; ///< Stores the error mode, sym, asym or func
   TString fOption = ""; ///< Stores the option which is given in the constructor as a string
   TString fH1DrawOpt = ""; ///< Stores draw option for h1 given in constructor
   TString fH2DrawOpt = ""; ///< Stores draw option for h2 given in constructor
   TString fGraphDrawOpt = ""; ///< Stores draw option for the lower plot graph given in constructor
   TString fFitDrawOpt = ""; ///< Stores draw option for the fit function in the fit residual case

   Float_t fSplitFraction = 0.3; ///< Stores the fraction at which the upper and lower pads meet

   TGraph *fRatioGraph = nullptr; ///< Stores the lower plot's graph
   TGraphErrors *fConfidenceInterval1 = nullptr; ///< Stores the graph for the 1 sigma band
   TGraphErrors *fConfidenceInterval2 = nullptr; ///< Stores the graph for the 2 sigma band
   Color_t fCi1Color = kYellow; ///< Stores the color for the 1 sigma band
   Color_t fCi2Color = kGreen; ///< Stores the color for the 2 sigma band

   Bool_t fShowConfidenceIntervals = kTRUE; ///< Stores whether to show the confidence interval bands. From Draw option

   Double_t fCl1 = 0.6827; ///< Stores the confidence level for the inner confidence interval band
   Double_t fCl2 = 0.9545; ///< Stores the confidence level for the outer confidence interval band

   Double_t fC1 = 1.; ///< Stores the scale factor for h1 (or THStack sum)
   Double_t fC2 = 1.; ///< Stores the scale factor for h2

   TFitResult *fFitResult = nullptr; ///< Stores the explicit fit result given in the fit residual case. Can be 0

   TAxis *fSharedXAxis = nullptr; ///< X axis that stores the range for both plots
   TGaxis *fUpperGXaxis = nullptr; ///< Upper graphical x axis
   TGaxis *fLowerGXaxis = nullptr; ///< Lower graphical x axis
   TGaxis *fUpperGYaxis = nullptr; ///< Upper graphical y axis
   TGaxis *fLowerGYaxis = nullptr; ///< Lower graphical y axis
   TGaxis *fUpperGXaxisMirror = nullptr; ///< Upper mirror of the x axis
   TGaxis *fLowerGXaxisMirror = nullptr; ///< Lower mirror of the x axis
   TGaxis *fUpperGYaxisMirror = nullptr; ///< Upper mirror of the y axis
   TGaxis *fLowerGYaxisMirror = nullptr; ///< Lower mirror of the y axis

   TAxis *fUpYaxis = nullptr; ///< Clone of the upper y axis
   TAxis *fLowYaxis = nullptr; ///< Clone of the lower y axis

   std::vector<TLine*> fGridlines; ///< Keeps TLine objects for the gridlines
   std::vector<double> fGridlinePositions; ///< Stores the y positions for the gridlines
   Bool_t fShowGridlines = kTRUE; ///< Stores whether to show the gridlines at all
   Int_t fHideLabelMode = TRatioPlot::HideLabelMode::kHideLow; ///< Stores which label to hide if the margin is to narrow, if at all

   // store margins to be able do determine
   // what has changed when user drags
   Float_t fUpTopMargin = 0.1; ///< Stores the top margin of the upper pad
   Float_t fUpBottomMargin = 0.05; ///< Stores the bottom margin of the upper pad
   Float_t fLowTopMargin = 0.05; ///< Stores the top margin of the lower pad
   Float_t fLowBottomMargin = 0.3; ///< Stores the bottom margin of the lower pad

   Float_t fLeftMargin = 0.1; ///< Stores the common left margin of both pads
   Float_t fRightMargin = 0.1; ///< Stores the common right margin of both pads

   Float_t fInsetWidth = 0.0025;

   Bool_t fIsUpdating = kFALSE; ///< Keeps track of whether its currently updating to reject other calls until done
   Bool_t fIsPadUpdating = kFALSE; ///< Keeps track whether pads are updating during resizing

   virtual void SyncAxesRanges();
   virtual void SetupPads();
   virtual void CreateVisualAxes();
   virtual Bool_t SyncPadMargins();
   void SetPadMargins();
   void CreateGridline();
   Int_t BuildLowerPlot();

   void ImportAxisAttributes(TGaxis* gaxis, TAxis* axis);

   Bool_t IsDrawn();

   virtual void Init(TH1* h1, TH1* h2, Option_t *option = "");

public:

   TRatioPlot();
   virtual ~TRatioPlot();
   TRatioPlot(TH1* h1, TH1* h2, Option_t *option = "pois");

   TRatioPlot(THStack* st, TH1* h2, Option_t *option = "pois");

   TRatioPlot(TH1* h1, Option_t *option = "", TFitResult *fitres = nullptr);

   void SetH1DrawOpt(Option_t *opt);
   void SetH2DrawOpt(Option_t *opt);
   void SetGraphDrawOpt(Option_t *opt);
   void SetFitDrawOpt(Option_t *opt);

   void SetInsetWidth(Double_t width);

   void Draw(Option_t *chopt = "") override;
   void Browse(TBrowser *b) override;


   void Paint(Option_t *opt = "") override;

   // Slots for signal receiving
   void UnZoomed();
   void RangeAxisChanged();
   void SubPadResized();

   // Getters
   TAxis *GetXaxis() const { return fSharedXAxis; }
   TAxis *GetUpYaxis() const { return fUpYaxis; }
   TAxis *GetLowYaxis() const { return fLowYaxis; }

   virtual TGraph *GetLowerRefGraph() const;

   ////////////////////////////////////////////////////////////////////////////////
   /// Shortcut for:
   ///
   /// ~~~{.cpp}
   /// rp->GetLowerRefGraph()->GetXaxis();
   /// ~~~

   TAxis *GetLowerRefXaxis() const { return GetLowerRefGraph()->GetXaxis(); }

   ////////////////////////////////////////////////////////////////////////////////
   /// Shortcut for:
   ///
   /// ~~~{.cpp}
   /// rp->GetLowerRefGraph()->GetYaxis();
   /// ~~~

   TAxis *GetLowerRefYaxis() const { return GetLowerRefGraph()->GetYaxis(); }

   virtual TObject *GetUpperRefObject() const;
   TAxis *GetUpperRefXaxis() const;
   TAxis *GetUpperRefYaxis() const;

   ////////////////////////////////////////////////////////////////////////////////
   /// Get the output of the calculation in the form of a graph. The type of
   /// the return value depends on the input option that was given in the constructor.

   TGraph *GetCalculationOutputGraph() const { return fRatioGraph; }

   ////////////////////////////////////////////////////////////////////////////////
   /// Returns the graph for the 1 sigma confidence interval in the fit residual case

   TGraphErrors *GetConfidenceInterval1() const { return fConfidenceInterval1; }

   ////////////////////////////////////////////////////////////////////////////////
   /// Returns the graph for the 2 sigma confidence interval in the fit residual case

   TGraphErrors *GetConfidenceInterval2() const { return fConfidenceInterval2; }

   TPad * GetUpperPad() const { return fUpperPad; }
   TPad * GetLowerPad() const { return fLowerPad; }

   // Setters

   ////////////////////////////////////////////////////////////////////////////////
   /// Explicitly specify the fit result that is to be used for fit residual calculation.
   /// If it is not provided, the last fit registered in the global fitter is used.
   /// The fit result can also be specified in the constructor.
   ///
   /// \param fitres The fit result coming from the fit function call

   void SetFitResult(TFitResultPtr fitres) { fFitResult = fitres.Get(); }

   // Setters for margins
   void SetUpTopMargin(Float_t margin);
   void SetUpBottomMargin(Float_t margin);
   void SetLowTopMargin(Float_t margin);
   void SetLowBottomMargin(Float_t margin);
   void SetLeftMargin(Float_t margin);
   void SetRightMargin(Float_t margin);

   void SetSeparationMargin(Float_t);
   Float_t GetSeparationMargin() const;
   void SetSplitFraction(Float_t sf);
   void SetConfidenceLevels(Double_t cl1, Double_t cl2);

   virtual void SetGridlines(Double_t *gridlines, Int_t numGridlines);
   virtual void SetGridlines(std::vector<double> gridlines);

   void SetConfidenceIntervalColors(Color_t ci1 = kYellow, Color_t ci2 = kGreen);

   void SetC1(Double_t c1) { fC1 = c1; }
   void SetC2(Double_t c2) { fC2 = c2; }

   ClassDefOverride(TRatioPlot, 1)  //A ratio of histograms
};

#endif
