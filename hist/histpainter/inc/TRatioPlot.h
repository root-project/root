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
#include "TH1F.h"


class TBrowser;
class TFileMergeInfo;

class TRatioPlot : public TNamed {
private:
   TRatioPlot& operator=(const TRatioPlot&); // Not implemented
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


   void BuildRatio();
   void CalculateSizes();
   void SetupPads();


public:

   TRatioPlot();
   TRatioPlot(TH1* h1, TH1* h2, const char *name /*=0*/, const char *title /*=0*/, Option_t *divideOption = "");
   TRatioPlot(const TRatioPlot &hrp);
   virtual ~TRatioPlot();
   virtual void     Draw(Option_t *chopt="");
   virtual void Browse(TBrowser *b);

   virtual void SetSplitFraction(Float_t sf);
   virtual void Paint(Option_t *opt = "");



   ClassDef(TRatioPlot, 1)  //A ratio of histograms
};

#endif
