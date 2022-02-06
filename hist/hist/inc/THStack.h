// @(#)root/hist:$Id$
// Author: Rene Brun   10/12/2001

/*************************************************************************
 * Copyright (C) 1995-2001, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_THStack
#define ROOT_THStack


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// THStack                                                              //
//                                                                      //
// A collection of histograms                                           //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TNamed.h"
#include "TObjArray.h"

#ifdef R__LESS_INCLUDES
class TH1;
class TList;
class TAxis;
#else
#include "TH1.h"
#endif

class TBrowser;
class TFileMergeInfo;

class THStack : public TNamed {
private:
   THStack& operator=(const THStack&) = delete;

protected:
   TList      *fHists{nullptr};      ///<  Pointer to array of TH1
   TObjArray  *fStack{nullptr};      ///<! Pointer to array of sums of TH1
   TH1        *fHistogram{nullptr};  ///<  Pointer to histogram used for drawing axis
   Double_t    fMaximum{-1111};      ///<  Maximum value for plotting along y
   Double_t    fMinimum{-1111};      ///<  Minimum value for plotting along y

   void BuildStack();

public:

   THStack() {}
   THStack(const char *name, const char *title);
   THStack(TH1* hist, Option_t *axis="x",
           const char *name = nullptr, const char *title = nullptr,
           Int_t firstbin = 1, Int_t lastbin = -1,
           Int_t firstbin2 = 1, Int_t lastbin2 = -1,
           Option_t *proj_option = "", Option_t *draw_option = "");
   THStack(const THStack &hstack);
   ~THStack() override;
   virtual void     Add(TH1 *h, Option_t *option="");
   void             Browse(TBrowser *b)  override;
   Int_t            DistancetoPrimitive(Int_t px, Int_t py) override;
   void             Draw(Option_t *chopt="")  override;
   TH1             *GetHistogram() const;
   TList           *GetHists()  const { return fHists; }
   TIter            begin() const;
   TIter            end() const { return TIter::End(); }
   Int_t            GetNhists() const;
   TObjArray       *GetStack();
   virtual Double_t GetMaximum(Option_t *option="");
   virtual Double_t GetMinimum(Option_t *option="");
   TAxis           *GetXaxis() const;
   TAxis           *GetYaxis() const;
   TAxis           *GetZaxis() const;
   void             ls(Option_t *option="") const  override;
   virtual Long64_t Merge(TCollection* li, TFileMergeInfo *info);
   virtual void     Modified();
   void             Paint(Option_t *chopt = "") override;
   void             Print(Option_t *chopt = "") const override;
   void             RecursiveRemove(TObject *obj)  override;
   void             SavePrimitive(std::ostream &out, Option_t *option = "")  override;
   virtual void     SetHistogram(TH1 *h) { fHistogram = h; }
   virtual void     SetMaximum(Double_t maximum=-1111); // *MENU*
   virtual void     SetMinimum(Double_t minimum=-1111); // *MENU*

   ClassDefOverride(THStack,2)  //A collection of histograms
};

#endif

