// @(#)root/hist:$Name:  $:$Id: THStack.cxx,v 1.9 2002/01/24 11:39:29 rdm Exp $
// Author: Rene Brun   10/12/2001

/*************************************************************************
 * Copyright (C) 1995-2001, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TROOT.h"
#include "THStack.h"
#include "TVirtualPad.h"
#include "TH2.h"
#include "Riostream.h"

ClassImp(THStack)

//______________________________________________________________________________
//
//   A THStack is a collection of TH1 (or derived) objects
//   Use THStack::Add to add a new histogram to the list.
//   The THStack owns the objects in the list.
//   By default (if option "nostack" is not specified), histograms will be paint
//   stacked on top of each other.
//   Example;
//      THStack hs("hs","test stacked histograms");
//      TH1F *h1 = new TH1F("h1","test hstack",100,-4,4);
//      h1->FillRandom("gaus",20000);
//      h1->SetFillColor(kRed);
//      hs.Add(h1);
//      TH1F *h2 = new TH1F("h2","test hstack",100,-4,4);
//      h2->FillRandom("gaus",15000);
//      h2->SetFillColor(kBlue);
//      hs.Add(h2);
//      TH1F *h3 = new TH1F("h3","test hstack",100,-4,4);
//      h3->FillRandom("gaus",10000);
//      h3->SetFillColor(kGreen);
//      hs.Add(h3);
//      TCanvas c1("c1","stacked hists",10,10,700,900);
//      c1.Divide(1,2);
//      c1.cd(1);
//      hs.Draw();
//      c1.cd(2);
//      hs->Draw("nostack");
//
//  See a more complex example in $ROOTSYS/tutorials/hstack.C
//
//  Note that picking is supported for all drawing modes.

//______________________________________________________________________________
THStack::THStack(): TNamed()
{
// THStack default constructor

   fHists     = 0;
   fStack     = 0;
   fHistogram = 0;
   fMaximum   = -1111;
   fMinimum   = -1111;
}

//______________________________________________________________________________
THStack::THStack(const char *name, const char *title)
       : TNamed(name,title)
{
// constructor with name and title
   fHists     = 0;
   fStack     = 0;
   fHistogram = 0;
   fMaximum   = -1111;
   fMinimum   = -1111;
}

//______________________________________________________________________________
THStack::~THStack()
{
// THStack destructor


   if (!fHists) return;
   fHists->Delete();
   delete fHists;
   fHists = 0;
   if (fStack) {fStack->Delete(); delete fStack;}
   delete fHistogram;
   fHistogram = 0;
}

//______________________________________________________________________________
void THStack::Add(TH1 *h1)
{
   // add a new histogram to the list
   // Only 1-d histograms currently supported.
   // Note that all histograms in the list must have the same number
   // of channels and the same X axis.

   if (!h1) return;
   if (h1->GetDimension() > 2) {
      Error("Add","THStack supports only 1-d and 2-d histograms");
      return;
   }
   if (!fHists) fHists = new TObjArray();
   fHists->Add(h1);
}

//______________________________________________________________________________
void THStack::Browse(TBrowser *)
{
    Draw();
    gPad->Update();
}

//______________________________________________________________________________
void THStack::BuildStack()
{
//  build sum of all histograms
//  Build a separate list fStack containing the running sum of all histograms

   if (fStack) return;
   Int_t nhists = fHists->GetEntriesFast();
   fStack = new TObjArray(nhists);
   Bool_t add = TH1::AddDirectoryStatus();
   TH1::AddDirectory(kFALSE);
   TH1 *h = (TH1*)fHists->At(0)->Clone();
   fStack->Add(h);
   for (Int_t i=1;i<nhists;i++) {
      h = (TH1*)fHists->At(i)->Clone();
      h->Add((TH1*)fStack->At(i-1));
      fStack->AddAt(h,i);
   }
   TH1::AddDirectory(add);
}

//______________________________________________________________________________
Int_t THStack::DistancetoPrimitive(Int_t px, Int_t py)
{
// Compute distance from point px,py to each graph
//

//*-*- Are we on the axis?
   const Int_t kMaxDiff = 10;
   Int_t distance = 9999;
   if (fHistogram) {
      distance = fHistogram->DistancetoPrimitive(px,py);
      if (distance <= 0) {return distance;}
      if (distance <= 1) {gPad->SetSelected(fHistogram);return distance;}
   }


//*-*- Loop on the list of histograms
   if (!fHists) return distance;
   TH1 *h = 0;
   const char *doption = GetDrawOption();
   Int_t nhists = fHists->GetEntriesFast();
   for (Int_t i=0;i<nhists;i++) {
      h = (TH1*)fHists->At(i);
      if (fStack && !strstr(doption,"nostack")) h = (TH1*)fStack->At(i);
      Int_t dist = h->DistancetoPrimitive(px,py);
      if (dist < kMaxDiff) {
         gPad->SetSelected(fHists->At(i));
         gPad->SetCursor(kPointer);
         return dist;
      }
   }
   return distance;
}

//______________________________________________________________________________
void THStack::Draw(Option_t *option)
{
//*-*-*-*-*-*-*-*-*-*-*Draw this multihist with its current attributes*-*-*-*-*-*-*
//*-*                  ==========================================
//
//   Options to draw histograms  are described in THistPainter::Paint
// By default (if option "nostack" is not specified), histograms will be paint
// stacked on top of each other.

   TString opt = option;
   opt.ToLower();
   if (gPad) {
      if (!gPad->IsEditable()) (gROOT->GetMakeDefCanvas())();
      if (!opt.Contains("same")) {
         //the following statement is necessary in case one attempts to draw
         //a temporary histogram already in the current pad
         if (TestBit(kCanDelete)) gPad->GetListOfPrimitives()->Remove(this);
         gPad->Clear();
      }
   }
   AppendPad(opt.Data());
}

//______________________________________________________________________________
TH1 *THStack::GetHistogram() const
{
//    Returns a pointer to the histogram used to draw the axis
//    Takes into account the two following cases.
//       1- option 'A' was specified in THStack::Draw. Return fHistogram
//       2- user had called TPad::DrawFrame. return pointer to hframe histogram

   if (fHistogram) return fHistogram;
   if (!gPad) return 0;
   gPad->Modified();
   gPad->Update();
   if (fHistogram) return fHistogram;
   TH1 *h1 = (TH1*)gPad->FindObject("hframe");
   return h1;
}

//______________________________________________________________________________
Double_t THStack::GetMaximum(Option_t *option)
{
//  returns the maximum of all added histograms
//  returns the maximum of all histograms if option "nostack".

   TString opt = option;
   opt.ToLower();
   Double_t them=0, themax = -1e300;
   Int_t nhists = fHists->GetEntriesFast();
   TH1 *h;
   if (!opt.Contains("nostack")) {
       BuildStack();
       h = (TH1*)fStack->At(nhists-1);
       themax = h->GetMaximum();
       if (strstr(opt.Data(),"e1")) themax += TMath::Sqrt(TMath::Abs(themax));
   } else {
      for (Int_t i=0;i<nhists;i++) {
         h = (TH1*)fHists->At(i);
         them = h->GetMaximum();
         if (strstr(opt.Data(),"e1")) them += TMath::Sqrt(TMath::Abs(them));
         if (them > themax) themax = them;
      }
   }
   return themax;
}

//______________________________________________________________________________
Double_t THStack::GetMinimum(Option_t *option)
{
//  returns the minimum of all added histograms
//  returns the minimum of all histograms if option "nostack".

   TString opt = option;
   opt.ToLower();
   Double_t them=0, themin = 1e300;
   Int_t nhists = fHists->GetEntriesFast();
   TH1 *h;
   if (!opt.Contains("nostack")) {
       BuildStack();
       h = (TH1*)fStack->At(nhists-1);
       themin = h->GetMinimum();
   } else {
      for (Int_t i=0;i<nhists;i++) {
         h = (TH1*)fHists->At(i);
         them = h->GetMinimum();
         if (them < themin) themin = them;
      }
   }
   return themin;
}

//______________________________________________________________________________
TAxis *THStack::GetXaxis() const
{
   // Get x axis of the graph.

   if (!gPad) return 0;
   return GetHistogram()->GetXaxis();
}

//______________________________________________________________________________
TAxis *THStack::GetYaxis() const
{
   // Get y axis of the graph.

   if (!gPad) return 0;
   return GetHistogram()->GetYaxis();
}

//______________________________________________________________________________
void THStack::ls(Option_t *option) const
{
   // List histograms in the stack

   TROOT::IndentLevel();
   cout <<IsA()->GetName()
        <<" Name= "<<GetName()<<" Title= "<<GetTitle()<<" Option="<<option<<endl;
   TROOT::IncreaseDirLevel();
   if (fHists) fHists->ls(option);
   TROOT::DecreaseDirLevel();
}

//______________________________________________________________________________
void THStack::Modified()
{
// invalidate sum of histograms

   if (!fStack) return;
   fStack->Delete();
   delete fStack;
   fStack = 0;
}

//______________________________________________________________________________
void THStack::Paint(Option_t *option)
{
// paint the list of histograms
// By default, histograms are shown stacked.
//    -the first histogram is paint
//    -then the sum of the first and second, etc
//
// If option "nostack" is specified, histograms are all paint in the same pad
// as if the option "same" had been specified.
//
// if option "pads" is specified, the current pad/canvas is subdivided into
// a number of pads equal to the number of histograms and each histogram 
// is paint into a separate pad.
//
// See THistPainter::Paint for a list of valid options.

   TString opt = option;
   opt.ToLower();
   
   if (opt.Contains("pads")) {
      Int_t npads = fHists->GetEntries();
      TVirtualPad *padsav = gPad;
      //if pad is not already divided into subpads, divide it
      Int_t nps = 0;
      TObject *obj;
      TIter nextp(padsav->GetListOfPrimitives());
      while ((obj = nextp())) {
         if (obj->InheritsFrom(TVirtualPad::Class())) nps++;
      }
      if (nps < npads) {
         padsav->Clear();
         Int_t nx = (Int_t)TMath::Sqrt((Double_t)npads);
         if (nx*nx < npads) nx++;
         padsav->Divide(nx,nx);
      }
      TH1 *h;
      TIter next(fHists);
      Int_t i = 0;
      while ((h=(TH1*)next())) {
         i++;
         padsav->cd(i);
         h->Draw();
      }
      padsav->cd();
      return;
   }
   
   char loption[32];
   sprintf(loption,"%s",opt.Data());
   char *nostack = strstr(loption,"nostack");
   // do not delete the stack. Another pad may contain the same object
   // drawn in stack mode!
   //if (nostack && fStack) {fStack->Delete(); delete fStack; fStack = 0;}

   if (!opt.Contains("nostack")) BuildStack();

   Double_t themax,themin;
   if (fMaximum == -1111) themax = GetMaximum(option);
   else                   themax = fMaximum;
   if (fMinimum == -1111) {themin = GetMinimum(option); if (themin > 0) themin = 0;}
   else                   themin = fMinimum;
   if (!fHistogram) {
      Bool_t add = TH1::AddDirectoryStatus();
      TH1::AddDirectory(kFALSE);
      TH1 *h = (TH1*)fHists->At(0);
      TAxis *xaxis = h->GetXaxis();
      TAxis *yaxis = h->GetYaxis();
      if (h->GetDimension() > 1) {
         if (strlen(option) == 0) strcpy(loption,"lego1");
         fHistogram = new TH2F(GetName(),GetTitle(),
                               xaxis->GetNbins(),xaxis->GetXmin(),xaxis->GetXmax(),
                               yaxis->GetNbins(),yaxis->GetXmin(),yaxis->GetXmax());
      } else {
         fHistogram = new TH1F(GetName(),GetTitle(),xaxis->GetNbins(),xaxis->GetXmin(),xaxis->GetXmax());
      }
      fHistogram->SetStats(0);
      TH1::AddDirectory(add);
   }

   if (nostack) {*nostack = 0; strcat(nostack,nostack+7);}
   else fHistogram->GetPainter()->SetStack(fHists);

   fHistogram->SetMaximum(1.05*themax);
   fHistogram->SetMinimum(themin);
   fHistogram->Paint(loption);

   if (fHistogram->GetDimension() > 1) SetDrawOption(loption);
   if (strstr(loption,"lego")) return;

   Int_t nhists = fHists->GetEntriesFast();
   strcat(loption,"same");
   for (Int_t i=0;i<nhists;i++) {
      if (nostack) fHists->At(i)->Paint(loption);
      else         fStack->At(nhists-i-1)->Paint(loption);
   }
}

//______________________________________________________________________________
void THStack::Print(Option_t *option) const
{
// Print the list of histograms

   TH1 *h;
   if (fHists) {
     TIter   next(fHists);
     while ((h = (TH1*) next())) {
       h->Print(option);
     }
   }
}

//______________________________________________________________________________
void THStack::SavePrimitive(ofstream &out, Option_t *option)
{
    // Save primitive as a C++ statement(s) on output stream out

   char quote = '"';
   out<<"   "<<endl;
   if (gROOT->ClassSaved(THStack::Class())) {
       out<<"   ";
   } else {
       out<<"   THStack *";
   }
   out<<"hstack = new THStack();"<<endl;
   out<<"   hstack->SetName("<<quote<<GetName()<<quote<<");"<<endl;
   out<<"   hstack->SetTitle("<<quote<<GetTitle()<<quote<<");"<<endl;

   TH1 *h;
   if (fHists) {
     TIter   next(fHists);
     while ((h = (TH1*) next())) {
       h->SavePrimitive(out,"nodraw");
       out<<"   hstack->Add("<<h->GetName()<<");"<<endl;
     }
   }
   out<<"   hstack->Draw("
      <<quote<<option<<quote<<");"<<endl;
}

//______________________________________________________________________________
void THStack::SetMaximum(Double_t maximum)
{
   fMaximum = maximum;
   if (fHistogram)  fHistogram->SetMaximum(maximum);
}

//______________________________________________________________________________
void THStack::SetMinimum(Double_t minimum)
{
   fMinimum = minimum;
   if (fHistogram) fHistogram->SetMinimum(minimum);
}
