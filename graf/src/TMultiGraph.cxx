// @(#)root/graf:$Name:  $:$Id: TMultiGraph.cxx,v 1.8 2002/02/19 17:43:41 brun Exp $
// Author: Rene Brun   12/10/2000

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TROOT.h"
#include "TMultiGraph.h"
#include "TGraph.h"
#include "TH1.h"
#include "TVirtualPad.h"
#include "Riostream.h"

#include <ctype.h>


ClassImp(TMultiGraph)

//______________________________________________________________________________
//
//   A TMultiGraph is a collection of TGraph (or derived) objects
//   Use TMultiGraph::Add to add a new graph to the list.
//   The TMultiGraph owns the objects in the list.
//   Drawing options are the same as for TGraph
//   Example;
//     TGraph *gr1 = new TGraph(...
//     TGraphErrors *gr2 = new TGraphErrors(...
//     TMultiGraph *mg = new TMultiGraph();
//     mg->Add(gr1,"lp");
//     mg->Add(gr2,"cp");
//     mg->Draw("a");
//
//  The drawing option for each TGraph may be specified as an optional
//  second argument of the Add function.
//  If a draw option is specified, it will be used to draw the graph,
//  otherwise the graph will be drawn with the option specified in
//  TMultiGraph::Draw

//______________________________________________________________________________
TMultiGraph::TMultiGraph(): TNamed()
{
// TMultiGraph default constructor

   fGraphs    = 0;
   fHistogram = 0;
   fMaximum   = -1111;
   fMinimum   = -1111;
}

//______________________________________________________________________________
TMultiGraph::TMultiGraph(const char *name, const char *title)
       : TNamed(name,title)
{
// constructor with name and title
   fGraphs    = 0;
   fHistogram = 0;
   fMaximum   = -1111;
   fMinimum   = -1111;
}

//______________________________________________________________________________
TMultiGraph::~TMultiGraph()
{
// TMultiGraph destructor


   if (!fGraphs) return;
   fGraphs->Delete();
   delete fGraphs;
   fGraphs = 0;
   delete fHistogram;
   fHistogram = 0;
}

//______________________________________________________________________________
void TMultiGraph::Add(TGraph *graph, Option_t *chopt)
{
   // add a new graph to the list of graphs

   if (!fGraphs) fGraphs = new TList();
   fGraphs->Add(graph,chopt);
}

//______________________________________________________________________________
void TMultiGraph::Browse(TBrowser *)
{
    Draw("alp");
    gPad->Update();
}

//______________________________________________________________________________
Int_t TMultiGraph::DistancetoPrimitive(Int_t px, Int_t py)
{
// Compute distance from point px,py to each graph
//

//*-*- Are we on the axis?
   const Int_t kMaxDiff = 10;
   Int_t distance = 9999;
   if (fHistogram) {
      distance = fHistogram->DistancetoPrimitive(px,py);
      if (distance <= 0) return distance;
   }


//*-*- Loop on the list of graphs
   if (!fGraphs) return distance;
   TGraph *g;
   TIter   next(fGraphs);
   while ((g = (TGraph*) next())) {
      Int_t dist = g->DistancetoPrimitive(px,py);
      if (dist < kMaxDiff) {gPad->SetSelected(g); return dist;}
   }
   return distance;
}

//______________________________________________________________________________
void TMultiGraph::Draw(Option_t *option)
{
//*-*-*-*-*-*-*-*-*-*-*Draw this multigraph with its current attributes*-*-*-*-*-*-*
//*-*                  ==========================================
//
//   Options to draw a graph are described in TGraph::PainGraph
//
//  The drawing option for each TGraph may be specified as an optional
//  second argument of the Add function.
//  If a draw option is specified, it will be used to draw the graph,
//  otherwise the graph will be drawn with the option specified in
//  TMultiGraph::Draw

  AppendPad(option);
}

//______________________________________________________________________________
TH1F *TMultiGraph::GetHistogram() const
{
//    Returns a pointer to the histogram used to draw the axis
//    Takes into account the two following cases.
//       1- option 'A' was specified in TMultiGraph::Draw. Return fHistogram
//       2- user had called TPad::DrawFrame. return pointer to hframe histogram

   if (fHistogram) return fHistogram;
   if (!gPad) return 0;
   gPad->Modified();
   gPad->Update();
   if (fHistogram) return fHistogram;
   TH1F *h1 = (TH1F*)gPad->FindObject("hframe");
   return h1;
}

//______________________________________________________________________________
TAxis *TMultiGraph::GetXaxis() const
{
   // Get x axis of the graph.

   if (!gPad) return 0;
   TH1 *h = GetHistogram();
   if (!h) return 0;
   return h->GetXaxis();
}

//______________________________________________________________________________
TAxis *TMultiGraph::GetYaxis() const
{
   // Get y axis of the graph.

   if (!gPad) return 0;
   TH1 *h = GetHistogram();
   if (!h) return 0;
   return h->GetYaxis();
}

//______________________________________________________________________________
void TMultiGraph::Paint(Option_t *option)
{
// paint all the graphs of this multigraph

  char *l;
  static char chopt[33];
  Int_t nch = strlen(option);
  Int_t i;
  for (i=0;i<nch;i++) chopt[i] = toupper(option[i]);
  chopt[nch] = 0;
  Double_t *x, *y;
  TGraph *g;

  l = strstr(chopt,"A");
  if (l) {
     *l = ' ';
     TIter   next(fGraphs);
     Int_t npt = 100;
     Double_t maximum, minimum, rwxmin, rwxmax, rwymin, rwymax, uxmin, uxmax, dx, dy;
     rwxmin    = gPad->GetUxmin();
     rwxmax    = gPad->GetUxmax();
     rwymin    = gPad->GetUymin();
     rwymax    = gPad->GetUymax();
     char *xtitle = 0;
     char *ytitle = 0;
     Int_t firstx = 0;
     Int_t lastx  = 0;
     
     if (fHistogram) {
        //cleanup in case of a previous unzoom
        if (fHistogram->GetMinimum() >= fHistogram->GetMaximum()) {
           Int_t nch = strlen(fHistogram->GetXaxis()->GetTitle());
           firstx = fHistogram->GetXaxis()->GetFirst();
           lastx  = fHistogram->GetXaxis()->GetLast();
           if (nch) {
              xtitle = new char[nch+1];
              strcpy(xtitle,fHistogram->GetXaxis()->GetTitle());
           }
           nch = strlen(fHistogram->GetYaxis()->GetTitle());
           if (nch) {
              ytitle = new char[nch+1];
              strcpy(ytitle,fHistogram->GetYaxis()->GetTitle());
           }
           delete fHistogram;
           fHistogram = 0;
        }
     }
     if (fHistogram) {
        minimum = fHistogram->GetYaxis()->GetXmin();
        maximum = fHistogram->GetYaxis()->GetXmax();
        uxmin   = gPad->PadtoX(rwxmin);
        uxmax   = gPad->PadtoX(rwxmax);
     } else {
        rwxmin = 1e100;
        rwxmax = -rwxmin;
        rwymin = rwxmin;
        rwymax = -rwymin;
        while ((g = (TGraph*) next())) {
           Int_t npoints = g->GetN();
           x = g->GetX();
           y = g->GetY();
           for (i=0;i<npoints;i++) {
              if (x[i] < rwxmin) rwxmin = x[i];
              if (x[i] > rwxmax) rwxmax = x[i];
              if (y[i] < rwymin) rwymin = y[i];
              if (y[i] > rwymax) rwymax = y[i];
           }
           g->ComputeRange(rwxmin, rwymin, rwxmax, rwymax);
           if (g->GetN() > npt) npt = g->GetN();
        }
        if (rwxmin == rwxmax) rwxmax += 1.;
        if (rwymin == rwymax) rwymax += 1.;
        dx = 0.05*(rwxmax-rwxmin);
        dy = 0.05*(rwymax-rwymin);
        uxmin    = rwxmin - dx;
        uxmax    = rwxmax + dx;
        minimum  = rwymin - dy;
        maximum  = rwymax + dy;
     }

     if (fMinimum != -1111) rwymin = minimum = fMinimum;
     if (fMaximum != -1111) rwymax = maximum = fMaximum;
     if (uxmin < 0 && rwxmin >= 0) {
        if (gPad->GetLogx()) uxmin = 0.9*rwxmin;
        //else                 uxmin = 0;
     }
     if (uxmax > 0 && rwxmax <= 0) {
        if (gPad->GetLogx()) uxmax = 1.1*rwxmax;
        //else                 uxmax = 0;
     }
     if (minimum < 0 && rwymin >= 0) {
        if(gPad->GetLogy()) minimum = 0.9*rwymin;
        //else                minimum = 0;
     }
     if (maximum > 0 && rwymax <= 0) {
        if(gPad->GetLogy()) maximum = 1.1*rwymax;
        //else                maximum = 0;
     }
     if (minimum <= 0 && gPad->GetLogy()) minimum = 0.001*maximum;
     if (uxmin <= 0 && gPad->GetLogx()) {
        if (uxmax > 1000) uxmin = 1;
        else              uxmin = 0.001*uxmax;
     }
     rwymin = minimum;
     rwymax = maximum;
     if (fHistogram) {
        fHistogram->GetYaxis()->SetLimits(rwymin,rwymax);
     }

//*-*-  Create a temporary histogram to draw the axis
     if (!fHistogram) {
        // the graph is created with at least as many channels as there are points
        // to permit zooming on the full range
        rwxmin = uxmin;
        rwxmax = uxmax;
        fHistogram = new TH1F(GetName(),GetTitle(),npt,rwxmin,rwxmax);
        if (!fHistogram) return;
        fHistogram->SetMinimum(rwymin);
        fHistogram->SetBit(TH1::kNoStats);
        fHistogram->SetMaximum(rwymax);
        fHistogram->GetYaxis()->SetLimits(rwymin,rwymax);
        fHistogram->SetDirectory(0);
        if (xtitle) {fHistogram->GetXaxis()->SetTitle(xtitle); delete [] xtitle;}
        if (ytitle) {fHistogram->GetYaxis()->SetTitle(ytitle); delete [] ytitle;}
        if (firstx != lastx) fHistogram->GetXaxis()->SetRange(firstx,lastx);
     }
     fHistogram->Paint("0");
   }

   if (fGraphs) {
      TObjOptLink *lnk = (TObjOptLink*)fGraphs->FirstLink();
      TObject *obj;

      while (lnk) {
         obj = lnk->GetObject();
         if (strlen(lnk->GetOption())) obj->Paint(lnk->GetOption());
         else                          obj->Paint(chopt);
         lnk = (TObjOptLink*)lnk->Next();
      }
   }
}

//______________________________________________________________________________
void TMultiGraph::Print(Option_t *option) const
{
// Print the list of graphs

   TGraph *g;
   if (fGraphs) {
     TIter   next(fGraphs);
     while ((g = (TGraph*) next())) {
       g->Print(option);
     }
   }
}

//______________________________________________________________________________
void TMultiGraph::SavePrimitive(ofstream &out, Option_t *option)
{
    // Save primitive as a C++ statement(s) on output stream out

   char quote = '"';
   out<<"   "<<endl;
   if (gROOT->ClassSaved(TMultiGraph::Class())) {
       out<<"   ";
   } else {
       out<<"   TMultiGraph *";
   }
   out<<"multigraph = new TMultiGraph();"<<endl;
   out<<"   multigraph->SetName("<<quote<<GetName()<<quote<<");"<<endl;
   out<<"   multigraph->SetTitle("<<quote<<GetTitle()<<quote<<");"<<endl;

   TGraph *g;
   if (fGraphs) {
     TIter   next(fGraphs);
     while ((g = (TGraph*) next())) {
       g->SavePrimitive(out,"multigraph");
     }
   }
   out<<"   multigraph->Draw("
      <<quote<<option<<quote<<");"<<endl;
}

//______________________________________________________________________________
void TMultiGraph::SetMaximum(Double_t maximum)
{
   fMaximum = maximum;
   if (fHistogram)  fHistogram->SetMaximum(maximum);
}

//______________________________________________________________________________
void TMultiGraph::SetMinimum(Double_t minimum)
{
   fMinimum = minimum;
   if (fHistogram) fHistogram->SetMinimum(minimum);
}
