// @(#)root/hist:$Id$
// Author: Olivier Couet   18/05/2022

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/


#include "TROOT.h"
#include "TBuffer.h"
#include "TScatter.h"
#include "TStyle.h"
#include "TMath.h"
#include "TVirtualPad.h"
#include "TH2.h"
#include "TVirtualGraphPainter.h"
#include "strtok.h"

#include <iostream>
#include <fstream>
#include <cstring>
#include <string>

ClassImp(TScatter);


////////////////////////////////////////////////////////////////////////////////

/** \class TScatter
    \ingroup Graphs
A TScatter is able to draw four variables scatter plot on a single plot. The two first
variables are the x and y position of the markers and the third is mapped on the current
color map and the fourth on the marker size.

The following example demonstrates how it works:

Begin_Macro(source)
../../../tutorials/graphs/scatter.C
End_Macro

*/


////////////////////////////////////////////////////////////////////////////////
/// TScatter default constructor.

TScatter::TScatter()
{
   fNpoints   = -1;
   fMaxSize   = -1;

   fGraph     = nullptr;
   fHistogram = nullptr;
   fScale     = 5.;
   fMargin    = 0.1;
   fSize      = nullptr;
   fColor     = nullptr;
}


////////////////////////////////////////////////////////////////////////////////
/// TScatter normal constructor.
///
///  the arrays are preset to zero

TScatter::TScatter(Int_t n)
{
   fGraph     = new TGraph(n);
   fNpoints   = fGraph->GetN();
   fMaxSize   = fGraph->GetMaxSize();
   fHistogram = nullptr;

   fColor = new Double_t[fMaxSize];
   fSize  = new Double_t[fMaxSize];

   memset(fColor, 0, fNpoints * sizeof(Double_t));
   memset(fSize, 0, fNpoints * sizeof(Double_t));
   fScale  = 5.;
   fMargin = 0.1;
}


////////////////////////////////////////////////////////////////////////////////
/// TScatter normal constructor.
///
///  if ex or ey are null, the corresponding arrays are preset to zero

TScatter::TScatter(Int_t n, const Double_t *x, const Double_t *y, const Double_t *col, const Double_t *size)
{
   fGraph     = new TGraph(n, x, y);
   fNpoints   = fGraph->GetN();
   fMaxSize   = fGraph->GetMaxSize();
   fHistogram = nullptr;

   fColor = new Double_t[fMaxSize];
   fSize  = new Double_t[fMaxSize];

   n = sizeof(Double_t) * fNpoints;
   if (col) memcpy(fColor, col, n);
   else     fColor = nullptr;
   if (size) memcpy(fSize, size, n);
   else      fSize = nullptr;

   fScale  = 5.;
   fMargin = 0.1;
}


////////////////////////////////////////////////////////////////////////////////
/// TScatter default destructor.

TScatter::~TScatter()
{
   delete fGraph;
   delete fHistogram;
   delete [] fColor;
   delete [] fSize;
}


////////////////////////////////////////////////////////////////////////////////
/// Compute distance from point px,py to a scatter plot.
///
///  Compute the closest distance of approach from point px,py to this scatter plot.
///  The distance is computed in pixels units.

Int_t TScatter::DistancetoPrimitive(Int_t px, Int_t py)
{
   TVirtualGraphPainter *painter = TVirtualGraphPainter::GetPainter();
   if (painter) return painter->DistancetoPrimitiveHelper(this->GetGraph(), px, py);
   else return 0;
}


////////////////////////////////////////////////////////////////////////////////
/// Execute action corresponding to one event.
///
///  This member function is called when a graph is clicked with the locator
///
///  If Left button clicked on one of the line end points, this point
///     follows the cursor until button is released.
///
///  if Middle button clicked, the line is moved parallel to itself
///     until the button is released.

void TScatter::ExecuteEvent(Int_t event, Int_t px, Int_t py)
{
   TVirtualGraphPainter *painter = TVirtualGraphPainter::GetPainter();
   if (painter) painter->ExecuteEventHelper(this->GetGraph(), event, px, py);
}


////////////////////////////////////////////////////////////////////////////////
/// Returns a pointer to the histogram used to draw the axis

TH2F *TScatter::GetHistogram() const
{
   if (!fHistogram) {
      // do not add the histogram to gDirectory
      // use local TDirectory::TContect that will set temporarly gDirectory to a nullptr and
      // will avoid that histogram is added in the global directory
      {
         TDirectory::TContext ctx(nullptr);
         double rwxmin, rwymin, rwxmax, rwymax;
         int npt = 50;
         fGraph->ComputeRange(rwxmin, rwymin, rwxmax, rwymax);
         double dx = (rwxmax-rwxmin)*fMargin;
         double dy = (rwymax-rwymin)*fMargin;
         auto h = new TH2F(TString::Format("%s_h",GetName()),GetTitle(),npt,rwxmin,rwxmax,npt,rwymin,rwymax);
//          h->SetMinimum(rwymin-dy);
//          h->SetMaximum(rwymax+dy);
         h->GetXaxis()->SetLimits(rwxmin-dx,rwxmax+dx);
         h->GetYaxis()->SetLimits(rwymin-dy,rwymax+dy);
         h->SetBit(TH1::kNoStats);
         h->SetDirectory(0);
         h->Sumw2(kFALSE);
         ((TScatter*)this)->fHistogram = h;
      }
   }
   return fHistogram;
}


////////////////////////////////////////////////////////////////////////////////
/// Paint this scatter plot with its current attributes.

void TScatter::Paint(Option_t *option)
{
   TVirtualGraphPainter *painter = TVirtualGraphPainter::GetPainter();
   if (painter) painter->PaintScatter(this, option);
}


////////////////////////////////////////////////////////////////////////////////
/// Print graph and errors values.

void TScatter::Print(Option_t *) const
{
   Double_t *X = fGraph->GetX();
   Double_t *Y = fGraph->GetY();
   for (Int_t i = 0; i < fNpoints; i++) {
      printf("x[%d]=%g, y[%d]=%g, color[%d]=%g, size[%d]=%g\n", i, X[i], i, Y[i], i, fColor[i], i, fSize[i]);
   }
}


////////////////////////////////////////////////////////////////////////////////
/// Set the margin around the plot in %

void TScatter::SetMargin(Double_t margin)
{
   if (fMargin != margin) {
      delete fHistogram;
      fHistogram = nullptr;
      fMargin = margin;
   }
}


////////////////////////////////////////////////////////////////////////////////
/// Save primitive as a C++ statement(s) on output stream out

void TScatter::SavePrimitive(std::ostream &out, Option_t *option /*= ""*/)
{
   char quote = '"';
   out << "   " << std::endl;
   static Int_t frameNumber = 1000;
   frameNumber++;

   Int_t i;
   Double_t *X        = fGraph->GetX();
   Double_t *Y        = fGraph->GetY();
   TString fXName     = TString::Format("%s_fx%d",GetName(),frameNumber);
   TString fYName     = TString::Format("%s_fy%d", GetName(),frameNumber);
   TString fColorName = TString::Format("%s_fcolor%d",GetName(),frameNumber);
   TString fSizeName  = TString::Format("%s_fsize%d",GetName(),frameNumber);
   out << "   Double_t " << fXName << "[" << fNpoints << "] = {" << std::endl;
   for (i = 0; i < fNpoints-1; i++) out << "   " << X[i] << "," << std::endl;
   out << "   " << X[fNpoints-1] << "};" << std::endl;
   out << "   Double_t " << fYName << "[" << fNpoints << "] = {" << std::endl;
   for (i = 0; i < fNpoints-1; i++) out << "   " << Y[i] << "," << std::endl;
   out << "   " << Y[fNpoints-1] << "};" << std::endl;
   out << "   Double_t " << fColorName << "[" << fNpoints << "] = {" << std::endl;
   for (i = 0; i < fNpoints-1; i++) out << "   " << fColor[i] << "," << std::endl;
   out << "   " << fColor[fNpoints-1] << "};" << std::endl;
   out << "   Double_t " << fSizeName << "[" << fNpoints << "] = {" << std::endl;
   for (i = 0; i < fNpoints-1; i++) out << "   " << fSize[i] << "," << std::endl;
   out << "   " << fSize[fNpoints-1] << "};" << std::endl;

   if (gROOT->ClassSaved(TScatter::Class())) out << "   ";
   else out << "   TScatter *";
   out << "scat = new TScatter(" << fNpoints << ","
                                    << fXName   << ","  << fYName  << ","
                                    << fColorName  << ","  << fSizeName << ");"
                                    << std::endl;

   out << "   scat->SetName(" << quote << GetName() << quote << ");" << std::endl;
   out << "   scat->SetTitle(" << quote << GetTitle() << quote << ");" << std::endl;

   SaveFillAttributes(out, "scat", 0, 1001);
   SaveLineAttributes(out, "scat", 1, 1, 1);
   SaveMarkerAttributes(out, "scat", 1, 1, 1);

   if (fHistogram) {
      TString hname = fHistogram->GetName();
      hname += frameNumber;
      fHistogram->SetName(TString::Format("Graph_%s", hname.Data()));
      fHistogram->SavePrimitive(out, "nodraw");
      out << "   scat->SetHistogram(" << fHistogram->GetName() << ");" << std::endl;
      out << "   " << std::endl;
   }

   out << "   scat->Draw(" << quote << option << quote << ");" << std::endl;
}
