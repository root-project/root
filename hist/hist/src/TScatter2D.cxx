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
#include "TScatter2D.h"
#include "TVirtualPad.h"
#include "TH3.h"
#include "TVirtualGraphPainter.h"

#include <iostream>
#include <cstring>

ClassImp(TScatter2D);


////////////////////////////////////////////////////////////////////////////////

/** \class TScatter2D
    \ingroup Graphs
A TScatter2D is able to draw give variables scatter plot on a single plot. The three first
variables are the x, y and z position of the markers, the fourth is mapped on the current
color map and the fifth on the marker size.

The following example demonstrates how it works:

Begin_Macro(source)
../../../tutorials/graphs/scatter2D.C
End_Macro

### TScatter2D's plotting options
TScatter2D can be drawn with the following options:

| Option   | Description                                                       |
|----------|-------------------------------------------------------------------|
| "A"      | Produce a new plot with Axis around the graph |

*/


////////////////////////////////////////////////////////////////////////////////
/// TScatter2D default constructor.

TScatter2D::TScatter2D()
{
}

////////////////////////////////////////////////////////////////////////////////
/// TScatter2D normal constructor.
///
///  the arrays are preset to zero

TScatter2D::TScatter2D(Int_t n)
{
   fGraph     = new TGraph2D(n);
   fNpoints   = fGraph->GetN();

   fColor = new Double_t[fNpoints];
   fSize  = new Double_t[fNpoints];

   memset(fColor, 0, fNpoints * sizeof(Double_t));
   memset(fSize, 0, fNpoints * sizeof(Double_t));
   fMaxMarkerSize = 5.;
   fMinMarkerSize = 1.;
   fMargin = 0.1;
}


////////////////////////////////////////////////////////////////////////////////
/// TScatter2D normal constructor.

TScatter2D::TScatter2D(Int_t n, Double_t *x, Double_t *y, Double_t *z, const Double_t *col, const Double_t *size)
{
   fGraph     = new TGraph2D(n, x, y, z);
   fNpoints   = fGraph->GetN();

   Int_t bufsize = sizeof(Double_t) * fNpoints;
   if (col) {
      fColor = new Double_t[fNpoints];
      memcpy(fColor, col, bufsize);
   }
   if (size) {
      fSize  = new Double_t[fNpoints];
      memcpy(fSize, size, bufsize);
   }

   fMaxMarkerSize = 5.;
   fMinMarkerSize = 1.;
   fMargin = 0.1;
}


////////////////////////////////////////////////////////////////////////////////
/// TScatter2D default destructor.

TScatter2D::~TScatter2D()
{
   delete fGraph;
   delete fHistogram;
   delete [] fColor;
   delete [] fSize;
}


////////////////////////////////////////////////////////////////////////////////
/// Compute distance from point px,py,pz to a scatter plot.
///
///  Compute the closest distance of approach from point px,py,pz to this scatter plot.
///  The distance is computed in pixels units.

Int_t TScatter2D::DistancetoPrimitive(Int_t px, Int_t py)
{
   // Are we on the axis?
   Int_t distance;
   if (this->GetHistogram()) {
      distance = this->GetHistogram()->DistancetoPrimitive(px,py);
      if (distance <= 5) return distance;
   }

   TVirtualGraphPainter *painter = TVirtualGraphPainter::GetPainter();
   /*if (painter)
      return painter->DistancetoPrimitiveHelper(this->GetGraph(), px, py);*/
   return 0;
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

void TScatter2D::ExecuteEvent(Int_t event, Int_t px, Int_t py)
{
   /*
   TVirtualGraphPainter *painter = TVirtualGraphPainter::GetPainter();
   if (painter) painter->ExecuteEventHelper(this->GetGraph(), event, px, py);
   */
}


////////////////////////////////////////////////////////////////////////////////
/// Returns a pointer to the histogram used to draw the axis

TH3F *TScatter2D::GetHistogram() const
{
   if (!fHistogram) {
      // do not add the histogram to gDirectory
      // use local TDirectory::TContext that will set temporarly gDirectory to a nullptr and
      // will avoid that histogram is added in the global directory
      TDirectory::TContext ctx(nullptr);
      double rwxmin, rwymin, rwzmin, rwxmax, rwymax, rwzmax;
      int npt = 25;
      fGraph->ComputeRange(rwxmin, rwymin, rwzmin, rwxmax, rwymax, rwzmax);
      double dx = (rwxmax-rwxmin)*fMargin;
      double dy = (rwymax-rwymin)*fMargin;
      double dz = (rwymax-rwymin)*fMargin;
      auto h = new TH3F(TString::Format("%s_h",GetName()),GetTitle(),npt,rwxmin-dx,rwxmax+dx,npt,rwymin-dy,rwymax+dy,npt,rwzmin-dz,rwzmax+dz);
      h->SetBit(TH1::kNoStats);
      h->SetDirectory(nullptr);
      h->Sumw2(kFALSE);
      const_cast<TScatter2D *>(this)->fHistogram = h;
   }
   return fHistogram;
}


////////////////////////////////////////////////////////////////////////////////
/// Get the scatter's x axis.

TAxis *TScatter2D::GetXaxis() const
{
   auto h = GetHistogram();
   return h ? h->GetXaxis() : nullptr;
}


////////////////////////////////////////////////////////////////////////////////
/// Get the scatter's y axis.

TAxis *TScatter2D::GetYaxis() const
{
   auto h = GetHistogram();
   return h ? h->GetYaxis() : nullptr;
}


////////////////////////////////////////////////////////////////////////////////
/// Get the scatter's z axis.

TAxis *TScatter2D::GetZaxis() const
{
   auto h = GetHistogram();
   return h ? h->GetZaxis() : nullptr;
}


////////////////////////////////////////////////////////////////////////////////
/// Paint this scatter plot with its current attributes.

void TScatter2D::Paint(Option_t *option)
{
   TVirtualGraphPainter *painter = TVirtualGraphPainter::GetPainter();
   if (painter) painter->PaintScatter(this, option);
}


////////////////////////////////////////////////////////////////////////////////
/// Print graph and errors values.

void TScatter2D::Print(Option_t *) const
{
   Double_t *X = fGraph->GetX();
   Double_t *Y = fGraph->GetY();
   Double_t *Z = fGraph->GetZ();
   for (Int_t i = 0; i < fNpoints; i++) {
      printf("x[%d]=%g, y[%d]=%g, z[%d]=%g", i, X[i], i, Y[i], i, Z[i]);
      if (fColor) printf(", color[%d]=%g", i, fColor[i]);
      if (fSize) printf(", size[%d]=%g", i, fSize[i]);
      printf("\n");
   }
}


////////////////////////////////////////////////////////////////////////////////
/// Set the margin around the plot in %

void TScatter2D::SetMargin(Double_t margin)
{
   if (fMargin != margin) {
      delete fHistogram;
      fHistogram = nullptr;
      fMargin = margin;
   }
}


////////////////////////////////////////////////////////////////////////////////
/// Save primitive as a C++ statement(s) on output stream out

void TScatter2D::SavePrimitive(std::ostream &out, Option_t *option /*= ""*/)
{
   char quote = '"';
   out << "   " << std::endl;
   static Int_t frameNumber = 1000;
   frameNumber++;

   Int_t i;
   Double_t *X        = fGraph->GetX();
   Double_t *Y        = fGraph->GetY();
   Double_t *Z        = fGraph->GetZ();
   TString fXName     = TString::Format("%s_fx%d",GetName(),frameNumber);
   TString fYName     = TString::Format("%s_fy%d", GetName(),frameNumber);
   TString fZName     = TString::Format("%s_fz%d", GetName(),frameNumber);
   TString fColorName = TString::Format("%s_fcolor%d",GetName(),frameNumber);
   TString fSizeName  = TString::Format("%s_fsize%d",GetName(),frameNumber);
   out << "   Double_t " << fXName << "[" << fNpoints << "] = {" << std::endl;
   for (i = 0; i < fNpoints-1; i++) out << "   " << X[i] << "," << std::endl;
   out << "   " << X[fNpoints-1] << "};" << std::endl;
   out << "   Double_t " << fYName << "[" << fNpoints << "] = {" << std::endl;
   for (i = 0; i < fNpoints-1; i++) out << "   " << Y[i] << "," << std::endl;
   out << "   " << Y[fNpoints-1] << "};" << std::endl;
   out << "   Double_t " << fZName << "[" << fNpoints << "] = {" << std::endl;
   for (i = 0; i < fNpoints-1; i++) out << "   " << Z[i] << "," << std::endl;
   out << "   " << Z[fNpoints-1] << "};" << std::endl;
   out << "   Double_t " << fColorName << "[" << fNpoints << "] = {" << std::endl;
   for (i = 0; i < fNpoints-1; i++) out << "   " << fColor[i] << "," << std::endl;
   out << "   " << fColor[fNpoints-1] << "};" << std::endl;
   out << "   Double_t " << fSizeName << "[" << fNpoints << "] = {" << std::endl;
   for (i = 0; i < fNpoints-1; i++) out << "   " << fSize[i] << "," << std::endl;
   out << "   " << fSize[fNpoints-1] << "};" << std::endl;

   if (gROOT->ClassSaved(TScatter2D::Class()))
      out << "   ";
   else
      out << "   TScatter2D *";
   out << "scat = new TScatter2D(" << fNpoints << "," << fXName << ","  << fYName  << "," << fZName << ","
                                 << fColorName  << ","  << fSizeName << ");" << std::endl;

   out << "   scat->SetName(" << quote << GetName() << quote << ");" << std::endl;
   out << "   scat->SetTitle(" << quote << GetTitle() << quote << ");" << std::endl;
   out << "   scat->SetMargin(" << GetMargin() << ");" << std::endl;
   out << "   scat->SetMinMarkerSize(" << GetMinMarkerSize() << ");" << std::endl;
   out << "   scat->SetMaxMarkerSize(" << GetMaxMarkerSize() << ");" << std::endl;

   SaveFillAttributes(out, "scat", 0, 1001);
   SaveLineAttributes(out, "scat", 1, 1, 1);
   SaveMarkerAttributes(out, "scat", 1, 1, 1);

   if (fHistogram) {
      TString hname = fHistogram->GetName();
      fHistogram->SetName(TString::Format("Graph_%s%d", hname.Data(), frameNumber));
      fHistogram->SavePrimitive(out, "nodraw");
      out << "   scat->SetHistogram(" << fHistogram->GetName() << ");" << std::endl;
      out << "   " << std::endl;
      fHistogram->SetName(hname);
   }

   out << "   scat->Draw(" << quote << option << quote << ");" << std::endl;
}
