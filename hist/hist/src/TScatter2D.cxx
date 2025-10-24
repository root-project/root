// @(#)root/hist:$Id$
// Author: Olivier Couet   23/09/2025

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/


#include "TROOT.h"
#include "TScatter2D.h"
#include "TH1.h"
#include "TVirtualGraphPainter.h"

#include <iostream>


////////////////////////////////////////////////////////////////////////////////

/** \class TScatter2D
    \ingroup Graphs
A TScatter2D is able to draw five variables scatter plot on a single plot. The three first
variables are the x, y and z position of the markers (stored in a TGraph2D), the fourth is
mapped on the current color map and the fifth on the marker size.

The following example demonstrates how it works:

Begin_Macro(source)
../../../tutorials/visualisation/graphs/gr019_scatter2d.C
End_Macro

### TScatter2D's plotting options
TScatter2D can be drawn with the following options:

| Option   | Description                                                       |
|----------|-------------------------------------------------------------------|
| "SAME"   | Superimpose on previous picture in the same pad.|

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
   Bool_t status = TH1::AddDirectoryStatus();
   TH1::AddDirectory(kFALSE);
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

   TH1::AddDirectory(status);
}


////////////////////////////////////////////////////////////////////////////////
/// TScatter2D default destructor.

TScatter2D::~TScatter2D()
{
   delete fGraph;
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
   Int_t distance = 9999;
   if (fGraph) distance = fGraph->DistancetoPrimitive(px, py);
   return distance;
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
   if (fGraph) fGraph->ExecuteEvent(event, px, py);
}


////////////////////////////////////////////////////////////////////////////////
/// Returns a pointer to the histogram used to draw the axis

TH2D *TScatter2D::GetHistogram() const
{
   return fGraph->GetHistogram();;
}


////////////////////////////////////////////////////////////////////////////////
/// Get the scatter's x axis.

TAxis *TScatter2D::GetXaxis() const
{
   return fGraph->GetXaxis();
}


////////////////////////////////////////////////////////////////////////////////
/// Get the scatter's y axis.

TAxis *TScatter2D::GetYaxis() const
{
   return fGraph->GetYaxis();
}


////////////////////////////////////////////////////////////////////////////////
/// Get the scatter's z axis.

TAxis *TScatter2D::GetZaxis() const
{
   return fGraph->GetZaxis();
}


////////////////////////////////////////////////////////////////////////////////
/// Paint this scatter plot with its current attributes.

void TScatter2D::Paint(Option_t *option)
{
   TVirtualGraphPainter *painter = TVirtualGraphPainter::GetPainter();
   if (painter) painter->PaintScatter2D(this, option);
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

   out << "   scat->Draw(" << quote << option << quote << ");" << std::endl;
}
