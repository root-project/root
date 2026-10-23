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
#include "TPad.h"
#include "TView.h"
#include "TMath.h"
#include "TString.h"
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
| "LOGC"   | Log scale for colors.|
| "LOGS"   | Log scale for sizes.|
| "FB"     | Suppress the Front-Box.|
| "BB"     | Suppress the Back-Box.|
| "A"      | Suppress the axis.|
| "P"      | Suppress the palette.|

In the case of the SAME option, the log scale for color and size is inherited from the
previously drawn TScatter2D. For example, if a TScatter2D is drawn on top of another one
that uses a log scale for color, the second TScatter2D will also use a log scale for its
colors, even if the log scale for color is not explicitly specified for the second plot.

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
   TDirectory::TContext ctx{nullptr}; // No self-registration to directories
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

   // Check distance to 3D scatter markers
   if (fGraph && gPad) {
      TView *view = gPad->GetView();
      if (view) {
         const Double_t *fX = fGraph->GetX();
         const Double_t *fY = fGraph->GetY();
         const Double_t *fZ = fGraph->GetZ();
         for (Int_t i = 0; i < fNpoints; i++) {
            Double_t p3[3] = {fX[i], fY[i], fZ[i]};
            if (gPad->GetLogx()) {
               if (p3[0] > 0) p3[0] = TMath::Log10(p3[0]);
               else continue;
            }
            if (gPad->GetLogy()) {
               if (p3[1] > 0) p3[1] = TMath::Log10(p3[1]);
               else continue;
            }
            if (gPad->GetLogz()) {
               if (p3[2] > 0) p3[2] = TMath::Log10(p3[2]);
               else continue;
            }
            Double_t pndc[3];
            view->WCtoNDC(p3, pndc);
            Int_t dpx = px - gPad->XtoAbsPixel(gPad->XtoPad(pndc[0]));
            Int_t dpy = py - gPad->YtoAbsPixel(gPad->YtoPad(pndc[1]));
            Int_t d = TMath::Nint(TMath::Sqrt(dpx * dpx + dpy * dpy));
            if (d < distance) distance = d;
            if (distance <= 5) return distance;
         }
      }
   }

   if (fGraph) {
      Int_t gdist = fGraph->DistancetoPrimitive(px, py);
      if (gdist < distance) distance = gdist;
   }
   return distance;
}


////////////////////////////////////////////////////////////////////////////////
/// Get information on point of 2D/3D scatter plot at cursor position (px, py).
/// Shows x, y, point index, x, y, z values, color value, and size value.

char *TScatter2D::GetObjectInfo(Int_t px, Int_t py) const
{
   if (!gPad) {
      Error("GetObjectInfo", "Cannot be used without gPad");
      return nullptr;
   }

   if (!fGraph) return (char *)"";

   Int_t ipoint = -2;
   const Double_t *fX = fGraph->GetX();
   const Double_t *fY = fGraph->GetY();
   const Double_t *fZ = fGraph->GetZ();

   TView *view = gPad->GetView();
   if (view) {
      Double_t min_dist2 = 25; // 5 pixel radius
      for (Int_t i = 0; i < fNpoints; i++) {
         Double_t p3[3] = {fX[i], fY[i], fZ[i]};
         if (gPad->GetLogx()) {
            if (p3[0] > 0) p3[0] = TMath::Log10(p3[0]);
            else continue;
         }
         if (gPad->GetLogy()) {
            if (p3[1] > 0) p3[1] = TMath::Log10(p3[1]);
            else continue;
         }
         if (gPad->GetLogz()) {
            if (p3[2] > 0) p3[2] = TMath::Log10(p3[2]);
            else continue;
         }
         Double_t pndc[3];
         view->WCtoNDC(p3, pndc);
         Int_t dpx = px - gPad->XtoAbsPixel(gPad->XtoPad(pndc[0]));
         Int_t dpy = py - gPad->YtoAbsPixel(gPad->YtoPad(pndc[1]));

         Double_t dist2 = dpx * dpx + dpy * dpy;
         if (dist2 < min_dist2) {
            min_dist2 = dist2;
            ipoint = i;
         }
      }
   }

   Double_t x = gPad->PadtoX(gPad->AbsPixeltoX(px));
   Double_t y = gPad->PadtoY(gPad->AbsPixeltoY(py));

   static TString info;

   if (ipoint == -2) {
      info = TString::Format("x=%g, y=%g", x, y);
      return (char *) info.Data();
   }

   Double_t xval = fX[ipoint];
   Double_t yval = fY[ipoint];
   Double_t zval = fZ[ipoint];

   info = TString::Format("point=%d, xval=%g, yval=%g, zval=%g", ipoint, xval, yval, zval);
   if (fColor)
      info += TString::Format(", col=%g", fColor[ipoint]);
   if (fSize)
      info += TString::Format(", size=%g", fSize[ipoint]);

   return (char *) info.Data();
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
   if (fGraph) return fGraph->GetHistogram();
   else return nullptr;
}


////////////////////////////////////////////////////////////////////////////////
/// Get the scatter's x axis.

TAxis *TScatter2D::GetXaxis() const
{
   if (fGraph) return fGraph->GetXaxis();
   else return nullptr;
}


////////////////////////////////////////////////////////////////////////////////
/// Get the scatter's y axis.

TAxis *TScatter2D::GetYaxis() const
{
   if (fGraph) return fGraph->GetYaxis();
   else return nullptr;
}


////////////////////////////////////////////////////////////////////////////////
/// Get the scatter's z axis.

TAxis *TScatter2D::GetZaxis() const
{
   if (fGraph) return fGraph->GetZaxis();
   else return nullptr;
}


////////////////////////////////////////////////////////////////////////////////
/// Set minimum value for the z axis / vertical scale.

void TScatter2D::SetMinimum(Double_t min)
{
   if (fGraph) fGraph->SetMinimum(min);
}


////////////////////////////////////////////////////////////////////////////////
/// Set maximum value for the z axis / vertical scale.

void TScatter2D::SetMaximum(Double_t max)
{
   if (fGraph) fGraph->SetMaximum(max);
}


////////////////////////////////////////////////////////////////////////////////
/// Get minimum value for the z axis / vertical scale.

Double_t TScatter2D::GetMinimum() const
{
   if (fGraph) return fGraph->GetMinimum();
   return -1111;
}


////////////////////////////////////////////////////////////////////////////////
/// Get maximum value for the z axis / vertical scale.

Double_t TScatter2D::GetMaximum() const
{
   if (fGraph) return fGraph->GetMaximum();
   return -1111;
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
   if (!fGraph) return;

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
   TString arr_x    = SavePrimitiveVector(out, "scat_x", fNpoints, fGraph->GetX(), kTRUE);
   TString arr_y    = SavePrimitiveVector(out, "scat_y", fNpoints, fGraph->GetY());
   TString arr_z    = SavePrimitiveVector(out, "scat_z", fNpoints, fGraph->GetZ());
   TString arr_col  = SavePrimitiveVector(out, "scat_col", fNpoints, fColor);
   TString arr_size = SavePrimitiveVector(out, "scat_size", fNpoints, fSize);

   SavePrimitiveConstructor(out, Class(), "scat",
                            TString::Format("%d, %s.data(), %s.data(), %s.data(), %s.data(), %s.data()", fNpoints, arr_x.Data(),
                                            arr_y.Data(), arr_z.Data(), arr_col.Data(), arr_size.Data()),
                            kFALSE);

   SavePrimitiveNameTitle(out, "scat");
   SaveFillAttributes(out, "scat", -1, -1);
   SaveLineAttributes(out, "scat", 1, 1, 1);
   SaveMarkerAttributes(out, "scat", 1, 1, 1);

   out << "   scat->SetMargin(" << GetMargin() << ");\n";
   out << "   scat->SetMinMarkerSize(" << GetMinMarkerSize() << ");\n";
   out << "   scat->SetMaxMarkerSize(" << GetMaxMarkerSize() << ");\n";

   SavePrimitiveDraw(out, "scat", option);
}