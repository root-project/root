// @(#)root/graf:$Id$
// Author: Guido Volpi, Olivier Couet 03/11/2006
// Author: Sergey Linev 09/2026

/*************************************************************************
 * Copyright (C) 1995-2026, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TPie.h"
#include "TPieSlice.h"

#include "TROOT.h"
#include "TVirtualPad.h"
#include "TCanvasImp.h"
#include "TArc.h"
#include "TLegend.h"
#include "TMath.h"
#include "TStyle.h"
#include "TLatex.h"
#include "TPaveText.h"
#include "TH1.h"
#include "TColor.h"
#include "TLine.h"

#include <iostream>


/** \class TPie
\ingroup BasicGraphics

Define a Pie Chart

\anchor PIE00
#### The possible drawing options are:

| Option | Description                                                       |
|--------|-------------------------------------------------------------------|
| "R"    | Print the labels along the central "R"adius of slices. |
| "T"    | Print the label in a direction "T"angent to circle that describes the TPie. |
| "SC"   | Paint the labels with the "S"ame "C"olor as the slices. |
| "3D"   | Draw the pie-chart with a pseudo 3D effect. |
| "NOL"  | No OutLine: Don't draw the slices' outlines, any property over the slices' line is ignored. |
| ">"    | Sort the slices in increasing order. |
| "<"    | Sort the slices in decreasing order. |

After the use of > or < options the internal order of the TPieSlices is changed.

\anchor PIE01
#### Labels format
The labels' format can be changed with TPie::SetLabelFormat().

The format string must contain one of these modifiers:

| Modifier | Description                                                       |
|----------|-------------------------------------------------------------------|
| "%txt"   | to print the text label associated with the slice |
| "%val"   | to print the numeric value of the slice |
| "%frac"  | to print the relative fraction of this slice |
| "%perc"  | to print the % of this slice |

Example:
```
mypie->SetLabelFormat("%txt (%frac)");
```
The numeric format in the label can be changed with TPie::SetFractionFormat.
The numeric format the slices' values can be changed with TPie::SetValueFormat.
The numeric format for the percent value of a slice can be changed with TPie::SetPercentFormat.

If the colors are not specified (`nullptr`) they are automatically
taken from the current color palette as shown in the following example.

Begin_Macro(source)
{
   float vals[] = {10., 2., 5., 7.5};
   const char* labels[]={"Label 1","Label 2","Label 3","Label4"};

   auto C = new TCanvas("C","TPie test",500,500);
   auto pie  = new TPie("pie", "Pie Title", 4, vals, nullptr, labels);
   pie->SetTextFont(42);
   pie->SetTextSize(0.04);
   pie->SetLabelFormat("#splitline{%txt}{%perc}");
   pie->SetLabelsOffset(.02);
   pie->SetRadius(.3);
   pie->Draw("");
}
End_Macro

More complex example:

Begin_Macro(source)
../../../tutorials/visualisation/graphics/piechart.C
End_Macro

*/

////////////////////////////////////////////////////////////////////////////////
/// Default constructor.

TPie::TPie() : TNamed()
{
   Init(1, 0, 0.5, 0.5, 0.4);
}

////////////////////////////////////////////////////////////////////////////////
/// This constructor creates a pie chart when only the number of
/// the slices is known. The number of slices is fixed.

TPie::TPie(const char *name, const char *title, Int_t npoints) :
           TNamed(name,title)
{
   Init(npoints, 0, 0.5, 0.5, 0.4);
}

////////////////////////////////////////////////////////////////////////////////
/// Normal constructor. The 1st and 2nd parameters are the name of the object
/// and its title.
///
/// The number of points passed at this point is used to allocate the memory.
///
/// Slices values are given as Double_t.
///
/// The 4th elements is an array containing, in double precision format,
/// the value of each slice. It is also possible to specify the filled color
/// of each slice. If the color array is not specified the slices are colored
/// using a color sequence in the standard palette.

TPie::TPie(const char *name, const char *title,
           Int_t npoints, Double_t *vals,
           Int_t *colors, const char *lbls[]) : TNamed(name,title)
{
   Init(npoints, 0, 0.5, 0.5, 0.4);
   for (Int_t i=0; i<fNvals; ++i) fPieSlices[i]->SetValue(vals[i]);

   SetFillColors(colors);
   SetLabels(lbls);
}

////////////////////////////////////////////////////////////////////////////////
/// Normal constructor (Float_t).

TPie::TPie(const char *name,
           const char *title,
           Int_t npoints, Float_t *vals,
           Int_t *colors, const char *lbls[]) : TNamed(name,title)
{
   Init(npoints, 0, 0.5, 0.5, 0.4);
   for (Int_t i=0; i<fNvals; ++i) fPieSlices[i]->SetValue(vals[i]);

   SetFillColors(colors);
   SetLabels(lbls);
}

////////////////////////////////////////////////////////////////////////////////
/// Constructor from a TH1

TPie::TPie(const TH1 *h) : TNamed(h->GetName(),h->GetTitle())
{
   Int_t i;

   const TAxis *axis = h->GetXaxis();
   Int_t first = axis->GetFirst();
   Int_t last  = axis->GetLast();
   Int_t np    = last-first+1;
   Init(np, 0, 0.5, 0.5, 0.4);

   for (i=first; i<=last; ++i) fPieSlices[i-first]->SetValue(h->GetBinContent(i));
   if (axis->GetLabels()) {
      for (i=first; i<=last; ++i) fPieSlices[i-first]->SetTitle(axis->GetBinLabel(i));
   } else {
      SetLabelFormat("%val");
   }
   SetTextSize(axis->GetLabelSize());
   SetTextColor(axis->GetLabelColor());
   SetTextFont(axis->GetLabelFont());
}

////////////////////////////////////////////////////////////////////////////////
/// Copy constructor.

TPie::TPie(const TPie &cpy) : TNamed(cpy), TAttText(cpy)
{
   Init(cpy.fNvals, cpy.fAngularOffset, cpy.fX, cpy.fY, cpy.fRadius);

   for (Int_t i=0;i<fNvals;++i)
      cpy.fPieSlices[i]->Copy(*fPieSlices[i]);
}

////////////////////////////////////////////////////////////////////////////////
/// Destructor.

TPie::~TPie()
{
   if (fNvals>0) {
      for (int i=0; i<fNvals; ++i) delete fPieSlices[i];
      delete [] fPieSlices;
   }

   if (fLegend) delete fLegend;
}

////////////////////////////////////////////////////////////////////////////////
/// Evaluate the distance to the chart in gPad.

Int_t TPie::DistancetoPrimitive(Int_t px, Int_t py)
{
   Int_t dist = 9999;
   if (gPad) {
      auto info = FindSlice(*gPad, px, py);
      if ((info.num >= 0) && (info.rad <= fRadius))
         dist = 0;
   }

   return dist;
}

////////////////////////////////////////////////////////////////////////////////
/// Find slice at given position

TPie::SliceInfo_t TPie::FindSlice(TVirtualPad &parent, Int_t px, Int_t py)
{
   SliceInfo_t res;
   res.num = -1;

   // coordinates
   Double_t xx = parent.AbsPixeltoX(px);
   Double_t yy = parent.AbsPixeltoY(py);

   // XY metric
   Double_t radX = GetRadius();
   Double_t radY = GetRadius();
   Double_t radXY = 1.;
   if (fIs3D) {
      radXY = TMath::Sin(fAngle3D / 180. * TMath::Pi());
      radY = radXY * radX;
   }

   auto angles = GetSlicesAngles();

   Float_t dPxl = (parent.PixeltoY(0) - parent.PixeltoY(1)) / radY;
   for (Int_t i = 0; i < fNvals; ++i) {

      // Angles' values for this slice
      Double_t phimin = angles[2 * i] * TMath::Pi() / 180.;
      Double_t cphi = angles[2 * i + 1] * TMath::Pi() / 180.;
      Double_t phimax = angles[2 * i + 2] * TMath::Pi() / 180.;

      Double_t radOffset = fPieSlices[i]->GetRadiusOffset();

      Double_t dx = (xx - fX - radOffset * TMath::Cos(cphi)) / radX;
      Double_t dy = (yy - fY - radOffset * TMath::Sin(cphi) * radXY) / radY;

      if (TMath::Abs(dy) < dPxl)
         dy = dPxl;

      Double_t ang = TMath::ATan2(dy, dx);
      if (ang < 0)
         ang += TMath::TwoPi();

      Double_t dist = TMath::Sqrt(dx * dx + dy * dy);

      if (((ang >= phimin && ang <= phimax) ||
           (phimax > TMath::TwoPi() && ang + TMath::TwoPi() >= phimin && ang + TMath::TwoPi() < phimax)) &&
          dist <= 1.) { // if true the pointer is in the slice region

         res.x = dx;
         res.y = dy;
         res.ang = ang;
         res.phi1 = phimin;
         res.phi2 = phimax;
         res.rad = dist * fRadius;

         if (dist < .95 && dist > .65) {
            Double_t range = phimax - phimin;
            Double_t lang = ang - phimin;
            Double_t rang = phimax - ang;
            if (lang < 0)
               lang += TMath::TwoPi();
            else if (lang >= TMath::TwoPi())
               lang -= TMath::TwoPi();
            if (rang < 0)
               rang += TMath::TwoPi();
            else if (rang >= TMath::TwoPi())
               rang -= TMath::TwoPi();

            if (lang / range < .25 || rang / range < .25) {
               res.num = -1;
            } else
               res.num = i;
         } else {
            res.num = i;
         }

         break;
      }
   }
   return res;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns the slice number at the pixel position (px,py).
/// Returns -1 if no slice is picked.

Int_t TPie::DistancetoSlice(Int_t px, Int_t py)
{
   if (!gPad)
      return -1;
   auto res = FindSlice(*gPad, px, py);
   return res.num;
}


////////////////////////////////////////////////////////////////////////////////
/// Draw the pie chart.
///
/// The possible options are listed in the TPie::Paint() method.

void TPie::Draw(Option_t *option)
{
   TString soption(option);
   soption.ToLower();

   if (soption.Length()==0) soption = "l";

   if (gPad) {
      if (!gPad->IsEditable()) gROOT->MakeDefCanvas();
      if (!soption.Contains("same")) {
         gPad->Clear();
         gPad->Range(0.,0.,1.,1.);
      }
   }

   for (Int_t i=0;i<fNvals;++i) fPieSlices[i]->AppendPad();
   AppendPad(soption.Data());
}

class TPieInteractive : public TVirtualPad::TInteractive {
   public:

   enum EMode { pNone = 0, pMovingPie, pMovingSlice, pResizing, pRotating } fMode = pNone;

   Int_t oldpx = 0, oldpy = 0;

   Double_t angularOffset0  = 0; // Reference angular offset

   Int_t    currentSlice = -1; // Current slice under mouse.
   Double_t sliceAngle = 0.;  // angle for moving slice
   Double_t sliceOffset = 0; // offset for moving slice
   Double_t angle0 = 0;  // previous angle to mouse when rotating
};



////////////////////////////////////////////////////////////////////////////////
/// This method is for internal use. It is used by Execute event to draw the
/// outline of "this" TPie. Used when the opaque movements are not permitted.

void TPie::DrawGhost(TVirtualPad &parent)
{
   TAttLine(kBlack, 1, 2).ModifyOn(parent);

   std::vector<Double_t> xx, yy;

   // XY metric
   Double_t radXY = !fIs3D ? 1. : TMath::Sin(fAngle3D/180.*TMath::Pi()), x0 = 0, y0 = 0;

   auto addPoint = [&parent, &xx, &yy, &x0, &y0](double x1, double y1) {
      xx.emplace_back(parent.XtoPad(x0 + x1));
      yy.emplace_back(parent.YtoPad(y0 + y1));
   };

   auto angles = GetSlicesAngles();

   for (Int_t loop3d = fIs3D ? 1 : 0; loop3d >= 0; loop3d--)
      for (Int_t i = 0; i < fNvals; ++i) {
         xx.clear();
         yy.clear();
         Float_t minphi = angles[i * 2] * TMath::Pi() / 180.;
         Float_t avgphi = angles[i * 2 + 1] * TMath::Pi() / 180.;
         Float_t maxphi = angles[i * 2 + 2] * TMath::Pi() / 180.;

         Double_t radOffset = fPieSlices[i]->GetRadiusOffset();
         x0 = fX + radOffset * TMath::Cos(avgphi);
         y0 = fY + radOffset * TMath::Sin(avgphi) * radXY - loop3d * fHeight; // draw layer beyond

         addPoint(0, 0);

         Double_t dphi = (maxphi - minphi);
         Int_t ndiv = dphi > 1.5 ? (Int_t)(dphi / .15) : 10;
         dphi /= ndiv;

         // Loop to draw the arc
         for (Int_t j = 0; j <= ndiv; ++j) {
            Double_t phi = minphi + dphi * j;
            addPoint(fRadius * TMath::Cos(phi), fRadius * TMath::Sin(phi) * radXY);
         }

         addPoint(0, 0);

         if (loop3d)
            addPoint(0, fHeight);

         parent.PaintPolyLine(xx.size(), xx.data(), yy.data(),
                              TString::Format("ipiechart_slice%d_loop%d", i, loop3d).Data());

         if (loop3d) {
            xx.clear();
            yy.clear();
            addPoint(fRadius * TMath::Cos(minphi), fRadius * TMath::Sin(minphi) * radXY);
            addPoint(fRadius * TMath::Cos(minphi), fRadius * TMath::Sin(minphi) * radXY + fHeight);
            parent.PaintPolyLine(xx.size(), xx.data(), yy.data(),
                                 TString::Format("ipiechart_slice%d_bindmin", i).Data());
            xx.clear();
            yy.clear();
            addPoint(fRadius * TMath::Cos(maxphi), fRadius * TMath::Sin(maxphi) * radXY);
            addPoint(fRadius * TMath::Cos(maxphi), fRadius * TMath::Sin(maxphi) * radXY + fHeight);
            parent.PaintPolyLine(xx.size(), xx.data(), yy.data(),
                                 TString::Format("ipiechart_slice%d_bindmax", i).Data());
         }
      }

   parent.UpdateAsync();
}

////////////////////////////////////////////////////////////////////////////////
/// Execute the mouse events.

void TPie::ExecuteEvent(Int_t event, Int_t px, Int_t py)
{
   if (!gPad || !gPad->IsEditable()) return;

   auto &parent = *gPad;

   Bool_t opaque = parent.OpaqueMoving();

   // Portion of pie considered as "border"
   const Double_t dr     = parent.PixeltoX(3);
   const Double_t minRad = parent.PixeltoX(10);

   // Angular divisions in radial direction
   const Double_t angstep1 = 0.5*TMath::PiOver4();
   const Double_t angstep2 = 1.5*TMath::PiOver4();
   const Double_t angstep3 = 2.5*TMath::PiOver4();
   const Double_t angstep4 = 3.5*TMath::PiOver4();
   const Double_t angstep5 = 4.5*TMath::PiOver4();
   const Double_t angstep6 = 5.5*TMath::PiOver4();
   const Double_t angstep7 = 6.5*TMath::PiOver4();
   const Double_t angstep8 = 7.5*TMath::PiOver4();

   // XY metric
   Double_t radXY = fIs3D ? TMath::Sin(fAngle3D/180.*TMath::Pi()) : 1.;

   auto inter = dynamic_cast<TPieInteractive *>(parent.Interactive(this));

   auto calcAngle = [this, &parent,px,py]() {
      Double_t xx = parent.AbsPixeltoX(px);
      Double_t yy = parent.AbsPixeltoY(py);

      Double_t dx1  = xx - GetX();
      Double_t dy1  = yy - GetY();

      return TMath::ATan2(dy1, dx1) / TMath::Pi() * 180.;
   };

   auto info = FindSlice(parent, px, py);
   if (info.num < 0)
      info.rad = fRadius * 100; // far away to exclude any slection

   switch(event) {
      case kArrowKeyPress:
      case kButton1Down:
         inter = new TPieInteractive();
         parent.Interactive(this, inter);

         // Current center and radius.
         inter->angularOffset0 = fAngularOffset;
         // no break

      case kMouseMotion: {
         TPieInteractive dummy;
         if (!inter) inter = &dummy; //

         if ((info.rad >= fRadius - 2. * dr) && (info.rad <= fRadius + dr)) {
            if (info.ang >= angstep8 || info.ang < angstep1)
               parent.SetCursor(kRightSide);
            else if (info.ang >= angstep1 && info.ang < angstep2)
               parent.SetCursor(kTopRight);
            else if (info.ang >= angstep2 && info.ang < angstep3)
               parent.SetCursor(kTopSide);
            else if (info.ang >= angstep3 && info.ang < angstep4)
               parent.SetCursor(kTopLeft);
            else if (info.ang >= angstep4 && info.ang <= angstep5)
               parent.SetCursor(kLeftSide);
            else if (info.ang >= angstep5 && info.ang < angstep6)
               parent.SetCursor(kBottomLeft);
            else if (info.ang >= angstep6 && info.ang < angstep7)
               parent.SetCursor(kBottomSide);
            else if (info.ang >= angstep7 && info.ang < angstep8)
               parent.SetCursor(kBottomRight);
            inter->fMode = TPieInteractive::pResizing;
         } else if ((info.rad > fRadius * 0.6) && (info.rad < fRadius)) {
            if (info.num >= 0) {
               parent.SetCursor(kPointer);
               auto angles = GetSlicesAngles();
               inter->currentSlice = info.num;
               inter->sliceAngle = angles[info.num*2+1] * TMath::Pi()/180.;
               inter->sliceOffset = fPieSlices[inter->currentSlice]->GetRadiusOffset();
               inter->fMode = TPieInteractive::pMovingSlice;
            } else
               parent.Interactive(); // reject interactive
         } else if ((info.rad >= fRadius * 0.3) && (info.rad <= fRadius * .6)) {
            parent.SetCursor(kRotate);
            inter->fMode = TPieInteractive::pRotating;
            inter->angle0 = calcAngle();
         } else if (info.rad <= fRadius * 0.3) {
            parent.SetCursor(kHand);
            inter->fMode = TPieInteractive::pMovingPie;
         }
         inter->oldpx = px;
         inter->oldpy = py;
         break;
      }

      case kArrowKeyRelease:
      case kButton1Motion: {
         if (!inter)
            break;

         Int_t dx = px - inter->oldpx;
         Int_t dy = py - inter->oldpy;

         Double_t mdx = parent.PixeltoX(dx);
         Double_t mdy = parent.PixeltoY(dy);

         if (inter->fMode == TPieInteractive::pMovingPie) {
            parent.SetCursor(kMove);
            fX += mdx;
            fY += mdy;
         } else if (inter->fMode == TPieInteractive::pMovingSlice) {
            parent.SetCursor(kMove);
            inter->sliceOffset += TMath::Cos(inter->sliceAngle)*mdx + TMath::Sin(inter->sliceAngle)*mdy/radXY;
            fPieSlices[inter->currentSlice]->SetRadiusOffset(TMath::Max(0., inter->sliceOffset));
         } else if (inter->fMode == TPieInteractive::pResizing) {
            Float_t dr1 = mdx*TMath::Cos(info.ang) + mdy*TMath::Sin(info.ang)/radXY;
            fRadius = TMath::Max(fRadius + dr1, minRad);
         } else if (inter->fMode == TPieInteractive::pRotating) {
            Double_t angle1 = calcAngle();
            SetAngularOffset(GetAngularOffset() + angle1 - inter->angle0);
            inter->angle0 = angle1;
         }

         if (!opaque)
            DrawGhost(parent);
         else
            parent.Modified();

         parent.UpdateAsync();

         inter->oldpx = px;
         inter->oldpy = py;

         break;
      }

      case kButton1Up:
         if (gROOT->IsEscaped()) {
            gROOT->SetEscape(kFALSE);
            fAngularOffset = inter->angularOffset0;
            parent.Modified();
         } else if (!opaque) {
            parent.Modified();
         }

         parent.Interactive(); // remove interactive object
         parent.UpdateAsync();

         break;

      case kButton1Locate:

         ExecuteEvent(kButton1Down, px, py);

         while (true) {
            px = py = 0;
            event = parent.GetCanvasImp()->RequestLocator(px, py);

            ExecuteEvent(kButton1Motion, px, py);

            if (event != -1) {                     // button is released
               ExecuteEvent(kButton1Up, px, py);
               return;
            }
         }
         break;

      case kMouseEnter:
         break;

      default:
         // unknown event
         break;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Returns the label of the entry number "i".

const char* TPie::GetEntryLabel(Int_t i) const
{
   return GetSlice(i)->GetTitle();
}

////////////////////////////////////////////////////////////////////////////////
/// Return the color of the slice number "i".

Int_t TPie::GetEntryFillColor(Int_t i) const
{
   return GetSlice(i)->GetFillColor();
}

////////////////////////////////////////////////////////////////////////////////
/// Return the style use to fill the slice number "i".

Int_t TPie::GetEntryFillStyle(Int_t i) const
{
   return GetSlice(i)->GetFillStyle();
}

////////////////////////////////////////////////////////////////////////////////
/// Return the line color used to outline thi "i" slice

Int_t TPie::GetEntryLineColor(Int_t i) const
{
   return GetSlice(i)->GetLineColor();
}

////////////////////////////////////////////////////////////////////////////////
/// Return the style used to outline thi "i" slice

Int_t TPie::GetEntryLineStyle(Int_t i) const
{
   return GetSlice(i)->GetLineStyle();
}

////////////////////////////////////////////////////////////////////////////////
/// Return the line width used to outline thi "i" slice

Int_t TPie::GetEntryLineWidth(Int_t i) const
{
   return GetSlice(i)->GetLineWidth();
}

////////////////////////////////////////////////////////////////////////////////
/// Return the radial offset's value for the slice number "i".

Double_t TPie::GetEntryRadiusOffset(Int_t i) const
{
   return GetSlice(i)->GetRadiusOffset();
}

////////////////////////////////////////////////////////////////////////////////
/// Return the value associated with the slice number "i".

Double_t TPie::GetEntryVal(Int_t i) const
{
   return GetSlice(i)->GetValue();
}

////////////////////////////////////////////////////////////////////////////////
/// Return sum of all entries.
/// Absolute value for each entry will be used

Double_t TPie::GetSumOfEntriesValues() const
{
   Double_t sum = 0;
   for (Int_t i = 0; i < fNvals; ++i)
      sum += TMath::Abs(fPieSlices[i]->GetValue());
   return sum;
}


////////////////////////////////////////////////////////////////////////////////
/// If created before by Paint option or by MakeLegend method return
/// the pointer to the legend, otherwise return 0;

TLegend* TPie::GetLegend() const
{
   return fLegend;
}

////////////////////////////////////////////////////////////////////////////////
/// Return the reference to the slice of index 'id'.

TPieSlice* TPie::GetSlice(Int_t id) const
{
   return (id >= 0) && (id < fNvals) ? fPieSlices[id] : nullptr;
}

////////////////////////////////////////////////////////////////////////////////
/// Common initialization for all constructors.
/// This is a private function called to allocate the memory.

void TPie::Init(Int_t np, Double_t ao, Double_t x, Double_t y, Double_t r)
{
   fAngularOffset = ao;
   fX             = x;
   fY             = y;
   fRadius        = r;
   fNvals         = np;
   fLegend        = nullptr;
   fHeight        = 0.08;
   fAngle3D       = 30;
   fIs3D          = kFALSE;

   fLabelsOffset = gStyle->GetLabelOffset();

   fPieSlices = new TPieSlice*[fNvals];

   Int_t ic = 0, dc = TMath::Max(1, gStyle->GetNumberOfColors()/fNvals);

   for (Int_t i=0;i<fNvals;++i) {
      TString tmplbl = "Slice";
      tmplbl += i;
      fPieSlices[i] = new TPieSlice(tmplbl.Data(), tmplbl.Data(), this);
      fPieSlices[i]->SetRadiusOffset(0.);
      fPieSlices[i]->SetLineColor(1);
      fPieSlices[i]->SetLineStyle(1);
      fPieSlices[i]->SetLineWidth(1);
      fPieSlices[i]->SetFillColor(gStyle->GetColorPalette(ic));
      ic = TMath::Min(gStyle->GetNumberOfColors(), ic+dc);
      fPieSlices[i]->SetFillStyle(1001);
   }

   fLabelFormat    = "%txt";
   fFractionFormat = "%3.2f";
   fValueFormat    = "%4.2f";
   fPercentFormat  = "%3.1f";
}

////////////////////////////////////////////////////////////////////////////////
/// This method create a legend that explains the contents
/// of the slice for this pie-chart.
///
/// The parameter passed reppresents the option passed to shown the slices,
/// see TLegend::AddEntry() for further details.
///
/// The pointer of the TLegend is returned.

TLegend* TPie::MakeLegend(Double_t x1, Double_t y1, Double_t x2, Double_t y2, const char *leg_header)
{
   if (!fLegend) fLegend = new TLegend(x1,y1,x2,y2,leg_header);
   else fLegend->Clear();

   for (Int_t i=0;i<fNvals;++i) {
      fLegend->AddEntry(*(fPieSlices+i),fPieSlices[i]->GetTitle(),"f");
   }

   if (gPad) fLegend->Draw();

   return fLegend;
}

////////////////////////////////////////////////////////////////////////////////
/// Paint a Pie chart in a canvas.
/// [See the class description for the possible drawing options](\ref PIE00)

void TPie::Paint(Option_t *option)
{
   TString soption(option);

   Bool_t optionSame = kFALSE;

   // if true the lines around the slices are drawn, if false not
   Bool_t optionLine = kTRUE;

   // if true the labels' colors are the same as the slices' colors
   Bool_t optionSameColor = kFALSE;

   // For the label orientation there are 3 possibilities:
   //   0: horizontal
   //   1: radial
   //   2: tangent
   Int_t lblor = 0;

   // Parse the options
   Int_t idx;
   // Paint the TPie in an existing canvas
   if ( (idx=soption.Index("same"))>=0 ) {
      optionSame = kTRUE;
      soption.Remove(idx,4);
   }

   if ( (idx=soption.Index("nol"))>=0 ) {
      optionLine = kFALSE;
      soption.Remove(idx,3);
   }

   if ( (idx=soption.Index("sc"))>=0 ) {
      optionSameColor = kTRUE;
      soption.Remove(idx,2);
   }

   // check if is active the pseudo-3d
   if ( (idx=soption.Index("3d"))>=0 ) {
      fIs3D = kTRUE;
      soption.Remove(idx,2);
   } else {
      fIs3D = kFALSE;
   }

   // seek if have to draw the labels around the pie chart
   if ( (idx=soption.Index("t"))>=0 ) {
      lblor = 2;
      soption.Remove(idx,1);
   }

   // Seek if have to paint the labels along the radii
   if ( (idx=soption.Index("r"))>=0 ) {
      lblor = 1;
      soption.Remove(idx,1);
   }

   // Seeks if has to paint sort the slices in increasing mode
   if ( (idx=soption.Index(">"))>=0 ) {
      SortSlices(kTRUE);
      soption.Remove(idx,1);
   }

   // Seeks if has to paint sort the slices in decreasing mode
   if ( (idx=soption.Index("<"))>=0 ) {
      SortSlices(kFALSE);
      soption.Remove(idx,1);
   }

   if (fNvals<=0) {
      Warning("Paint","No vals");
      return;
   }

   if (!fPieSlices) {
      Warning("Paint","No valid arrays of values");
      return;
   }

   // Check if gPad exists and define the drawing range.
   if (!gPad) return;

   // Objects useful to draw labels and slices
   TLatex textlabel;
   TArc arc;
   TLine line;

   // XY metric
   Double_t radX  = fRadius;
   Double_t radY  = fRadius;
   Double_t radXY = 1.;

   if (fIs3D) {
      radXY = TMath::Sin(fAngle3D/180.*TMath::Pi());
      radY = fRadius*radXY;
   }

   auto angles = GetSlicesAngles();

   // Draw the slices.
   Int_t pixelHeight = gPad->YtoPixel(0)-gPad->YtoPixel(fHeight);
   for (Int_t pi = 0; pi < pixelHeight && fIs3D; ++pi) { // loop for pseudo-3d effect
      for (Int_t i=0;i<fNvals;++i) {
         // draw the arc
         // set the color of the next slice
         if (pi>0) {
            arc.SetFillStyle(0);
            arc.SetLineColor(TColor::GetColorDark((fPieSlices[i]->GetFillColor())));
         } else {
            arc.SetFillStyle(0);
            if (optionLine) {
               arc.SetLineColor(fPieSlices[i]->GetLineColor());
               arc.SetLineStyle(fPieSlices[i]->GetLineStyle());
               arc.SetLineWidth(fPieSlices[i]->GetLineWidth());
            } else {
               arc.SetLineWidth(1);
               arc.SetLineColor(TColor::GetColorDark((fPieSlices[i]->GetFillColor())));
            }
         }
         // Paint the slice
         Float_t aphi = angles[2*i+1]*TMath::Pi()/180.;

         Double_t ax = fX+TMath::Cos(aphi)*fPieSlices[i]->GetRadiusOffset();
         Double_t ay = fY+TMath::Sin(aphi)*fPieSlices[i]->GetRadiusOffset()*radXY+gPad->PixeltoY(pixelHeight-pi);

         arc.PaintEllipse(ax, ay, radX, radY, angles[2 * i], angles[2 * i + 2], 0.);

         if (optionLine) {
            line.SetLineColor(fPieSlices[i]->GetLineColor());
            line.SetLineStyle(fPieSlices[i]->GetLineStyle());
            line.SetLineWidth(fPieSlices[i]->GetLineWidth());
            line.PaintLine(ax,ay,ax,ay);

            Double_t x0, y0;
            x0 = ax+radX*TMath::Cos(angles[2*i]/180.*TMath::Pi());
            y0 = ay+radY*TMath::Sin(angles[2*i]/180.*TMath::Pi());
            line.PaintLine(x0,y0,x0,y0);

            x0 = ax+radX*TMath::Cos(angles[2*i+2]/180.*TMath::Pi());
            y0 = ay+radY*TMath::Sin(angles[2*i+2]/180.*TMath::Pi());
            line.PaintLine(x0,y0,x0,y0);
         }
      }
   } // end loop for pseudo-3d effect

   for (Int_t i=0;i<fNvals;++i) { // loop for the piechart
      // Set the color of the next slice
      arc.SetFillColor(fPieSlices[i]->GetFillColor());
      arc.SetFillStyle(fPieSlices[i]->GetFillStyle());
      if (optionLine) {
         arc.SetLineColor(fPieSlices[i]->GetLineColor());
         arc.SetLineStyle(fPieSlices[i]->GetLineStyle());
         arc.SetLineWidth(fPieSlices[i]->GetLineWidth());
      } else {
         arc.SetLineWidth(1);
         arc.SetLineColor(fPieSlices[i]->GetFillColor());
      }

      // Paint the slice
      Float_t aphi = angles[2*i+1]*TMath::Pi()/180.;

      Double_t ax = fX+TMath::Cos(aphi)*fPieSlices[i]->GetRadiusOffset();
      Double_t ay = fY+TMath::Sin(aphi)*fPieSlices[i]->GetRadiusOffset()*radXY;
      arc.PaintEllipse(ax, ay, radX, radY, angles[2*i], angles[2*i+2], 0.);

   } // end loop to draw the slices


   // Set the font
   textlabel.SetTextFont(GetTextFont());
   textlabel.SetTextSize(GetTextSize());
   textlabel.SetTextColor(GetTextColor());

   Double_t sum = GetSumOfEntriesValues();
   if (sum <= 0) sum = 1.; // just to avoid zero division

   // Loop to place the labels.
   for (Int_t i=0;i<fNvals;++i) {
      Float_t aphi = angles[2*i+1]*TMath::Pi()/180.;
      //aphi = TMath::ATan2(TMath::Sin(aphi)*radXY,TMath::Cos(aphi));

      Float_t label_off = fLabelsOffset;

      // Paint the text in the pad
      TString tmptxt  = fLabelFormat;

      tmptxt.ReplaceAll("%txt",fPieSlices[i]->GetTitle());
      tmptxt.ReplaceAll("%val",TString::Format(fValueFormat.Data(),fPieSlices[i]->GetValue()));
      tmptxt.ReplaceAll("%frac",TString::Format(fFractionFormat.Data(),fPieSlices[i]->GetValue()/sum));
      tmptxt.ReplaceAll("%perc",TString::Format(TString::Format("%s %s",fPercentFormat.Data(),"%s").Data(),(fPieSlices[i]->GetValue()/sum)*100,"%"));

      textlabel.SetTitle(tmptxt.Data());
      Double_t h = textlabel.GetYsize();
      Double_t w = textlabel.GetXsize();

      Float_t lx = fX+(fRadius+fPieSlices[i]->GetRadiusOffset()+label_off)*TMath::Cos(aphi);
      Float_t ly = fY+(fRadius+fPieSlices[i]->GetRadiusOffset()+label_off)*TMath::Sin(aphi)*radXY;

      Double_t lblang = 0;

      if (lblor==1) { // radial direction for the label
         aphi = TMath::ATan2(TMath::Sin(aphi)*radXY,TMath::Cos(aphi));
         lblang += aphi;
         if (lblang<=0) lblang += TMath::TwoPi();
         if (lblang>TMath::TwoPi()) lblang-= TMath::TwoPi();

         lx += h/2.*TMath::Sin(lblang);
         ly -= h/2.*TMath::Cos(lblang);

         // This control prevent text up-side
         if (lblang>TMath::PiOver2() && lblang<=3.*TMath::PiOver2()) {
            lx     += w*TMath::Cos(lblang)-h*TMath::Sin(lblang);
            ly     += w*TMath::Sin(lblang)+h*TMath::Cos(lblang);
            lblang -= TMath::Pi();
         }
      } else if (lblor==2) { // tangential direction of the labels
         aphi -=TMath::PiOver2();
         aphi = TMath::ATan2(TMath::Sin(aphi)*radXY,TMath::Cos(aphi));
         lblang += aphi;//-TMath::PiOver2();
         if (lblang<0) lblang+=TMath::TwoPi();

         lx -= w/2.*TMath::Cos(lblang);
         ly -= w/2.*TMath::Sin(lblang);

         if (lblang>TMath::PiOver2() && lblang<3.*TMath::PiOver2()) {
            lx     += w*TMath::Cos(lblang)-h*TMath::Sin(lblang);
            ly     += w*TMath::Sin(lblang)+h*TMath::Cos(lblang);
            lblang -= TMath::Pi();
         }
      } else { // horizontal labels (default direction)
         aphi = TMath::ATan2(TMath::Sin(aphi)*radXY,TMath::Cos(aphi));
         if (aphi>TMath::PiOver2() || aphi<=-TMath::PiOver2()) lx -= w;
         if (aphi<0)                                           ly -= h;
      }

      Float_t rphi = TMath::ATan2((ly-fY)*radXY,lx-fX);
      if (rphi < 0 && fIs3D && label_off>=0.)
         ly -= fHeight;

      if (optionSameColor) textlabel.SetTextColor((fPieSlices[i]->GetFillColor()));
      textlabel.PaintLatex(lx,ly,
                           lblang*180/TMath::Pi()+GetTextAngle(),
                           GetTextSize(), tmptxt.Data());
   }

   if (optionSame) return;

   // Draw title
   TPaveText *title = nullptr;
   if (auto obj = gPad->GetListOfPrimitives()->FindObject("title")) {
      title = dynamic_cast<TPaveText*>(obj);
   }

   // Check the OptTitle option
   if (strlen(GetTitle()) == 0 || gStyle->GetOptTitle() <= 0) {
      if (title) delete title;
      return;
   }

   // Height and width of the title
   Double_t ht = gStyle->GetTitleH();
   Double_t wt = gStyle->GetTitleW();
   if (ht<=0) ht = 1.1*gStyle->GetTitleFontSize();
   if (ht<=0) ht = 0.05; // minum height
   if (wt<=0) { // eval the width of the title
      TLatex l;
      l.SetTextSize(ht);
      l.SetTitle(GetTitle());
      // adjustment in case the title has several lines (#splitline)
      ht = TMath::Max(ht, 1.2*l.GetYsize()/(gPad->GetY2() - gPad->GetY1()));
      Double_t wndc = l.GetXsize()/(gPad->GetX2() - gPad->GetX1());
      wt = TMath::Min(0.7, 0.02+wndc);
   }

   if (title) {
      TText *t0 = (TText*)title->GetLine(0);
      if (t0) {
         if (!strcmp(t0->GetTitle(),GetTitle())) return;
         t0->SetTitle(GetTitle());
         if (wt > 0) title->SetX2NDC(title->GetX1NDC()+wt);
      }
      return;
   }

   Int_t talh = gStyle->GetTitleAlign()/10;
   if (talh < 1) talh = 1; else if (talh > 3) talh = 3;
   Int_t talv = gStyle->GetTitleAlign()%10;
   if (talv < 1) talv = 1; else if (talv > 3) talv = 3;
   Double_t xpos, ypos;
   xpos = gStyle->GetTitleX();
   ypos = gStyle->GetTitleY();
   if (talh == 2) xpos = xpos-wt/2.;
   if (talh == 3) xpos = xpos-wt;
   if (talv == 2) ypos = ypos+ht/2.;
   if (talv == 1) ypos = ypos+ht;

   title = new TPaveText(xpos,ypos-ht,xpos+wt,ypos,"blNDC");
   title->SetFillColor(gStyle->GetTitleFillColor());
   title->SetFillStyle(gStyle->GetTitleStyle());
   title->SetName("title");

   title->SetBorderSize(gStyle->GetTitleBorderSize());
   title->SetTextColor(gStyle->GetTitleTextColor());
   title->SetTextFont(gStyle->GetTitleFont(""));
   if (gStyle->GetTitleFont("")%10 > 2)
      title->SetTextSize(gStyle->GetTitleFontSize());
   title->AddText(GetTitle());

   title->SetBit(kCanDelete);

   title->Draw();
   title->Paint();
}

////////////////////////////////////////////////////////////////////////////////
/// Save primitive as a C++ statement(s) on output stream out

void TPie::SavePrimitive(std::ostream &out, Option_t *option)
{
   SavePrimitiveConstructor(out, Class(), "pie", TString::Format("\"%s\", \"%s\", %d", GetName(), TString(GetTitle()).ReplaceSpecialCppChars().Data(), fNvals));

   out << "   pie->SetCircle(" << fX << ", " << fY << ", " << fRadius << ");\n";
   out << "   pie->SetValueFormat(\"" << GetValueFormat() << "\");\n";
   out << "   pie->SetLabelFormat(\"" << GetLabelFormat() << "\");\n";
   out << "   pie->SetPercentFormat(\"" << GetPercentFormat() << "\");\n";
   out << "   pie->SetLabelsOffset(" << GetLabelsOffset() << ");\n";
   out << "   pie->SetAngularOffset(" << GetAngularOffset() << ");\n";

   SaveTextAttributes(out, "pie", 11, 0, 1, 62, 0.05);

   // Save the values for the slices
   for (Int_t i = 0; i < fNvals; ++i) {
      auto slice_name = TString::Format("pie->GetSlice(%d)", i);
      fPieSlices[i]->SavePrimitive(out, slice_name.Data());
   }

   out << "   gPad->Add(pie, \"" << TString(option).ReplaceSpecialCppChars() << "\");\n";
}

////////////////////////////////////////////////////////////////////////////////
/// Set the value of for the pseudo 3D view angle, in degree.
/// The range of the permitted values is: [0,90]

void TPie::SetAngle3D(Float_t val) {
   // check if val is in the permitted range
   while (val>360.) val -= 360.;
   while (val<0)    val += 360.;
   if      (val>=90 && val<180)   val = 180-val;
   else if (val>=180 && val<=360) val = 360-val;

   fAngle3D = val;
}

////////////////////////////////////////////////////////////////////////////////
/// Set the global angular offset for slices in degree [0,360]

void TPie::SetAngularOffset(Double_t offset)
{
   fAngularOffset = offset;

   while (fAngularOffset >= 360.)
      fAngularOffset -= 360.;
   while (fAngularOffset < 0.)
      fAngularOffset += 360.;
}

////////////////////////////////////////////////////////////////////////////////
/// Set the coordinates of the circle that describe the pie:
/// - the 1st and the 2nd arguments are the x and y center's coordinates.
/// - the 3rd value is the pie-chart's radius.
///
/// All the coordinates are in NDC space.

void TPie::SetCircle(Double_t x, Double_t y, Double_t rad)
{
   fX      = x;
   fY      = y;
   fRadius = rad;
}

////////////////////////////////////////////////////////////////////////////////
/// Set slice number "i" label. The first parameter is the index of the slice,
/// the other is the label text.

void TPie::SetEntryLabel(Int_t i, const char *text)
{
   // Set the Label of a single slice
   if (i>=0 && i<fNvals) fPieSlices[i]->SetTitle(text);
}

////////////////////////////////////////////////////////////////////////////////
/// Set the color for the slice's outline. "i" is the slice number.

void TPie::SetEntryLineColor(Int_t i, Int_t color)
{
   if (i>=0 && i<fNvals) fPieSlices[i]->SetLineColor(color);
}

////////////////////////////////////////////////////////////////////////////////
/// Set the style for the slice's outline. "i" is the slice number.

void TPie::SetEntryLineStyle(Int_t i, Int_t style)
{
   if (i>=0 && i<fNvals) fPieSlices[i]->SetLineStyle(style);
}

////////////////////////////////////////////////////////////////////////////////
/// Set the width of the slice's outline. "i" is the slice number.

void TPie::SetEntryLineWidth(Int_t i, Int_t width)
{
   if (i>=0 && i<fNvals) fPieSlices[i]->SetLineWidth(width);
}

////////////////////////////////////////////////////////////////////////////////
/// Set the color for the slice "i".

void TPie::SetEntryFillColor(Int_t i, Int_t color)
{
   if (i>=0 && i<fNvals) fPieSlices[i]->SetFillColor(color);
}

////////////////////////////////////////////////////////////////////////////////
/// Set the fill style for the "i" slice

void TPie::SetEntryFillStyle(Int_t i, Int_t style)
{
   if (i>=0 && i<fNvals) fPieSlices[i]->SetFillStyle(style);
}

////////////////////////////////////////////////////////////////////////////////
/// Set the distance, in the direction of the radius of the slice.

void TPie::SetEntryRadiusOffset(Int_t i, Double_t shift)
{
   if (i>=0 && i<fNvals) fPieSlices[i]->SetRadiusOffset(shift);
}

////////////////////////////////////////////////////////////////////////////////
/// Set the value of a slice

void TPie::SetEntryVal(Int_t i, Double_t val)
{
   if (i>=0 && i<fNvals) fPieSlices[i]->SetValue(val);
}

////////////////////////////////////////////////////////////////////////////////
/// Set the fill colors for all the TPie's slices.

void TPie::SetFillColors(Int_t *colors)
{
   if (!colors) return;
   for (Int_t i=0;i<fNvals;++i) fPieSlices[i]->SetFillColor(colors[i]);
}

////////////////////////////////////////////////////////////////////////////////
/// Set the height, in pixel, for the piechart if is drawn using
/// the pseudo-3d mode.
///
/// The default value is 20 pixel.

void TPie::SetHeight(Double_t val/*=20*/)
{
   fHeight = val;
}

////////////////////////////////////////////////////////////////////////////////
/// This method is used to customize the label format.
/// [See the class description for the possible labels format](\ref PIE01)

void TPie::SetLabelFormat(const char *fmt)
{
   fLabelFormat = fmt;
}

////////////////////////////////////////////////////////////////////////////////
/// Set numeric format in the label, is used if the label format
/// there is the modifier %frac, in this case the value is printed
/// using this format.
///
/// The numeric format use the standard C modifier used in stdio library:
/// %f, %2.1$, %e... for further documentation you can use the printf
/// man page ("man 3 printf" on linux)
///
/// Example:
/// ~~~ {.cpp}
///       mypie->SetLabelFormat("%txt (%frac)");
///       mypie->SetFractionFormat("2.1f");
/// ~~~

void TPie::SetFractionFormat(const char *fmt)
{
   fFractionFormat = fmt;
}

////////////////////////////////////////////////////////////////////////////////
/// Set the labels for all the slices.

void TPie::SetLabels(const char *lbls[])
{
   if (!lbls) return;
   for (Int_t i=0;i<fNvals;++i) fPieSlices[i]->SetTitle(lbls[i]);
}

////////////////////////////////////////////////////////////////////////////////
/// Set the distance between the label end the external line of the TPie.

void TPie::SetLabelsOffset(Float_t labelsoffset)
{
   fLabelsOffset = labelsoffset;
}

////////////////////////////////////////////////////////////////////////////////
/// Set the numeric format for the percent value of a slice, default: %3.1f

void TPie::SetPercentFormat(const char *fmt)
{
   fPercentFormat = fmt;
}

////////////////////////////////////////////////////////////////////////////////
/// Set the pie chart's radius' value.

void TPie::SetRadius(Double_t rad)
{
   if (rad>0) {
      fRadius = rad;
   } else {
      Warning("SetRadius",
              "It's not possible set the radius to a negative value");
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Set the numeric format the slices' values.
/// Used by %val (see SetLabelFormat()).

void TPie::SetValueFormat(const char *fmt)
{
   fValueFormat = fmt;
}

////////////////////////////////////////////////////////////////////////////////
/// Set X value.

void TPie::SetX(Double_t x)
{
   fX = x;
}

////////////////////////////////////////////////////////////////////////////////
/// Set Y value.

void TPie::SetY(Double_t y)
{
   fY = y;
}

////////////////////////////////////////////////////////////////////////////////
/// Return calculated angle for all slices
/// Only for internal use

std::vector<Float_t> TPie::GetSlicesAngles() const
{
   std::vector<Float_t> angles(2 * fNvals + 1, 0.);

   Double_t sum = GetSumOfEntriesValues();

   if (sum > 0)
      sum = 360. / sum;
   else
      sum = 1.;

   angles[0] = GetAngularOffset();
   for (Int_t i = 0; i < fNvals; ++i) {
      Float_t dphi = TMath::Abs(fPieSlices[i]->GetValue()) * sum;
      angles[2 * i + 1] = angles[2 * i] + dphi / 2.;
      angles[2 * i + 2] = angles[2 * i] + dphi;
   }

   return angles;
}


////////////////////////////////////////////////////////////////////////////////
/// Make the slices.
/// Deprecated, no longer necessary

void TPie::MakeSlices(Bool_t)
{
 }

////////////////////////////////////////////////////////////////////////////////
/// This method, mainly intended for internal use, ordered the  slices according their values.
/// The default (amode=kTRUE) is increasing order, but is also possible in decreasing order (amode=kFALSE).
///
/// If the merge_threshold>0 the slice that contains a quantity smaller than merge_threshold are merged
/// together

void TPie::SortSlices(Bool_t amode, Float_t merge_threshold)
{

   // main loop to order, bubble sort, the array
   Bool_t isDone = kFALSE;

   while (isDone==kFALSE) {
      isDone = kTRUE;

      for (Int_t i=0;i<fNvals-1;++i) { // loop over the values
         if ( (amode && (fPieSlices[i]->GetValue()>fPieSlices[i+1]->GetValue())) ||
              (!amode && (fPieSlices[i]->GetValue()<fPieSlices[i+1]->GetValue()))
            )
         {
            // exchange the order
            TPieSlice *tmpcpy = fPieSlices[i];
            fPieSlices[i] = fPieSlices[i+1];
            fPieSlices[i+1] = tmpcpy;

            isDone = kFALSE;
         }
      }  // end loop the values
   } // end main ordering loop

   if (merge_threshold>0) {
      // merge smallest slices
      TPieSlice *merged_slice = new TPieSlice("merged","other",this);
      merged_slice->SetRadiusOffset(0.);
      merged_slice->SetLineColor(1);
      merged_slice->SetLineStyle(1);
      merged_slice->SetLineWidth(1);
      merged_slice->SetFillColor(gStyle->GetColorPalette( (amode ? 0 : fNvals-1) ));
      merged_slice->SetFillStyle(1001);

      if (amode) {
         // search slices under the threshold
         Int_t iMerged = 0;
         for (;iMerged<fNvals&&fPieSlices[iMerged]->GetValue()<merge_threshold;++iMerged) {
            merged_slice->SetValue( merged_slice->GetValue()+fPieSlices[iMerged]->GetValue() );
         }

         // evaluate number of valid slices
         if (iMerged<=1) { // no slices to merge
            delete merged_slice;
         }
         else { // write a new array with the right dimension
            Int_t old_fNvals = fNvals;
            fNvals = fNvals-iMerged+1;
            TPieSlice **new_array = new TPieSlice*[fNvals];
            new_array[0] = merged_slice;
            for (Int_t i=0;i<old_fNvals;++i) {
               if (i<iMerged) delete fPieSlices[i];
               else new_array[i-iMerged+1] = fPieSlices[i];
            }
            delete [] fPieSlices;
            fPieSlices = new_array;
         }
      }
      else {
         Int_t iMerged = fNvals-1;
         for (;iMerged>=0&&fPieSlices[iMerged]->GetValue()<merge_threshold;--iMerged) {
            merged_slice->SetValue( merged_slice->GetValue()+fPieSlices[iMerged]->GetValue() );
         }

         // evaluate number of valid slices
         Int_t nMerged = fNvals-1-iMerged;
         if (nMerged<=1) { // no slices to merge
            delete merged_slice;
         }
         else { // write a new array with the right dimension
            Int_t old_fNvals = fNvals;
            fNvals = fNvals-nMerged+1;
            TPieSlice **new_array = new TPieSlice*[fNvals];
            new_array[fNvals-1] = merged_slice;
            for (Int_t i=old_fNvals-1;i>=0;--i) {
               if (i>iMerged) delete fPieSlices[i];
               else new_array[i-nMerged-1] = fPieSlices[i];
            }
            delete [] fPieSlices;
            fPieSlices = new_array;
         }

      }
   }
}

