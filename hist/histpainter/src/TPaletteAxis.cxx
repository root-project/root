// @(#)root/histpainter:$Id$
// Author: Rene Brun   15/11/2002

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "TROOT.h"
#include "TPaletteAxis.h"
#include "TVirtualPad.h"
#include "TStyle.h"
#include "TMath.h"
#include "TView.h"
#include "TH1.h"
#include "TGaxis.h"
#include "TLatex.h"

#include <cstdio>
#include <iostream>
#include <memory>



////////////////////////////////////////////////////////////////////////////////

/*! \class TPaletteAxis
    \ingroup Histpainter
    \brief The palette painting class.

A `TPaletteAxis` object is used to display the color palette when
drawing 2-d histograms.

The `TPaletteAxis` is automatically created drawn when drawing a 2-D
histogram when the option "Z" is specified.

A `TPaletteAxis` object is added to the histogram list of functions and
can be retrieved doing:

    TPaletteAxis *palette = (TPaletteAxis*)h->GetListOfFunctions()->FindObject("palette");

then the pointer `palette` can be used to change the palette attributes.

Because the palette is created at painting time only, one must issue a:

    gPad->Update();

before retrieving the palette pointer in order to create the palette. The following
macro gives an example.

Begin_Macro(source)
{
   auto h2 = new TH2F("h2","Example of a resized palette ",40,-4,4,40,-20,20);
   Float_t px, py;
   for (Int_t i = 0; i < 25000; i++) {
      gRandom->Rannor(px,py);
      h2->Fill(px,5*py);
   }
   gStyle->SetPalette(1);
   h2->Draw("COLZ");
   gPad->Update();
   auto palette = (TPaletteAxis*)h2->GetListOfFunctions()->FindObject("palette");
   palette->SetY2NDC(0.7);
}
End_Macro

`TPaletteAxis` inherits from `TBox` and `TPave`. The methods
allowing to specify the palette position are inherited from these two classes.

The palette can be interactively moved and resized. The context menu
can be used to set the axis attributes.

It is possible to select a range on the axis to set the min/max in z

As default labels and ticks are drawn by `TGAxis` at equidistant (lin or log)
points as controlled by SetNdivisions.
If option "CJUST" is given labels and ticks are justified at the
color boundaries defined by the contour levels.
In this case no optimization can be done. It is responsibility of the
user to adjust minimum, maximum of the histogram and/or the contour levels
to get a reasonable look of the plot.
Only overlap of the labels is avoided if too many contour levels are used.

This option is especially useful with user defined contours.
An example is shown here:

Begin_Macro(source)
{
   gStyle->SetOptStat(0);
   auto c = new TCanvas("c","exa_CJUST",300,10,400,400);
   auto hpxpy = new TH2F("hpxpy","py vs px",40,-4,4,40,-4,4);
   // Fill histograms randomly
   TRandom3 randomNum;
   Float_t px, py;
   for (Int_t i = 0; i < 25000; i++) {
      randomNum.Rannor(px,py);
      hpxpy->Fill(px,py);
   }
   hpxpy->SetMaximum(200);
   Double_t zcontours[5] = {0, 20, 40, 80, 120};
   hpxpy->SetContour(5, zcontours);
   hpxpy->GetZaxis()->SetTickSize(0.01);
   hpxpy->GetZaxis()->SetLabelOffset(0.01);
   gPad->SetRightMargin(0.13);
   hpxpy->SetTitle("User contours, CJUST");
   hpxpy->Draw("COL Z CJUST");
}
End_Macro
*/


////////////////////////////////////////////////////////////////////////////////
/// Palette default constructor.

TPaletteAxis::TPaletteAxis()
{
   fH = nullptr;
   fLog = 0;
   SetName("");
}


////////////////////////////////////////////////////////////////////////////////
/// Palette normal constructor.

TPaletteAxis::TPaletteAxis(Double_t x1, Double_t y1, Double_t x2, Double_t  y2, TH1 *h)
   : TPave(x1, y1, x2, y2)
{
   fH = h;
   fLog = 0;
   if (!fH) return;
   SetName("palette");
   TAxis *zaxis = fH->GetZaxis();
   fAxis.ImportAxisAttributes(zaxis);
   if (gPad->GetView()) SetBit(kHasView);
}


////////////////////////////////////////////////////////////////////////////////
/// Palette normal constructor.

TPaletteAxis::TPaletteAxis(Double_t x1, Double_t y1, Double_t x2, Double_t  y2, Double_t min, Double_t max)
   : TPave(x1, y1, x2, y2)
{
   fH = nullptr;
   fLog = 0;
   fAxis.SetWmin(min);
   fAxis.SetWmax(max);
   SetName("palette");
   if (gPad->GetView()) SetBit(kHasView);
}


////////////////////////////////////////////////////////////////////////////////
/// Palette normal constructor.

TPaletteAxis::TPaletteAxis(Double_t x1, Double_t y1, Double_t x2, Double_t  y2, TAxis *ax)
   : TPave(x1, y1, x2, y2)
{
   fH = nullptr;
   fLog = 0;
   SetName("palette");
   fAxis.ImportAxisAttributes(ax);
   if (gPad->GetView()) SetBit(kHasView);
}


////////////////////////////////////////////////////////////////////////////////
/// Palette destructor.

TPaletteAxis::~TPaletteAxis()
{
}


////////////////////////////////////////////////////////////////////////////////
/// Palette copy constructor.

TPaletteAxis::TPaletteAxis(const TPaletteAxis &palette) : TPave(palette)
{
   palette.TPaletteAxis::Copy(*this);
}


////////////////////////////////////////////////////////////////////////////////
/// Assignment operator.

TPaletteAxis& TPaletteAxis::operator=(const TPaletteAxis &orig)
{
   if (this != &orig)
      orig.TPaletteAxis::Copy(*this);
   return *this;
}


////////////////////////////////////////////////////////////////////////////////
/// Copy a palette to a palette.

void TPaletteAxis::Copy(TObject &obj) const
{
   TPave::Copy(obj);
   ((TPaletteAxis&)obj).fH    = fH;
}

////////////////////////////////////////////////////////////////////////////////
/// Check if mouse on the axis region.

Int_t TPaletteAxis::DistancetoPrimitive(Int_t px, Int_t py)
{
   Bool_t isHorizontal = GetX2NDC() - GetX1NDC() > GetY2NDC() - GetY1NDC();
   Int_t plxmin = gPad->XtoAbsPixel(GetX1());
   Int_t plxmax = gPad->XtoAbsPixel(GetX2());
   Int_t plymin = gPad->YtoAbsPixel(GetY1());
   Int_t plymax = gPad->YtoAbsPixel(GetY2());

   if (isHorizontal) {
      if (px >= plxmin && px <= plxmax && py > plymin && py <= plymin + 30)
         return py - plymin;
   } else {
      if (px > plxmax && px < plxmax + 30 && py >= plymax && py <= plymin)
         return px - plxmax;
   }

   //otherwise check if inside the box
   return TPave::DistancetoPrimitive(px, py);
}

class TPaletteAxisInteractive : public TVirtualPad::TInteractive {
   public:
      Double_t ratio1 = 0, ratio2 = 1;
      Bool_t fHorizontal = kFALSE;
      TPaletteAxisInteractive(Bool_t h) { fHorizontal = h; }
      void SetPosition(TVirtualPad &parent, Int_t px, Int_t py, Double_t x1, Double_t y1, Double_t x2, Double_t y2, Bool_t first = kFALSE)
      {
         Double_t r = 0;
         if (fHorizontal)
            r = (parent.AbsPixeltoX(px) - x1) / (x2 - x1);
         else
            r = (parent.AbsPixeltoY(py) - y1) / (y2 - y1);

         ratio2 = r;
         if (first)
            ratio1 = r;
      }

      void Paint(TVirtualPad &parent, Double_t x1, Double_t y1, Double_t x2, Double_t y2)
      {
         if (fHorizontal)
            parent.PaintBox(x1 + ratio1 * (x2 - x1), y1, x1 + ratio2 * (x2 - x1), y2, "ilpaletteaxis");
         else
            parent.PaintBox(x1, y1 + ratio1 * (y2 - y1), x2, y1 + ratio2 * (y2 - y1), "ilpaletteaxis");
      }


};

////////////////////////////////////////////////////////////////////////////////
/// Check if mouse on the axis region.

void TPaletteAxis::ExecuteEvent(Int_t event, Int_t px, Int_t py)
{
   if (!gPad) return;

   auto &parent = *gPad;

   auto inter0 = parent.Interactive(this);
   auto inter = dynamic_cast<TPaletteAxisInteractive *> (inter0);

   Bool_t isHorizontal = GetX2NDC() - GetX1NDC() > GetY2NDC() - GetY1NDC();

   // when mouse pointer over palette itself - use TBox interactivity
   Bool_t useBoxHandler = isHorizontal ? (py <= parent.YtoAbsPixel(GetY1())) : (px <= parent.XtoAbsPixel(GetX2()));

   if (!inter && (inter0 || useBoxHandler)) {
      TBox::ExecuteEvent(event, px, py);
      return;
   }

   parent.SetCursor(kHand);

   switch (event) {

      case kButton1Down:
         inter = new TPaletteAxisInteractive(isHorizontal);
         parent.Interactive(this, inter);
         // No break !!!

      case kButton1Motion:
         if (inter) {
            inter->SetPosition(parent, px, py, GetX1(), GetY1(), GetX2(), GetY2(), event == kButton1Down);
            inter->Paint(parent, GetX1(), GetY1(), GetX2(), GetY2());
            parent.UpdateAsync();
         }
         break;

      case kButton1Up:
         if (gROOT->IsEscaped()) {
            gROOT->SetEscape(kFALSE);
            break;
         }
         if (!inter)
            break;

         inter->SetPosition(parent, px, py, GetX1(), GetY1(), GetX2(), GetY2());
         if (inter->ratio1 > inter->ratio2)
            std::swap(inter->ratio1, inter->ratio2);
         if ((inter->ratio2 - inter->ratio1 > 0.05) && fH && (fH->GetDimension() == 2)) {
            Double_t zmin = fH->GetMinimum();
            Double_t zmax = fH->GetMaximum();
            if (GetLog()) {
               if (zmin <= 0 && zmax > 0)
                  zmin = TMath::Min((Double_t)1, (Double_t)0.001 * zmax);
               zmin = TMath::Log10(zmin);
               zmax = TMath::Log10(zmax);
            }
            Double_t newmin = zmin + (zmax - zmin) * inter->ratio1;
            Double_t newmax = zmin + (zmax - zmin) * inter->ratio2;
            if (newmin < zmin)
               newmin = fH->GetBinContent(fH->GetMinimumBin());
            if (newmax > zmax)
               newmax = fH->GetBinContent(fH->GetMaximumBin());
            if (GetLog()) {
               newmin = TMath::Exp(2.302585092994 * newmin);
               newmax = TMath::Exp(2.302585092994 * newmax);
            }
            fH->SetMinimum(newmin);
            fH->SetMaximum(newmax);
            fH->SetBit(TH1::kIsZoomed);
            parent.Modified(kTRUE);
         }
         parent.Interactive(); // clear interactive object
         break;
   }
}

////////////////////////////////////////////////////////////////////////////////
/// Returns whether the palette is in log scale.
///
/// By default, the palette axis is logarithmic if the TPad's option Logz is set.
/// However, in some cases, such as with TScatter2D, the color represents a fourth dimension,
/// and the log scale is not defined by the Z axis but by a dedicated setting specified
/// through a separate option.
///
/// This method handles these various cases and returns the correct state of the palette axis.

Int_t TPaletteAxis::GetLog() const
{
   if (fLog == 0) return gPad->GetLogz();
   return fLog-1;
}

////////////////////////////////////////////////////////////////////////////////
/// Returns the color index of the bin (i,j).
///
/// This function should be used after an histogram has been plotted with the
/// option COL or COLZ like in the following example:
///
///     h2->Draw("COLZ");
///     gPad->Update();
///     TPaletteAxis *palette = (TPaletteAxis*)h2->GetListOfFunctions()->FindObject("palette");
///     Int_t ci = palette->GetBinColor(20,15);
///
/// Then it is possible to retrieve the RGB components in the following way:
///
///     TColor *c = gROOT->GetColor(ci);
///     float x,y,z;
///     c->GetRGB(x,y,z);

Int_t TPaletteAxis::GetBinColor(Int_t i, Int_t j)
{
   if (!fH) return 0;
   Double_t zc = fH->GetBinContent(i, j);
   return GetValueColor(zc);
}


////////////////////////////////////////////////////////////////////////////////
/// Displays the z value corresponding to cursor position py.

char *TPaletteAxis::GetObjectInfo(Int_t /* px */, Int_t py) const
{
   Double_t z;
   static char info[64];

   Double_t zmin = 0.;
   Double_t zmax = 0.;
   if (fH) {
      zmin = fH->GetMinimum();
      zmax = fH->GetMaximum();
   }
   Int_t   y1   = gPad->GetWh() - gPad->VtoPixel(fY1NDC);
   Int_t   y2   = gPad->GetWh() - gPad->VtoPixel(fY2NDC);
   Int_t   y    = gPad->GetWh() - py;

   if (GetLog()) {
      if (zmin <= 0 && zmax > 0) zmin = TMath::Min((Double_t)1,
                                                      (Double_t)0.001 * zmax);
      Double_t zminl = TMath::Log10(zmin);
      Double_t zmaxl = TMath::Log10(zmax);
      Double_t zl    = (zmaxl - zminl) * ((Double_t)(y - y1) / (Double_t)(y2 - y1)) + zminl;
      z = TMath::Power(10., zl);
   } else {
      z = (zmax - zmin) * ((Double_t)(y - y1) / (Double_t)(y2 - y1)) + zmin;
   }

   snprintf(info, 64, "(z=%g)", z);
   return info;
}


////////////////////////////////////////////////////////////////////////////////
/// Returns the color index of the given z value
///
/// This function should be used after an histogram has been plotted with the
/// option COL or COLZ like in the following example:
///
///     h2->Draw("COLZ");
///     gPad->Update();
///     TPaletteAxis *palette = (TPaletteAxis*)h2->GetListOfFunctions()->FindObject("palette");
///     Int_t ci = palette->GetValueColor(30.);
///
/// Then it is possible to retrieve the RGB components in the following way:
///
///     TColor *c = gROOT->GetColor(ci);
///     float x,y,z;
///     c->GetRGB(x,y,z);

Int_t TPaletteAxis::GetValueColor(Double_t zc)
{
   if (!fH) return 0;

   Double_t wmin  = fH->GetMinimum();
   Double_t wmax  = fH->GetMaximum();
   Double_t wlmin = wmin;
   Double_t wlmax = wmax;

   if (GetLog()) {
      if (wmin <= 0 && wmax > 0) wmin = TMath::Min((Double_t)1,
                                                      (Double_t)0.001 * wmax);
      wlmin = TMath::Log10(wmin);
      wlmax = TMath::Log10(wmax);
   }

   Int_t ncolors = gStyle->GetNumberOfColors();
   Int_t ndivz =0;
   if (fH) ndivz = fH->GetContour();
   if (ndivz == 0) return 0;
   ndivz = TMath::Abs(ndivz);
   Int_t theColor, color;
   Double_t scale = ndivz / (wlmax - wlmin);

   if (fH->TestBit(TH1::kUserContour) && GetLog()) zc = TMath::Log10(zc);
   if (zc < wlmin) zc = wlmin;

   color = Int_t(0.01 + (zc - wlmin) * scale);

   theColor = Int_t((color + 0.99) * Double_t(ncolors) / Double_t(ndivz));
   return gStyle->GetColorPalette(theColor);
}


////////////////////////////////////////////////////////////////////////////////
/// Paint the palette.

void TPaletteAxis::Paint(Option_t *)
{
   ConvertNDCtoPad();

   SetFillStyle(1001);
   Double_t ymin = GetY1();
   Double_t ymax = GetY2();
   Double_t xmin = GetX1();
   Double_t xmax = GetX2();
   Double_t wmin, wmax;
   if (fH) {
      wmin = fH->GetMinimum();
      wmax = fH->GetMaximum();
   } else {
      wmin = fAxis.GetWmin();
      wmax = fAxis.GetWmax();
   }
   Double_t wlmin = wmin;
   Double_t wlmax = wmax;
   Double_t b1, b2, w1, w2, zc;

   if ((wlmax - wlmin) <= 0) {
      Double_t mz = wlmin * 0.1;
      if (mz == 0) mz = 0.1;
      wlmin = wlmin - mz;
      wlmax = wlmax + mz;
      wmin  = wlmin;
      wmax  = wlmax;
   }

   Bool_t isHorizontal = GetX2NDC() - GetX1NDC() > GetY2NDC() - GetY1NDC();

   if (GetLog()) {
      if (wmin <= 0 && wmax > 0) wmin = TMath::Min((Double_t)1, (Double_t)0.001 * wmax);
      wlmin = TMath::Log10(wmin);
      wlmax = TMath::Log10(wmax);
   }
   Double_t ws    = wlmax - wlmin;
   Int_t ncolors = gStyle->GetNumberOfColors();
   Int_t ndivz;
   if (fH)
      ndivz = fH->GetContour();
   else
      ndivz = ncolors;
   if (ndivz == 0)
      return;
   ndivz = TMath::Abs(ndivz);
   Int_t theColor, color;
   // import Attributes already here since we might need them for CJUST
   if (fH && fH->GetDimension() == 2) {
      fAxis.ImportAxisAttributes(fH->GetZaxis());
      TString ztit = fAxis.GetTitle();
      if (ztit.Index(";") > 0) {
         ztit.Remove(ztit.Index(";"), ztit.Length());
         fAxis.SetTitle(ztit.Data());
      }
   }
   // the 3D histogram's title is stored in Zaxis
   if (fH && fH->GetDimension() >  2) {
      TString ztit = fH->GetZaxis()->GetTitle();
      fAxis.SetTitle("");
      if (ztit.Index(";")>0) {
         ztit.Remove(0,ztit.Index(";")+1);
         fAxis.SetTitle(ztit.Data());
      }
   }
   // case option "CJUST": put labels directly at color boundaries
   std::unique_ptr<TLatex> label;
   std::unique_ptr<TLine> line;
   Double_t prevlab = 0;
   if (fH) {
      TString opt(fH->GetDrawOption());
      if (opt.Contains("CJUST", TString::kIgnoreCase)) {
         label = std::make_unique<TLatex>();
         label->SetTextFont(fAxis.GetLabelFont());
         label->SetTextColor(fAxis.GetLabelColor());
         if (isHorizontal)
            label->SetTextAlign(kHAlignCenter + kVAlignTop);
         else
            label->SetTextAlign(kHAlignLeft + kVAlignCenter);
         line = std::make_unique<TLine>();
         line->SetLineColor(fAxis.GetLineColor());
         if (isHorizontal)
            line->PaintLine(xmin, ymin, xmax, ymin);
         else
            line->PaintLine(xmax, ymin, xmax, ymax);
      }
   }
   Double_t scale = ndivz / (wlmax - wlmin);
   Double_t dw = (wlmax - wlmin) / ndivz;
   for (Int_t i = 0; i < ndivz; i++) {

      if (fH) zc = fH->GetContourLevel(i);
      else    zc = wlmin + i*dw;
      if (fH && fH->TestBit(TH1::kUserContour) && GetLog())
         zc = TMath::Log10(zc);
      w1 = zc;
      if (w1 < wlmin) w1 = wlmin;

      w2 = wlmax;
      if (i < ndivz - 1) {
         if (fH) zc = fH->GetContourLevel(i + 1);
         else    zc = wlmin + (i+1)*dw;
         if (fH && fH->TestBit(TH1::kUserContour) && GetLog())
            zc = TMath::Log10(zc);
         w2 = zc;
      }

      if (w2 <= wlmin) continue;
      if (isHorizontal) {
         b1 = xmin + (w1 - wlmin) * (xmax - xmin) / ws;
         b2 = xmin + (w2 - wlmin) * (xmax - xmin) / ws;
      } else {
         b1 = ymin + (w1 - wlmin) * (ymax - ymin) / ws;
         b2 = ymin + (w2 - wlmin) * (ymax - ymin) / ws;
      }

      if (fH && fH->TestBit(TH1::kUserContour)) {
         color = i;
      } else {
         color = Int_t(0.01 + (w1 - wlmin) * scale);
      }

      theColor = Int_t((color + 0.99) * Double_t(ncolors) / Double_t(ndivz));
      SetFillColor(gStyle->GetColorPalette(theColor));
      TAttFill::Modify();
      if (isHorizontal)
         gPad->PaintBox(b1, ymin, b2, ymax);
      else
         gPad->PaintBox(xmin, b1, xmax, b2);
      // case option "CJUST": put labels directly
      if (fH && label) {
         Double_t lof = fAxis.GetLabelOffset()*(gPad->GetUxmax()-gPad->GetUxmin());
         // the following assumes option "S"
         Double_t tlength = fAxis.GetTickSize() * (gPad->GetUxmax()-gPad->GetUxmin());
         Double_t lsize = fAxis.GetLabelSize();
         Double_t lsize_user = lsize*(gPad->GetUymax()-gPad->GetUymin());
         Double_t zlab = fH->GetContourLevel(i);
         if (GetLog() && !fH->TestBit(TH1::kUserContour)) {
            zlab = TMath::Power(10, zlab);
         }
         // make sure labels dont overlap
         if (i == 0 || (b1 - prevlab) > 1.5*lsize_user) {
            if (isHorizontal)
               label->PaintLatex(b1, ymin - lof, 0, lsize, TString::Format("%g", zlab));
            else
               label->PaintLatex(xmax + lof, b1, 0, lsize, TString::Format("%g", zlab));
            prevlab = b1;
         }
         if (isHorizontal)
            line->PaintLine(b2, ymin + tlength, b2, ymin);
         else
            line->PaintLine(xmax - tlength, b1, xmax, b1);
         if (i == ndivz - 1) {
            // label + tick at top of axis
            if (fH && (b2 - prevlab > 1.5*lsize_user)) {
               if (isHorizontal)
                  label->PaintLatex(b2, ymin - lof, 0, lsize, TString::Format("%g", fH->GetMaximum()));
               else
                  label->PaintLatex(xmax + lof, b2, 0, lsize, TString::Format("%g", fH->GetMaximum()));
            }
            if (isHorizontal)
               line->PaintLine(b1, ymin + tlength, b1, ymin);
            else
               line->PaintLine(xmax - tlength, b2, xmax, b2);
         }
      }
   }

   // case option "CJUST" - just cleanup
   if (label)
      return;

   // Take primary divisions only
   Int_t ndiv;
   if (fH)
      ndiv = fH->GetZaxis()->GetNdivisions();
   else
      ndiv = fAxis.GetNdiv();
   Bool_t isOptimized = ndiv > 0;
   Int_t absDiv = TMath::Abs(ndiv);
   Int_t maxD = absDiv / 1000000;
   ndiv = absDiv % 100 + maxD * 1000000;

   TString chopt = "S+L";
   if (!isOptimized)
      chopt.Append("N");
   if (GetLog()) {
      wmin = TMath::Power(10., wlmin);
      wmax = TMath::Power(10., wlmax);
      chopt.Append("G");
   }
   if (isHorizontal)
      fAxis.PaintAxis(xmin, ymin, xmax, ymin, wmin, wmax, ndiv, chopt);
   else
      fAxis.PaintAxis(xmax, ymin, xmax, ymax, wmin, wmax, ndiv, chopt);
}


////////////////////////////////////////////////////////////////////////////////
/// Save primitive as a C++ statement(s) on output stream out.

void TPaletteAxis::SavePrimitive(std::ostream &out, Option_t * /*= ""*/)
{
   if (!fH)
      return;

   SavePrimitiveConstructor(out, Class(), "palette", GetSavePaveArgs(fH->GetName(), kFALSE));
   out << "   palette->SetNdivisions(" << fH->GetZaxis()->GetNdivisions() << ");\n";
   out << "   palette->SetAxisColor(" << TColor::SavePrimitiveColor(fH->GetZaxis()->GetAxisColor()) << ");\n";
   out << "   palette->SetLabelColor(" << TColor::SavePrimitiveColor(fH->GetZaxis()->GetLabelColor()) << ");\n";
   out << "   palette->SetLabelFont(" << fH->GetZaxis()->GetLabelFont() << ");\n";
   out << "   palette->SetLabelOffset(" << fH->GetZaxis()->GetLabelOffset() << ");\n";
   out << "   palette->SetLabelSize(" << fH->GetZaxis()->GetLabelSize() << ");\n";
   out << "   palette->SetMaxDigits(" << fH->GetZaxis()->GetMaxDigits() << ");\n";
   out << "   palette->SetTickLength(" << fH->GetZaxis()->GetTickLength() << ");\n";
   out << "   palette->SetTitleOffset(" << fH->GetZaxis()->GetTitleOffset() << ");\n";
   out << "   palette->SetTitleSize(" << fH->GetZaxis()->GetTitleSize() << ");\n";
   out << "   palette->SetTitleColor(" << TColor::SavePrimitiveColor(fH->GetZaxis()->GetTitleColor()) << ");\n";
   out << "   palette->SetTitleFont(" << fH->GetZaxis()->GetTitleFont() << ");\n";
   out << "   palette->SetTitle(\"" << TString(fH->GetZaxis()->GetTitle()).ReplaceSpecialCppChars() << "\");\n";
   SaveFillAttributes(out, "palette", -1, -1);
   SaveLineAttributes(out, "palette", 1, 1, 1);
}

////////////////////////////////////////////////////////////////////////////////
/// Unzoom the palette

void TPaletteAxis::UnZoom()
{
   if (!fH)
      return;
   // if view exists - it will be deleted
   if (gPad)
      gPad->SetView(nullptr);
   fH->GetZaxis()->SetRange(0, 0);
   if (fH->GetDimension() == 2) {
      fH->SetMinimum();
      fH->SetMaximum();
      fH->ResetBit(TH1::kIsZoomed);
   }
}
