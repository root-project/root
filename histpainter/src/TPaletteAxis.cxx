// @(#)root/histpainter:$Name:  $:$Id: TPaletteAxis.cxx,v 1.2 2002/11/20 09:50:34 brun Exp $
// Author: Rene Brun   15/11/2002

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "Riostream.h"
#include "TROOT.h"
#include "TPaletteAxis.h"
#include "TVirtualPad.h"
#include "TStyle.h"
#include "TClass.h"
#include "TMath.h"
#include "TView.h"

ClassImp(TPaletteAxis)

//______________________________________________________________________________
//
// a TPaletteAxis object is used to display the color palette when
// drawing 2-d histograms.
// The object is automatically created when drawing a 2-D histogram
// when the option "z" is specified.
// The object is added to the histogram list of functions and can be retrieved
// and its attributes changed with:
//  TPaletteAxis *palette = (TPaletteAxis*)h->GetListOfFunctions()->FindObject("palette");
//
// The palette can be interactively moved and resized. The context menu
// can be used to set the axis attributes.
//
// It is possible to select a range on the axis to set the min/max in z
//
//Begin_Html
/*
<img src="gif/palette.gif">
*/
//End_Html
//

//______________________________________________________________________________
TPaletteAxis::TPaletteAxis(): TBox()
{
// palette default constructor

   fH  = 0;
   SetName("");
}

//______________________________________________________________________________
TPaletteAxis::TPaletteAxis(Double_t x1, Double_t y1,Double_t x2, Double_t  y2, TH1 *h)
       :TBox(x1,y1,x2,y2)
{
// palette normal constructor

   fH = h;
   SetName("palette");
   TAxis *zaxis = fH->GetZaxis();
   fAxis.ImportAxisAttributes(zaxis);
   if (gPad->GetView()) SetBit(kHasView);
}

//______________________________________________________________________________
TPaletteAxis::~TPaletteAxis()
{
   if (fH) fH->GetListOfFunctions()->Remove(this);
}

//______________________________________________________________________________
TPaletteAxis::TPaletteAxis(const TPaletteAxis &palette) : TBox(palette)
{
   ((TPaletteAxis&)palette).Copy(*this);
}

//______________________________________________________________________________
void TPaletteAxis::Copy(TObject &obj) const
{
//*-*-*-*-*-*-*-*-*-*-*Copy this pave to pave*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  ======================

   TBox::Copy(obj);
   ((TPaletteAxis&)obj).fH    = fH;
   ((TPaletteAxis&)obj).fName = fName;
}

//______________________________________________________________________________
Int_t TPaletteAxis::DistancetoPrimitive(Int_t px, Int_t py)
{
   //check if mouse on the axis region
   Int_t plxmax = gPad->XtoAbsPixel(fX2);
   Int_t plymin = gPad->YtoAbsPixel(fY1);
   Int_t plymax = gPad->YtoAbsPixel(fY2);
   if (px > plxmax && px < plxmax+30 && py >= plymax &&py <= plymin) return px-plxmax;

   //otherwise check if inside the box
   return TBox::DistancetoPrimitive(px,py);
}

//______________________________________________________________________________
void TPaletteAxis::ExecuteEvent(Int_t event, Int_t px, Int_t py)
{
   //check if mouse on the axis region
   static Int_t kmode = 0;
   Int_t plxmin = gPad->XtoAbsPixel(fX1);
   Int_t plxmax = gPad->XtoAbsPixel(fX2);
   if (kmode != 0 || px <= plxmax) {
      if (event == kButton1Down) kmode = 1;
      TBox::ExecuteEvent(event,px,py);
      if (event == kButton1Up) kmode = 0;
      return;
   }
   gPad->SetCursor(kHand);
   static Double_t ratio1, ratio2;
   static Int_t px1old, py1old, px2old, py2old;
   Double_t temp, xmin,xmax;

   switch (event) {

   case kButton1Down:
      ratio1 = (gPad->AbsPixeltoY(py) - fY1)/(fY2 - fY1);
      py1old = gPad->YtoAbsPixel(fY1+ratio1*(fY2 - fY1));
      px1old = plxmin;
      px2old = plxmax;
      py2old = py1old;
      gVirtualX->DrawBox(px1old, py1old, px2old, py2old, TVirtualX::kHollow);
      gVirtualX->SetLineColor(-1);
      // No break !!!

   case kButton1Motion:
      gVirtualX->DrawBox(px1old, py1old, px2old, py2old, TVirtualX::kHollow);
      ratio2 = (gPad->AbsPixeltoY(py) - fY1)/(fY2 - fY1);
      py2old = gPad->YtoAbsPixel(fY1+ratio2*(fY2 - fY1));
      gVirtualX->DrawBox(px1old, py1old, px2old, py2old, TVirtualX::kHollow);
      break;

   case kButton1Up:
      ratio2 = (gPad->AbsPixeltoY(py) - fY1)/(fY2 - fY1);
      xmin = ratio1;
      xmax = ratio2;
      if (xmin > xmax) {
         temp   = xmin;
         xmin   = xmax;
         xmax   = temp;
         temp   = ratio1;
         ratio1 = ratio2;
         ratio2 = temp;
      }
      if (ratio2 - ratio1 > 0.05) {
         if (fH->GetDimension() == 2) {
            Float_t zmin = fH->GetMinimum();
            Float_t zmax = fH->GetMaximum();
            Float_t newmin = zmin + (zmax-zmin)*ratio1;
            Float_t newmax = zmin + (zmax-zmin)*ratio2;
            if(newmin < zmin)newmin = fH->GetBinContent(fH->GetMinimumBin());
            if(newmax > zmax)newmax = fH->GetBinContent(fH->GetMaximumBin());
            fH->SetMinimum(newmin);
            fH->SetMaximum(newmax);
            fH->SetBit(TH1::kIsZoomed);
         }
         gPad->Modified(kTRUE);
      }
      gVirtualX->SetLineColor(-1);
      kmode = 0;
      break;
   }
}

//______________________________________________________________________________
void TPaletteAxis::Paint(Option_t *)
{

   SetFillStyle(1001);
   Double_t ymin = fY1;
   Double_t ymax = fY2;
   Double_t xmin = fX1;
   Double_t xmax = fX2;
   Double_t wmin = fH->GetMinimum();
   Double_t wmax = fH->GetMaximum();
   Double_t wlmin = wmin;
   Double_t wlmax = wmax;
   Double_t y1,y2;
   //if (Hoption.Logz) {
   if (gPad->GetLogz()) {
      wlmin = TMath::Log10(wmin);
      wlmax = TMath::Log10(wmax);
   }
   Double_t ws    = wlmax-wlmin;
   Int_t ncolors = gStyle->GetNumberOfColors();
   Int_t ndivz = TMath::Abs(fH->GetContour());
   Int_t theColor,color;
   Double_t scale = ndivz/(wlmax - wlmin);
   for (Int_t i=0;i<ndivz;i++) {
      Double_t w1 = fH->GetContourLevel(i);
      if (w1 < wlmin) w1 = wlmin;
      Double_t w2 = wlmax;
      if (i < ndivz-1) w2 = fH->GetContourLevel(i+1);
      if (w2 <= wlmin) continue;
      y1 = ymin + (w1-wlmin)*(ymax-ymin)/ws;
      y2 = ymin + (w2-wlmin)*(ymax-ymin)/ws;
      color = Int_t(0.01+(w1-wlmin)*scale);
      theColor = Int_t((color+0.99)*Float_t(ncolors)/Float_t(ndivz));
      SetFillColor(gStyle->GetColorPalette(theColor));
      TAttFill::Modify();
      gPad->PaintBox(xmin,y1,xmax,y2);
   }
   Int_t ndiv  = fH->GetZaxis()->GetNdivisions()%100; //take primary divisions only
   char chopt[5] = "";
   chopt[0] = 0;
   strcat(chopt, "+L");
   if (ndiv < 0) {
      ndiv =TMath::Abs(ndiv);
      strcat(chopt, "N");
   }
   if (gPad->GetLogz()) {
      wmin = TMath::Power(10.,wlmin);
      wmax = TMath::Power(10.,wlmax);
      strcat(chopt, "G");
   }
   fAxis.PaintAxis(xmax,ymin,xmax,ymax,wmin,wmax,ndiv,chopt);
}

//______________________________________________________________________________
void TPaletteAxis::SavePrimitive(ofstream &, Option_t *)
{
   // Save primitive as a C++ statement(s) on output stream out.
}

//______________________________________________________________________________
void TPaletteAxis::UnZoom()
{
   TView *view = gPad->GetView();
   if (view) {
      delete view;
      gPad->SetView(0);
   }
   fH->GetZaxis()->SetRange(0,0);
   if (fH->GetDimension() == 2) {
      fH->SetMinimum();
      fH->SetMaximum();
      fH->ResetBit(TH1::kIsZoomed);
   }
}
