// @(#)root/histpainter:$Name:  $:$Id: TPaletteAxis.cxx,v 1.10 2005/03/21 09:15:05 brun Exp $
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
TPaletteAxis::TPaletteAxis(): TPave()
{
// palette default constructor

   fH  = 0;
   SetName("");
}

//______________________________________________________________________________
TPaletteAxis::TPaletteAxis(Double_t x1, Double_t y1,Double_t x2, Double_t  y2, TH1 *h)
       :TPave(x1,y1,x2,y2)
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
TPaletteAxis::TPaletteAxis(const TPaletteAxis &palette) : TPave(palette)
{
   ((TPaletteAxis&)palette).Copy(*this);
}

//______________________________________________________________________________
void TPaletteAxis::Copy(TObject &obj) const
{
//*-*-*-*-*-*-*-*-*-*-*Copy this pave to pave*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  ======================

   TPave::Copy(obj);
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
   return TPave::DistancetoPrimitive(px,py);
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
//*-* In case pave coordinates have been modified, recompute NDC coordinates
      Double_t dpx  = gPad->GetX2() - gPad->GetX1();
      Double_t dpy  = gPad->GetY2() - gPad->GetY1();
      Double_t xp1  = gPad->GetX1();
      Double_t yp1  = gPad->GetY1();
      fX1NDC = (fX1-xp1)/dpx;
      fY1NDC = (fY1-yp1)/dpy;
      fX2NDC = (fX2-xp1)/dpx;
      fY2NDC = (fY2-yp1)/dpy;
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
            Double_t zmin = fH->GetMinimum();
            Double_t zmax = fH->GetMaximum();
            if(gPad->GetLogz()){
               if (zmin <= 0 && zmax > 0) zmin = TMath::Min((Double_t)1,
                                                            (Double_t)0.001*zmax);
               zmin = TMath::Log10(zmin);
               zmax = TMath::Log10(zmax);
            }
            Double_t newmin = zmin + (zmax-zmin)*ratio1;
            Double_t newmax = zmin + (zmax-zmin)*ratio2;
            if(newmin < zmin)newmin = fH->GetBinContent(fH->GetMinimumBin());
            if(newmax > zmax)newmax = fH->GetBinContent(fH->GetMaximumBin());
            if(gPad->GetLogz()){
               newmin = TMath::Exp(2.302585092994*newmin);
               newmax = TMath::Exp(2.302585092994*newmax);
            }
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
char *TPaletteAxis::GetObjectInfo(Int_t /* px */, Int_t py) const
{
//   Redefines TObject::GetObjectInfo.
//   Displays the z value corresponding to cursor position py
// 
   Double_t z;
   static char info[64];

   Double_t zmin = fH->GetMinimum();
   Double_t zmax = fH->GetMaximum();
   Int_t   y1   = gPad->GetWh()-gPad->VtoPixel(fY1NDC);
   Int_t   y2   = gPad->GetWh()-gPad->VtoPixel(fY2NDC);
   Int_t   y    = gPad->GetWh()-py;

   if (gPad->GetLogz()) {
      if (zmin <= 0 && zmax > 0) zmin = TMath::Min((Double_t)1,
                                                   (Double_t)0.001*zmax);
      Double_t zminl = TMath::Log10(zmin);
      Double_t zmaxl = TMath::Log10(zmax);
      Double_t zl    = (zmaxl-zminl)*((Double_t)(y-y1)/(Double_t)(y2-y1))+zminl;
      z = TMath::Power(10.,zl);
   } else {
      z = (zmax-zmin)*((Double_t)(y-y1)/(Double_t)(y2-y1))+zmin;
   }

   sprintf(info,"(z=%g)",z);
   return info;
}


//______________________________________________________________________________
void TPaletteAxis::Paint(Option_t *)
{

   ConvertNDCtoPad();
   
   SetFillStyle(1001);
   Double_t ymin = fY1;
   Double_t ymax = fY2;
   Double_t xmin = fX1;
   Double_t xmax = fX2;
   Double_t wmin = fH->GetMinimum();
   Double_t wmax = fH->GetMaximum();
   Double_t wlmin = wmin;
   Double_t wlmax = wmax;
   Double_t y1,y2,w1,w2,zc;
   if (gPad->GetLogz()) {
      if (wmin <= 0 && wmax > 0) wmin = TMath::Min((Double_t)1,
                                                   (Double_t)0.001*wmax);
      wlmin = TMath::Log10(wmin);
      wlmax = TMath::Log10(wmax);
   }
   Double_t ws    = wlmax-wlmin;
   Int_t ncolors = gStyle->GetNumberOfColors();
   Int_t ndivz = TMath::Abs(fH->GetContour());
   Int_t theColor,color;
   Double_t scale = ndivz/(wlmax - wlmin);
   for (Int_t i=0;i<ndivz;i++) {

      zc = fH->GetContourLevel(i);
      if (fH->TestBit(TH1::kUserContour) && gPad->GetLogz())
         zc = TMath::Log10(zc);
      w1 = zc;
      if (w1 < wlmin) w1 = wlmin;

      w2 = wlmax;
      if (i < ndivz-1) {
         zc = fH->GetContourLevel(i+1);
         if (fH->TestBit(TH1::kUserContour) && gPad->GetLogz())
            zc = TMath::Log10(zc);
         w2 = zc;
      }

      if (w2 <= wlmin) continue;
      y1 = ymin + (w1-wlmin)*(ymax-ymin)/ws;
      y2 = ymin + (w2-wlmin)*(ymax-ymin)/ws;

      if (fH->TestBit(TH1::kUserContour)) {
         color = i;
      } else {
         color = Int_t(0.01+(w1-wlmin)*scale);
      }

      theColor = Int_t((color+0.99)*Double_t(ncolors)/Double_t(ndivz));
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
void TPaletteAxis::SavePrimitive(ofstream &out, Option_t *)
{
   // Save primitive as a C++ statement(s) on output stream out.

   //char quote = '"';
   out<<"   "<<endl;
   if (gROOT->ClassSaved(TPaletteAxis::Class())) {
       out<<"   ";
   } else {
       out<<"   "<<ClassName()<<" *";
   }
   if (fOption.Contains("NDC")) {
      out<<"palette = new "<<ClassName()<<"("<<fX1NDC<<","<<fY1NDC<<","<<fX2NDC<<","<<fY2NDC
      <<","<<fH->GetName()<<");"<<endl;
   } else {
      out<<"palette = new "<<ClassName()<<"("<<fX1<<","<<fY1<<","<<fX2<<","<<fY2
      <<","<<fH->GetName()<<");"<<endl;
   }
   out<<"palette->SetLabelColor(" <<fAxis.GetLabelColor()<<");"<<endl;
   out<<"palette->SetLabelFont("  <<fAxis.GetLabelFont()<<");"<<endl;
   out<<"palette->SetLabelOffset("<<fAxis.GetLabelOffset()<<");"<<endl;
   out<<"palette->SetLabelSize("  <<fAxis.GetLabelSize()<<");"<<endl;
   out<<"palette->SetTitleOffset("<<fAxis.GetTitleOffset()<<");"<<endl;
   out<<"palette->SetTitleSize("  <<fAxis.GetTitleSize()<<");"<<endl;
   SaveFillAttributes(out,"palette",-1,-1);
   SaveLineAttributes(out,"palette",1,1,1);
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
