// @(#)root/histpainter:$Id$
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
#include "TMath.h"
#include "TView.h"
#include "TH1.h"
#include "TGaxis.h"

ClassImp(TPaletteAxis)


//______________________________________________________________________________
/* Begin_Html
<center><h2>The palette painting class</h2></center>

A <tt>TPaletteAxis</tt> object is used to display the color palette when
drawing 2-d histograms.
<p>
The <tt>TPaletteAxis</tt> is automatically created drawn when drawing a 2-D
histogram when the option "Z" is specified.
<p>
A <tt>TPaletteAxis</tt> object is added to the histogram list of functions and
can be retrieved doing:
<pre>
   TPaletteAxis *palette = (TPaletteAxis*)h->GetListOfFunctions()->FindObject("palette");
</pre>
then the pointer <tt>palette</tt> can be used to change the pallette attributes.
<p>
Because the palette is created at painting time only, one must issue a:
<pre>
   gPad->Update();
</pre>
before retrieving the palette pointer in order to create the palette. The following
macro gives an example.

End_Html
Begin_Macro(source)
{
   TCanvas *c1 = new TCanvas("c1","c1",600,400);
   TH2F *h2 = new TH2F("h2","Example of a resized palette ",40,-4,4,40,-20,20);
   Float_t px, py;
   for (Int_t i = 0; i < 25000; i++) {
      gRandom->Rannor(px,py);
      h2->Fill(px,5*py);
   }
   gStyle->SetPalette(1);
   h2->Draw("COLZ");
   gPad->Update();
   TPaletteAxis *palette = (TPaletteAxis*)h2->GetListOfFunctions()->FindObject("palette");
   palette->SetY2NDC(0.7);
   return c1;
}
End_Macro
Begin_Html

<tt>TPaletteAxis</tt> inherits from <tt>TBox</tt> and <tt>TPave</tt>. The methods
allowing to specify the palette position are inherited from these two classes.
<p>
The palette can be interactively moved and resized. The context menu
can be used to set the axis attributes.
<p>
It is possible to select a range on the axis to set the min/max in z

End_Html */


//______________________________________________________________________________
TPaletteAxis::TPaletteAxis(): TPave()
{
   // Palette default constructor.

   fH  = 0;
   SetName("");
}


//______________________________________________________________________________
TPaletteAxis::TPaletteAxis(Double_t x1, Double_t y1,Double_t x2, Double_t  y2, TH1 *h)
       :TPave(x1,y1,x2,y2)
{
   // Palette normal constructor.

   fH = h;
   SetName("palette");
   TAxis *zaxis = fH->GetZaxis();
   fAxis.ImportAxisAttributes(zaxis);
   if (gPad->GetView()) SetBit(kHasView);
}


//______________________________________________________________________________
TPaletteAxis::~TPaletteAxis()
{
   // Palette destructor.

   if (fH) fH->GetListOfFunctions()->Remove(this);
}


//______________________________________________________________________________
TPaletteAxis::TPaletteAxis(const TPaletteAxis &palette) : TPave(palette)
{
   // Palette copy constructor.

   ((TPaletteAxis&)palette).Copy(*this);
}


//______________________________________________________________________________
void TPaletteAxis::Copy(TObject &obj) const
{
   // Copy a palette to a palette.

   TPave::Copy(obj);
   ((TPaletteAxis&)obj).fH    = fH;
   ((TPaletteAxis&)obj).fName = fName;
}


//______________________________________________________________________________
Int_t TPaletteAxis::DistancetoPrimitive(Int_t px, Int_t py)
{
   // Check if mouse on the axis region.

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
   // Check if mouse on the axis region.

   static Int_t kmode = 0;
   Int_t plxmin = gPad->XtoAbsPixel(fX1);
   Int_t plxmax = gPad->XtoAbsPixel(fX2);
   if (kmode != 0 || px <= plxmax) {
      if (event == kButton1Down) kmode = 1;
      TBox::ExecuteEvent(event,px,py);
      if (event == kButton1Up) kmode = 0;
      // In case palette coordinates have been modified, recompute NDC coordinates
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
      if (gROOT->IsEscaped()) {
         gROOT->SetEscape(kFALSE);
         break;
      }

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
Int_t TPaletteAxis::GetBinColor(Int_t i, Int_t j)
{
   // Returns the color index of the bin (i,j).
   //
   // This function should be used after an histogram has been plotted with the
   // option COL or COLZ like in the following example:
   //
   //   h2->Draw("COLZ");
   //   gPad->Update();
   //   TPaletteAxis *palette =
   //      (TPaletteAxis*)h2->GetListOfFunctions()->FindObject("palette");
   //   Int_t ci = palette->GetBinColor(20,15);
   //
   // Then it is possible to retrieve the RGB components in the following way:
   //
   //   TColor *c = gROOT->GetColor(ci);
   //   float x,y,z;
   //   c->GetRGB(x,y,z);

   Double_t zc = fH->GetBinContent(i,j);
   return GetValueColor(zc);
}


//______________________________________________________________________________
char *TPaletteAxis::GetObjectInfo(Int_t /* px */, Int_t py) const
{
   // Displays the z value corresponding to cursor position py.

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

   snprintf(info,64,"(z=%g)",z);
   return info;
}


//______________________________________________________________________________
Int_t TPaletteAxis::GetValueColor(Double_t zc)
{
   // Returns the color index of the given z value
   //
   // This function should be used after an histogram has been plotted with the
   // option COL or COLZ like in the following example:
   //
   //   h2->Draw("COLZ");
   //   gPad->Update();
   //   TPaletteAxis *palette =
   //      (TPaletteAxis*)h2->GetListOfFunctions()->FindObject("palette");
   //   Int_t ci = palette->GetValueColor(30.);
   //
   // Then it is possible to retrieve the RGB components in the following way:
   //
   //   TColor *c = gROOT->GetColor(ci);
   //   float x,y,z;
   //   c->GetRGB(x,y,z);

   Double_t wmin  = fH->GetMinimum();
   Double_t wmax  = fH->GetMaximum();
   Double_t wlmin = wmin;
   Double_t wlmax = wmax;

   if (gPad->GetLogz()) {
      if (wmin <= 0 && wmax > 0) wmin = TMath::Min((Double_t)1,
                                                   (Double_t)0.001*wmax);
      wlmin = TMath::Log10(wmin);
      wlmax = TMath::Log10(wmax);
   }

   Int_t ncolors = gStyle->GetNumberOfColors();
   Int_t ndivz   = TMath::Abs(fH->GetContour());
   Int_t theColor,color;
   Double_t scale = ndivz/(wlmax - wlmin);

   if (fH->TestBit(TH1::kUserContour) && gPad->GetLogz()) zc = TMath::Log10(zc);
   if (zc < wlmin) zc = wlmin;

   color = Int_t(0.01+(zc-wlmin)*scale);

   theColor = Int_t((color+0.99)*Double_t(ncolors)/Double_t(ndivz));
   return gStyle->GetColorPalette(theColor);
}


//______________________________________________________________________________
void TPaletteAxis::Paint(Option_t *)
{
   // Paint the palette.

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

   if ((wlmax - wlmin) <= 0) { 
      Double_t mz = wlmin*0.1;
      wlmin = wlmin-mz;
      wlmax = wlmax+mz;
      wmin = wlmin;
      wmax = wlmax;
   }

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
   char chopt[5] = "S   ";
   chopt[1] = 0;
   strncat(chopt, "+L", 2);
   if (ndiv < 0) {
      ndiv =TMath::Abs(ndiv);
      strncat(chopt, "N", 1);
   }
   if (gPad->GetLogz()) {
      wmin = TMath::Power(10.,wlmin);
      wmax = TMath::Power(10.,wlmax);
      strncat(chopt, "G", 1);
   }
   fAxis.PaintAxis(xmax,ymin,xmax,ymax,wmin,wmax,ndiv,chopt);
}


//______________________________________________________________________________
void TPaletteAxis::SavePrimitive(ostream &out, Option_t * /*= ""*/)
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
   // Unzoom the palette

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
