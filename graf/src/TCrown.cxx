// @(#)root/graf:$Name:  $:$Id: TCrown.cxx,v 1.6 2002/05/18 08:21:59 brun Exp $
// Author: Rene Brun   108/08/2002

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "Riostream.h"
#include "TROOT.h"
#include "TCrown.h"
#include "TVirtualPad.h"

ClassImp(TCrown)

//______________________________________________________________________________
//
//  A crown is specified with the position of its centre, its inner/outer radius
//  a minimum and maximum angle.
//  The attributes of the outline line are given via TAttLine
//  The attributes of the fill area are given via TAttFill
//

//______________________________________________________________________________
TCrown::TCrown(): TEllipse()
{
//*-*-*-*-*-*-*-*-*-*-*crown  default constructor*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  ========================

}
//______________________________________________________________________________
TCrown::TCrown(Double_t x1, Double_t y1,Double_t radin, Double_t radout,Double_t phimin,Double_t phimax)
      :TEllipse(x1,y1,radin,radout,phimin,phimax,0)
{
//*-*-*-*-*-*-*-*-*-*-*Crown  normal constructor*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  =======================
//  x1,y1  : coordinates of centre of crown
//  radin  : inner crown radius
//  radout : outer crown radius
//  phimin : min and max angle in degrees (default is 0-->360)
//  phimax :
//
//  When a crown sector only is drawn, the lines connecting the center
//  of the crown to the edges are drawn by default. One can specify
//  the drawing option "only" to not draw these lines.
//
}

//______________________________________________________________________________
TCrown::TCrown(const TCrown &crown) : TEllipse(crown)
{
   ((TCrown&)crown).Copy(*this);
}

//______________________________________________________________________________
TCrown::~TCrown()
{
// crown  default destructor

}

//______________________________________________________________________________
void TCrown::Copy(TObject &crown)
{
// Copy this crown to crown

   TEllipse::Copy(crown);
}

//______________________________________________________________________________
Int_t TCrown::DistancetoPrimitive(Int_t px, Int_t py)
{
// Compute distance from point px,py to an ellipse
//
//  Compute the closest distance of approach from point px,py to this ellipse.
//  The distance is computed in pixels units.
//

    const Double_t kPI = TMath::Pi();
    Double_t x = gPad->AbsPixeltoX(px);
    Double_t y = gPad->AbsPixeltoY(py);

    Double_t dxnr = x - GetX1();
    Double_t dynr = y - GetY1();

    Double_t ct = TMath::Cos(kPI*GetTheta()/180.0);
    Double_t st = TMath::Sin(kPI*GetTheta()/180.0);

    Double_t dx =  dxnr*ct + dynr*st;
    Double_t dy = -dxnr*st + dynr*ct;

    Double_t r1 = GetR1();
    Double_t r2 = GetR2();

    if (dx == 0 || r1 == 0 || r2 == 0) return 9999;
    Double_t distp = TMath::Sqrt(dx*dx + dy*dy);

    Double_t tana = dy/dx;
    tana *= tana;
    Double_t distr = TMath::Sqrt((1+tana)/(1.0/(r1*r1) + tana/(r2*r2)));
    Int_t dist = 9999;
    if (GetFillColor() && GetFillStyle()) {
       if (distr > distp) dist = 0;
    } else {
       if (TMath::Abs(distr-distp)/(r1+r2) < 0.01) dist = 0;
    }
    return dist;
}

//______________________________________________________________________________
void TCrown::DrawCrown(Double_t x1, Double_t y1,Double_t radin,Double_t radout,Double_t phimin,Double_t phimax,Option_t *option)
{
// Draw this crown with new coordinates

   TCrown *newcrown = new TCrown(x1, y1, radin, radout, phimin, phimax);
   TAttLine::Copy(*newcrown);
   TAttFill::Copy(*newcrown);
   newcrown->AppendPad(option);
}

//______________________________________________________________________________
void TCrown::ExecuteEvent(Int_t event, Int_t px, Int_t py)
{
// Execute action corresponding to one event
//
//  This member function is called when a line is clicked with the locator
//
//  If Left button clicked on one of the line end points, this point
//     follows the cursor until button is released.
//
//  if Middle button clicked, the line is moved parallel to itself
//     until the button is released.
//


   const Double_t kPI = TMath::Pi();
   Int_t kMaxDiff = 10;
   const Int_t kMinSize = 25;
   const Int_t np = 40;
   static Int_t x[np+2], y[np+2];
   static Int_t px1,py1,npe,R1,R2,sav1,sav2;
   static Int_t pxold, pyold;
   static Int_t sig,impair;
   Int_t i, dpx, dpy;
   Double_t angle,dx,dy,dphi,ct,st,fTy,fBy,fLx,fRx;
   static Bool_t T, L, R, B, INSIDE;
   static Int_t Tx,Ty,Lx,Ly,Rx,Ry,Bx,By;

   if (!gPad->IsEditable()) return;

   switch (event) {

   case kButton1Down:
      gVirtualX->SetLineColor(-1);
      TAttLine::Modify();
      dphi = (fPhimax-fPhimin)*kPI/(180*np);
      ct   = TMath::Cos(kPI*fTheta/180);
      st   = TMath::Sin(kPI*fTheta/180);
      for (i=0;i<np;i++) {
         angle = fPhimin*kPI/180 + Double_t(i)*dphi;
         dx    = fR1*TMath::Cos(angle);
         dy    = fR2*TMath::Sin(angle);
         x[i]  = gPad->XtoAbsPixel(fX1 + dx*ct - dy*st);
         y[i]  = gPad->YtoAbsPixel(fY1 + dx*st + dy*ct);
      }
      if (fPhimax-fPhimin >= 360 ) {
         x[np] = x[0];
         y[np] = y[0];
         npe = np;
      } else {
         x[np]   = gPad->XtoAbsPixel(fX1);
         y[np]   = gPad->YtoAbsPixel(fY1);
         x[np+1] = x[0];
         y[np+1] = y[0];
         npe = np + 1;
      }
      impair = 0;
      px1 = gPad->XtoAbsPixel(fX1);
      py1 = gPad->YtoAbsPixel(fY1);
      Tx = Bx = px1;
      Ly = Ry = py1;
      Ty = gPad->YtoAbsPixel(fR2+fY1);
      By = gPad->YtoAbsPixel(-fR2+fY1);
      Lx = gPad->XtoAbsPixel(-fR1+fX1);
      Rx = gPad->XtoAbsPixel(fR1+fX1);
      R2 = (By-Ty)/2;
      R1 = (Rx-Lx)/2;
      gVirtualX->DrawLine(Rx+4, py1+4, Rx-4, py1+4);
      gVirtualX->DrawLine(Rx-4, py1+4, Rx-4, py1-4);
      gVirtualX->DrawLine(Rx-4, py1-4, Rx+4, py1-4);
      gVirtualX->DrawLine(Rx+4, py1-4, Rx+4, py1+4);
      gVirtualX->DrawLine(Lx+4, py1+4, Lx-4, py1+4);
      gVirtualX->DrawLine(Lx-4, py1+4, Lx-4, py1-4);
      gVirtualX->DrawLine(Lx-4, py1-4, Lx+4, py1-4);
      gVirtualX->DrawLine(Lx+4, py1-4, Lx+4, py1+4);
      gVirtualX->DrawLine(px1+4, By+4, px1-4, By+4);
      gVirtualX->DrawLine(px1-4, By+4, px1-4, By-4);
      gVirtualX->DrawLine(px1-4, By-4, px1+4, By-4);
      gVirtualX->DrawLine(px1+4, By-4, px1+4, By+4);
      gVirtualX->DrawLine(px1+4, Ty+4, px1-4, Ty+4);
      gVirtualX->DrawLine(px1-4, Ty+4, px1-4, Ty-4);
      gVirtualX->DrawLine(px1-4, Ty-4, px1+4, Ty-4);
      gVirtualX->DrawLine(px1+4, Ty-4, px1+4, Ty+4);
      // No break !!!

   case kMouseMotion:
      px1 = gPad->XtoAbsPixel(fX1);
      py1 = gPad->YtoAbsPixel(fY1);
      Tx = Bx = px1;
      Ly = Ry = py1;
      Ty = gPad->YtoAbsPixel(fR2+fY1);
      By = gPad->YtoAbsPixel(-fR2+fY1);
      Lx = gPad->XtoAbsPixel(-fR1+fX1);
      Rx = gPad->XtoAbsPixel(fR1+fX1);
      T = L = R = B = INSIDE = kFALSE;
      if ((TMath::Abs(px - Tx) < kMaxDiff) &&
          (TMath::Abs(py - Ty) < kMaxDiff)) {             // top edge
         T = kTRUE;
         gPad->SetCursor(kTopSide);
      }
      else
      if ((TMath::Abs(px - Bx) < kMaxDiff) &&
          (TMath::Abs(py - By) < kMaxDiff)) {             // bottom edge
         B = kTRUE;
         gPad->SetCursor(kBottomSide);
      }
      else
      if ((TMath::Abs(py - Ly) < kMaxDiff) &&
          (TMath::Abs(px - Lx) < kMaxDiff)) {             // left edge
         L = kTRUE;
         gPad->SetCursor(kLeftSide);
      }
      else
      if ((TMath::Abs(py - Ry) < kMaxDiff) &&
          (TMath::Abs(px - Rx) < kMaxDiff)) {             // right edge
         R = kTRUE;
         gPad->SetCursor(kRightSide);
      }
      else {INSIDE= kTRUE; gPad->SetCursor(kMove); }
      pxold = px;  pyold = py;

      break;

   case kButton1Motion:
      gVirtualX->DrawLine(Rx+4, py1+4, Rx-4, py1+4);
      gVirtualX->DrawLine(Rx-4, py1+4, Rx-4, py1-4);
      gVirtualX->DrawLine(Rx-4, py1-4, Rx+4, py1-4);
      gVirtualX->DrawLine(Rx+4, py1-4, Rx+4, py1+4);
      gVirtualX->DrawLine(Lx+4, py1+4, Lx-4, py1+4);
      gVirtualX->DrawLine(Lx-4, py1+4, Lx-4, py1-4);
      gVirtualX->DrawLine(Lx-4, py1-4, Lx+4, py1-4);
      gVirtualX->DrawLine(Lx+4, py1-4, Lx+4, py1+4);
      gVirtualX->DrawLine(px1+4, By+4, px1-4, By+4);
      gVirtualX->DrawLine(px1-4, By+4, px1-4, By-4);
      gVirtualX->DrawLine(px1-4, By-4, px1+4, By-4);
      gVirtualX->DrawLine(px1+4, By-4, px1+4, By+4);
      gVirtualX->DrawLine(px1+4, Ty+4, px1-4, Ty+4);
      gVirtualX->DrawLine(px1-4, Ty+4, px1-4, Ty-4);
      gVirtualX->DrawLine(px1-4, Ty-4, px1+4, Ty-4);
      gVirtualX->DrawLine(px1+4, Ty-4, px1+4, Ty+4);
      for (i=0;i<npe;i++) gVirtualX->DrawLine(x[i], y[i], x[i+1], y[i+1]);
      if (T) {
         sav1 = py1;
         sav2 = R2;
         py1 += (py - pyold)/2;
         R2 -= (py - pyold)/2;
         if (TMath::Abs(pyold-py)%2==1) impair++;
         if (py-pyold>0) sig=+1;
         else sig=-1;
         if (impair==2) { impair = 0; py1 += sig; R2 -= sig;}
         if (py1 > By-kMinSize) {py1 = sav1; R2 = sav2; py = pyold;}
      }
      if (B) {
         sav1 = py1;
         sav2 = R2;
         py1 += (py - pyold)/2;
         R2 += (py - pyold)/2;
         if (TMath::Abs(pyold-py)%2==1) impair++;
         if (py-pyold>0) sig=+1;
         else sig=-1;
         if (impair==2) { impair = 0; py1 += sig; R2 += sig;}
         if (py1 < Ty+kMinSize) {py1 = sav1; R2 = sav2; py = pyold;}
      }
      if (L) {
         sav1 = px1;
         sav2 = R1;
         px1 += (px - pxold)/2;
         R1 -= (px - pxold)/2;
         if (TMath::Abs(pxold-px)%2==1) impair++;
         if (px-pxold>0) sig=+1;
         else sig=-1;
         if (impair==2) { impair = 0; px1 += sig; R1 -= sig;}
         if (px1 > Rx-kMinSize) {px1 = sav1; R1 = sav2; px = pxold;}
      }
      if (R) {
         sav1 = px1;
         sav2 = R1;
         px1 += (px - pxold)/2;
         R1 += (px - pxold)/2;
         if (TMath::Abs(pxold-px)%2==1) impair++;
         if (px-pxold>0) sig=+1;
         else sig=-1;
         if (impair==2) { impair = 0; px1 += sig; R1 += sig;}
         if (px1 < Lx+kMinSize) {px1 = sav1; R1 = sav2; px = pxold;}
      }
      if (T || B || L || R) {
         gVirtualX->SetLineColor(-1);
         TAttLine::Modify();
         dphi = (fPhimax-fPhimin)*kPI/(180*np);
         ct   = TMath::Cos(kPI*fTheta/180);
         st   = TMath::Sin(kPI*fTheta/180);
         for (i=0;i<np;i++) {
            angle = fPhimin*kPI/180 + Double_t(i)*dphi;
            dx    = R1*TMath::Cos(angle);
            dy    = R2*TMath::Sin(angle);
            x[i]  = px1 + Int_t(dx*ct - dy*st);
            y[i]  = py1 + Int_t(dx*st + dy*ct);
         }
         if (fPhimax-fPhimin >= 360 ) {
            x[np] = x[0];
            y[np] = y[0];
            npe = np;
         } else {
            x[np]   = px1;
            y[np]   = py1;
            x[np+1] = x[0];
            y[np+1] = y[0];
            npe = np + 1;
         }
         for (i=0;i<npe;i++) gVirtualX->DrawLine(x[i], y[i], x[i+1], y[i+1]);
      }
      if (INSIDE) {
          dpx  = px-pxold;  dpy = py-pyold;
          px1 += dpx; py1 += dpy;
          for (i=0;i<=npe;i++) { x[i] += dpx; y[i] += dpy;}
          for (i=0;i<npe;i++) gVirtualX->DrawLine(x[i], y[i], x[i+1], y[i+1]);
      }
      Tx = Bx = px1;
      Rx = px1+R1;
      Lx = px1-R1;
      Ry = Ly = py1;
      Ty = py1-R2;
      By = py1+R2;
      gVirtualX->DrawLine(Rx+4, py1+4, Rx-4, py1+4);
      gVirtualX->DrawLine(Rx-4, py1+4, Rx-4, py1-4);
      gVirtualX->DrawLine(Rx-4, py1-4, Rx+4, py1-4);
      gVirtualX->DrawLine(Rx+4, py1-4, Rx+4, py1+4);
      gVirtualX->DrawLine(Lx+4, py1+4, Lx-4, py1+4);
      gVirtualX->DrawLine(Lx-4, py1+4, Lx-4, py1-4);
      gVirtualX->DrawLine(Lx-4, py1-4, Lx+4, py1-4);
      gVirtualX->DrawLine(Lx+4, py1-4, Lx+4, py1+4);
      gVirtualX->DrawLine(px1+4, By+4, px1-4, By+4);
      gVirtualX->DrawLine(px1-4, By+4, px1-4, By-4);
      gVirtualX->DrawLine(px1-4, By-4, px1+4, By-4);
      gVirtualX->DrawLine(px1+4, By-4, px1+4, By+4);
      gVirtualX->DrawLine(px1+4, Ty+4, px1-4, Ty+4);
      gVirtualX->DrawLine(px1-4, Ty+4, px1-4, Ty-4);
      gVirtualX->DrawLine(px1-4, Ty-4, px1+4, Ty-4);
      gVirtualX->DrawLine(px1+4, Ty-4, px1+4, Ty+4);
      pxold = px;
      pyold = py;
      break;

   case kButton1Up:
      fX1 = gPad->AbsPixeltoX(px1);
      fY1 = gPad->AbsPixeltoY(py1);
      fBy = gPad->AbsPixeltoY(py1+R2);
      fTy = gPad->AbsPixeltoY(py1-R2);
      fLx = gPad->AbsPixeltoX(px1+R1);
      fRx = gPad->AbsPixeltoX(px1-R1);
      fR1 = TMath::Abs(fRx-fLx)/2;
      fR2 = TMath::Abs(fTy-fBy)/2;
      gPad->Modified(kTRUE);
      gVirtualX->SetLineColor(-1);
   }
}

//______________________________________________________________________________
void TCrown::Paint(Option_t *)
{
// Paint this crown with its current attributes

   const Double_t kPI = TMath::Pi();
   const Int_t np = 40;
   static Double_t x[2*np+3], y[2*np+3];
   TAttLine::Modify();
   TAttFill::Modify();

   Double_t angle,dx,dy;
   Double_t dphi = (fPhimax-fPhimin)*kPI/(180*np);
   Double_t ct   = TMath::Cos(kPI*fTheta/180);
   Double_t st   = TMath::Sin(kPI*fTheta/180);
   Int_t i;
   //compute outer points
   for (i=0;i<=np;i++) {
      angle = fPhimin*kPI/180 + Double_t(i)*dphi;
      dx    = fR2*TMath::Cos(angle);
      dy    = fR2*TMath::Sin(angle);
      x[i]  = fX1 + dx*ct - dy*st;
      y[i]  = fY1 + dx*st + dy*ct;
   }
   //compute inner points
   for (i=0;i<=np;i++) {
      angle = fPhimin*kPI/180 + Double_t(i)*dphi;
      dx    = fR1*TMath::Cos(angle);
      dy    = fR1*TMath::Sin(angle);
      x[2*np-i+1]  = fX1 + dx*ct - dy*st;
      y[2*np-i+1]  = fY1 + dx*st + dy*ct;
   }
   x[2*np+2]  = x[0];
   y[2*np+2]  = y[0];
   if (fPhimax-fPhimin >= 360 ) {
      // a complete filled crown
      if (GetFillColor()  && GetFillStyle()) {
         gPad->PaintFillArea(2*np+2,x,y); 
      }
      // a complete empty crown
      if (GetLineStyle()) {
         gPad->PaintPolyLine(np+1,x,y);
         gPad->PaintPolyLine(np+1,&x[np+1],&y[np+1]);
      }
  } else {
      //crown segment
     if (GetFillColor()  && GetFillStyle()) gPad->PaintFillArea(2*np+2,x,y);
      if (GetLineStyle()) gPad->PaintPolyLine(2*np+3,x,y);
   }
}

//______________________________________________________________________________
void TCrown::SavePrimitive(ofstream &out, Option_t *)
{
   // Save primitive as a C++ statement(s) on output stream out

   out<<"   "<<endl;
   if (gROOT->ClassSaved(TCrown::Class())) {
       out<<"   ";
   } else {
       out<<"   TCrown *";
   }
   out<<"crown = new TCrown("<<fX1<<","<<fY1<<","<<fR1<<","<<fR2
      <<","<<fPhimin<<","<<fPhimax<<");"<<endl;

   SaveFillAttributes(out,"crown",0,1001);
   SaveLineAttributes(out,"crown",1,1,1);

   out<<"   crown->Draw();"<<endl;
}
