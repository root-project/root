// @(#)root/graf:$Name:  $:$Id: TCrown.cxx,v 1.1 2002/08/09 20:47:17 brun Exp $
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
//  Example
//{
//   gROOT->Reset();
//   TCanvas c1("c1","c1",400,400);
//   TCrown cr1(.5,.5,.3,.4);
//   cr1->SetLineStyle(2);
//   cr1->SetLineWidth(4);
//   cr1.Draw();
//   TCrown cr2(.5,.5,.2,.3,45,315);
//   cr2.SetFillColor(38);
//   cr2.SetFillStyle(3010);
//   cr2.Draw();
//   TCrown cr3(.5,.5,.2,.3,-45,45);
//   cr3.SetFillColor(50);
//   cr3.SetFillStyle(3025);
//   cr3.Draw();
//   TCrown cr4(.5,.5,.0,.2);
//   cr4.SetFillColor(4);
//   cr4.SetFillStyle(3008);
//   cr4.Draw();
//}
//Begin_Html
/*
<img src="gif/crown.gif">
*/
//End_Html
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
// Compute distance from point px,py to a crown
//
// if crown is filled, return OK if we are inside
// otherwise, crown is found if near the crown edges

    const Double_t kPI = TMath::Pi();
    Double_t x = gPad->AbsPixeltoX(px) - GetX1();
    Double_t y = gPad->AbsPixeltoY(py) - GetY1();
    Double_t r = TMath::Sqrt(x*x+y*y);
   
    Int_t dist = 9999;
    if (r > fR2) return dist;
    if (r < fR1) return dist;
    if (fPhimax-fPhimin < 360) {
       Double_t phi = 180*TMath::ACos(x/r)/kPI;
       if (phi < fPhimin) return dist;
       if (phi > fPhimax) return dist;
    }

    if (GetFillColor() && GetFillStyle()) {
       return 0;
    } else {
       if (TMath::Abs(fR2-r)/fR2 < 0.02) return 0;
       if (TMath::Abs(fR1-r)/fR1 < 0.02) return 0;
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
//  For the time being TEllipse::ExecuteEvent is OK

   TEllipse::ExecuteEvent(event,px,py);
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
