// @(#)root/graf:$Name:  $:$Id: TArrow.cxx,v 1.5 2002/01/24 11:39:28 rdm Exp $
// Author: Rene Brun   17/10/95

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "Riostream.h"
#include "TROOT.h"
#include "TArrow.h"
#include "TVirtualPad.h"

ClassImp(TArrow)

//______________________________________________________________________________
//  Different arrow formats as explained in TArrow::TArrow or in
//  the picture below are provided.
//  Once an arrow is drawn on the screen:
//    - One can click on one of the edges and move this edge.
//    - One can click on any other arrow part to move the entire arrow.
//Begin_Html
/*
<img src="gif/arrow.gif">
*/
//End_Html
//

//______________________________________________________________________________
TArrow::TArrow(): TLine(),TAttFill()
{
//*-*-*-*-*-*-*-*-*-*-*arrow default constructor*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  ========================

   fAngle = 60;
}

//______________________________________________________________________________
TArrow::TArrow(Double_t x1, Double_t y1,Double_t x2, Double_t  y2,
             Float_t arrowsize ,Option_t *option)
       :TLine(x1,y1,x2,y2), TAttFill(0,1001)
{
//*-*-*-*-*-*-*-*-*-*-*arrow normal constructor*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  ========================
// Define an arrow between points x1,y1 and x2,y2
// the arrowsize is in percentage of the pad height
// Opening angle between the two sides of the arrow is fAngle (60 degrees)
//  option = ">"      -------->
//  option = "<"      <--------
//  option = "->-"    ---->----
//  option = "-<-"    ----<----
//  option = "-|>-"   ---|>----
//  option = "<>"     <------->
//  option = "<|>"    <|-----|>  arrow defined by a triangle
//                   If FillColor == 0 draw open triangle
//                   else  draw full triangle with fillcolor
//

   fAngle       = 60;
   fArrowSize   = arrowsize;
   fOption      = option;
   SetLineColor(1);
   SetLineStyle(1);
}

//______________________________________________________________________________
TArrow::~TArrow()
{
//*-*-*-*-*-*-*-*-*-*-*arrow default destructor*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  ========================
}

//______________________________________________________________________________
TArrow::TArrow(const TArrow &arrow) : TLine(arrow), TAttFill(arrow)
{
   ((TArrow&)arrow).Copy(*this);
}

//______________________________________________________________________________
void TArrow::Copy(TObject &obj)
{
//*-*-*-*-*-*-*-*-*-*-*Copy this arrow to arrow*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  ========================

   TLine::Copy(obj);
   TAttFill::Copy(((TArrow&)obj));
   ((TArrow&)obj).fAngle      = fAngle;
   ((TArrow&)obj).fArrowSize  = fArrowSize;
   ((TArrow&)obj).fOption     = fOption;
}

//______________________________________________________________________________
void TArrow::Draw(Option_t *option)
{
//*-*-*-*-*-*-*-*-*-*-*Draw this arrow with its current attributes*-*-*-*-*-*-*
//*-*                  ===========================================

   Option_t *opt;
   if (strlen(option)) opt = option;
   else                opt = (char*)GetOption();

   AppendPad(opt);

}

//______________________________________________________________________________
void TArrow::DrawArrow(Double_t x1, Double_t y1,Double_t x2, Double_t  y2,
                     Float_t arrowsize ,Option_t *option)
{
//*-*-*-*-*-*-*-*-*-*-*Draw this arrow with new coordinates*-*-*-*-*-*-*-*-*-*
//*-*                  ====================================
   TArrow *newarrow = new TArrow(x1,y1,x2,y2,arrowsize,option);
   TAttLine::Copy(*newarrow);
   TAttFill::Copy(*newarrow);
   newarrow->AppendPad(option);
}

//______________________________________________________________________________
void TArrow::Paint(Option_t *option)
{
//*-*-*-*-*-*-*-*-*-*-*Paint this arrow with its current attributes*-*-*-*-*-*-*
//*-*                  ============================================

   PaintArrow(fX1, fY1, fX2, fY2, fArrowSize, option);
}


//______________________________________________________________________________
void TArrow::PaintArrow(Double_t x1, Double_t y1, Double_t x2, Double_t y2,
                        Float_t arrowsize, Option_t *option)
{
//*-*-*-*-*-*-*-*-*-*-*Draw this arrow with new coordinates*-*-*-*-*-*-*-*-*-*
//*-*                  ====================================
//
//                                               (P2)
//                                                -
//                                                .  -
//                                                .     -
//                                                .        -
//     -------------------------------------------.(P0)------- (P1)
//                                                .        -
//                                                .     -
//                                                .  -
//                                                -
//                                               (P3)
//
//

   TString opt = option;
   opt.ToLower();
   TAttLine::Modify();
   TAttFill::Modify();

//*-*- take u1,v1 as origin. take arrow line as new x axis
   Int_t px1    = gPad->XtoAbsPixel(x1);
   Int_t py1    = gPad->YtoAbsPixel(y1);
   Int_t px2    = gPad->XtoAbsPixel(x2);
   Int_t py2    = gPad->YtoAbsPixel(y2);
   Float_t lp   = TMath::Sqrt(Double_t((px2-px1)*(px2-px1) + (py2-py1)*(py2-py1)));
   Float_t rSiz = arrowsize*gPad->GetAbsHNDC()*gPad->GetWh();
   Float_t dSiz = rSiz*TMath::Tan(3.141592*fAngle/360);
   if (lp <= 0) return;
   Double_t ct = (px2-px1)/lp;
   Double_t st = (py1-py2)/lp;
   Int_t P2x,P2y,P3x,P3y,P0x,P0y;

// Otto start:  define default line  before move of origin of arrow
   Double_t XP0;
   Double_t YP0;
   Double_t XP0L;
   Double_t YP0L;
   XP0  = gPad->AbsPixeltoX(px2);
   YP0  = gPad->AbsPixeltoY(py2);
   XP0L = gPad->AbsPixeltoX(px1);
   YP0L = gPad->AbsPixeltoY(py1);
// move origin of arrow
   if (opt.Contains("-")) {
      px1 = Int_t(0.5 *(px2 + px1));
      py1 = Int_t(0.5 *(py2 + py1));
      px2 = px1;
      py2 = py1;
   }
// ----------
   if (opt.Contains(">")) {
      P2x = px2 - Int_t(rSiz*ct+st*dSiz-0.5);
      P2y = py2 + Int_t(rSiz*st-ct*dSiz+0.5);
      P3x = px2 - Int_t(rSiz*ct-st*dSiz-0.5);
      P3y = py2 + Int_t(rSiz*st+ct*dSiz+0.5);
      P0x = px2 - Int_t(rSiz*ct-0.5);
      P0y = py2 + Int_t(rSiz*st+0.5);
   } else {
      P2x = px2;
      P2y = py2;
      P3x = px2;
      P3y = py2;
      P0x = px2;
      P0y = py2;
   }
   Int_t P2xL,P2yL,P3xL,P3yL,P0xL,P0yL;

   if (opt.Contains("<")) {
      P2xL = px1 + Int_t(rSiz*ct-st*dSiz+0.5);
      P2yL = py1 - Int_t(rSiz*st+ct*dSiz-0.5);
      P3xL = px1 + Int_t(rSiz*ct+st*dSiz+0.5);
      P3yL = py1 - Int_t(rSiz*st-ct*dSiz-0.5);
      P0xL = px1 + Int_t(rSiz*ct+0.5);
      P0yL = py1 - Int_t(rSiz*st-0.5);
   } else {
      P2xL = px1;
      P2yL = py1;
      P3xL = px1;
      P3yL = py1;
      P0xL = px1;
      P0yL = py1;
   }
   Double_t XP2  = gPad->AbsPixeltoX(P2x);
   Double_t YP2  = gPad->AbsPixeltoY(P2y);
   Double_t XP3  = gPad->AbsPixeltoX(P3x);
   Double_t YP3  = gPad->AbsPixeltoY(P3y);
   Double_t XP2L = gPad->AbsPixeltoX(P2xL);
   Double_t YP2L = gPad->AbsPixeltoY(P2yL);
   Double_t XP3L = gPad->AbsPixeltoX(P3xL);
   Double_t YP3L = gPad->AbsPixeltoY(P3yL);
//   move up
   if (opt.Contains("|") && !opt.Contains("-")) {
      XP0  = gPad->AbsPixeltoX(P0x);
      YP0  = gPad->AbsPixeltoY(P0y);
      XP0L = gPad->AbsPixeltoX(P0xL);
      YP0L = gPad->AbsPixeltoY(P0yL);
   }

   gPad->PaintLine(XP0,YP0,XP0L,YP0L);

//*-*- Convert points to pad reference system
   Double_t xp1[4],yp1[4],xp2[4],yp2[4];

   xp1[0] = XP2;    yp1[0] = YP2;
   xp1[1] = gPad->AbsPixeltoX(px2);     yp1[1] = gPad->AbsPixeltoY(py2);
   xp1[2] = XP3;    yp1[2] = YP3;
   xp1[3] = XP2;    yp1[3] = YP2;

   xp2[0] = XP2L;   yp2[0] = YP2L;
   xp2[1] = gPad->AbsPixeltoX(px1);     yp2[1] = gPad->AbsPixeltoY(py1);
   xp2[2] = XP3L;   yp2[2] = YP3L;
   xp2[3] = XP2L;   yp2[3] = YP2L;
   if (opt.Contains(">")) {
      if (opt.Contains("|>")) {
         if (GetFillColor()) {
            gPad->PaintFillArea(3,xp1,yp1);
            gPad->PaintPolyLine(4,xp1,yp1);
         } else {
            gPad->PaintPolyLine(4,xp1,yp1);
         }
      } else {
         gPad->PaintPolyLine(3,xp1,yp1);
      }
   }
   if (opt.Contains("<")) {
      if (opt.Contains("<|")) {
         if (GetFillColor()) {
            gPad->PaintFillArea(3,xp2,yp2);
            gPad->PaintPolyLine(4,xp2,yp2);
         } else {
            gPad->PaintPolyLine(4,xp2,yp2);
         }
      } else {
         gPad->PaintPolyLine(3,xp2,yp2);
      }
   }

}

//______________________________________________________________________________
void TArrow::SavePrimitive(ofstream &out, Option_t *)
{
    // Save primitive as a C++ statement(s) on output stream out

   char quote = '"';
   if (gROOT->ClassSaved(TArrow::Class())) {
       out<<"   ";
   } else {
       out<<"   TArrow *";
   }
   out<<"arrow = new TArrow("<<fX1<<","<<fY1<<","<<fX2<<","<<fY2
      <<","<<fArrowSize<<","<<quote<<fOption<<quote<<");"<<endl;

   SaveFillAttributes(out,"arrow",0,1);
   SaveLineAttributes(out,"arrow",1,1,1);

   out<<"   arrow->Draw();"<<endl;
}
