// @(#)root/graf:$Name:  $:$Id: TText.cxx,v 1.5 2000/11/21 20:28:13 brun Exp $
// Author: Nicolas Brun   12/12/94

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include <fstream.h>
#include <iostream.h>

#include "TROOT.h"
#include "TVirtualPad.h"
#include "TText.h"
#include "TVirtualX.h"
#include "TMath.h"


ClassImp(TText)

//______________________________________________________________________________
//
//   TText is the base class for several text objects.
//   See TAttText for a list of text attributes or fonts,
//   and also for a discussion on text spped and font quality.
//
//  By default, the text is drawn in the pad coordinates system.
//  One can draw in NDC coordinates [0,1] if the function SetNDC
//  is called for a TText object.
//

//______________________________________________________________________________
TText::TText(): TNamed(), TAttText()
{
//*-*-*-*-*-*-*-*-*-*-*Text default constructor*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  ========================
}
//______________________________________________________________________________
TText::TText(Double_t x, Double_t y, const char *text) : TNamed("",text), TAttText()
{
//*-*-*-*-*-*-*-*-*-*-*Text normal constructor*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  =======================
   fX=x; fY=y;
}

//______________________________________________________________________________
TText::~TText()
{
//*-*-*-*-*-*-*-*-*-*-*Text default destructor*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  =======================
}

//______________________________________________________________________________
TText::TText(const TText &text)
{
   ((TText&)text).Copy(*this);
}

//______________________________________________________________________________
void TText::Copy(TObject &obj)
{
//*-*-*-*-*-*-*-*-*-*-*Copy this text to text*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
//*-*                  ======================

   ((TText&)obj).fX = fX;
   ((TText&)obj).fY = fY;
   TNamed::Copy(obj);
   TAttText::Copy(((TText&)obj));
}

//______________________________________________________________________________
Int_t TText::DistancetoPrimitive(Int_t px, Int_t py)
{
//*-*-*-*-*-*-*-*-*-*-*Compute distance from point px,py to a string*-*-*-*-*
//*-*                  =============================================
//  Compute the closest distance of approach from point px,py to this text.
//  The rectangle surrounding this string is evaluated.
//  if point is in rectangle, the distance is set to zero.
//
   Int_t Ax,Ay,Bx,By,Cx,Cy,Dx,Dy,pxl,pxt,pyl,pyt;
   Int_t ptx, pty;
   Ax = Ay = pxl = pxt = pyl = pyt = 0;
   if (TestBit(kTextNDC)) {
      ptx   = gPad->UtoPixel(fX);
      pty   = gPad->VtoPixel(fY);
   } else {
      ptx    = gPad->XtoAbsPixel(fX);
      pty    = gPad->YtoAbsPixel(fY);
   }
   const char *text = GetTitle();
   Int_t len    = strlen(text);
   Double_t fh  = (fTextSize*gPad->GetAbsHNDC())*Double_t(gPad->GetWh());
   Int_t h      = Int_t(fh/2);
   Int_t w      = h*len;
   Int_t err = 1;
   Int_t X1c,X2c,X3c,X4c;
   Short_t halign = fTextAlign/10;
   Short_t valign = fTextAlign - 10*halign;
   if (fTextAngle < err || TMath::Abs(fTextAngle-180) < err || fTextAngle>360-err) {
     Int_t signe = 1;
     if (TMath::Abs(fTextAngle-180) < err) signe = -1;
     switch (halign) {  //*-* compute bounding box horizontal alignment
        case 1 : pxl = ptx;              pxt = ptx + signe*w; break;     //left
        case 2 : pxl = ptx - signe*w/2;  pxt = ptx + signe*w/2; break;   //center
        case 3 : pxl = ptx - signe*w;    pxt = ptx; break;         //right
     };
     switch (valign) {  //*-* compute bounding box vertical alignment
        case 1 : pyl = pty - signe*3/2*h;    pyt = pty; break;         //bottom
        case 2 : pyl = pty - signe*h/2;      pyt = pty + signe*h/2; break;   //center
        case 3 : pyl = pty + signe*h/2;      pyt = pty + signe*3*h/2; break; //top
     };
     if (fTextAngle <err || fTextAngle>360-err) if ((px >= pxl && px <= pxt) && (py >= pyl && py <= pyt)) return 0;
     if (TMath::Abs(fTextAngle-180)<2) if ((px <= pxl && px >= pxt) && (py <= pyl && py >= pyt)) return 0;
     return 9999;
   }
   if (TMath::Abs(fTextAngle-90) < err || TMath::Abs(fTextAngle-270)<err) {
     Int_t signe = 1;
     if (TMath::Abs(fTextAngle-270)<err) signe = -1;
     switch (halign) {  //*-* compute bounding box horizontal alignment
        case 1 : pyt = pty;              pyl = pty - signe*w; break;     //left
        case 2 : pyl = pty - signe*w/2;  pyt = pty + signe*w/2; break;   //center
        case 3 : pyt = pty + signe*w;    pyl = pty; break;         //right
     };
     switch (valign) {  //*-* compute bounding box vertical alignment
        case 1 : pxl = ptx - signe*3/2*h;    pxt = ptx; break;         //bottom
        case 2 : pxl = ptx - signe*h/2;      pxt = ptx + signe*h/2; break;   //center
        case 3 : pxl = ptx + signe*h/2;      pxt = ptx + signe*3*h/2; break; //top
     };
     if (TMath::Abs(fTextAngle-90)<err)  if ((px >= pxl && px <= pxt) && (py >= pyl && py <= pyt)) return 0;
     if (TMath::Abs(fTextAngle-270)<err) if ((px <= pxl && px >= pxt) && (py <= pyl && py >= pyt)) return 0;
     return 9999;
   }
   Double_t co = TMath::Cos(fTextAngle*0.0175);
   Double_t si = TMath::Sin(fTextAngle*0.0175);
   if (halign == 1) {
      switch (valign) {
         case 1 : Ax = ptx;                         Ay = pty; break;
         case 2 : Ax = ptx + Int_t((si*h/2+0.5));   Ay = pty + Int_t((co*h/2+0.5)); break;
         case 3 : Ax = ptx + Int_t((si*h*3/2+0.5)); Ay = pty + Int_t((co*h*3/2+0.5)); break;
      }
   }
   if (halign == 2) {
      switch (valign) {
         case 1 : Ax = ptx - Int_t((co*w/2+0.5));          Ay = pty + Int_t((si*w/2+0.5)); break;
         case 2 : Ax = ptx - Int_t((co*w/2+si*h/2+0.5));   Ay = pty + Int_t((si*w/2+co*h/2+0.5)); break;
         case 3 : Ax = ptx - Int_t((co*w/2+si*h*3/2+0.5)); Ay = pty + Int_t((si*w/2+co*h*3/2+0.5)); break;
      }
   }
   if (halign == 3) {
      switch (valign) {
         case 1 : Ax = ptx - Int_t((co*w+0.5));          Ay = pty + Int_t((si*w+0.5)); break;
         case 2 : Ax = ptx - Int_t((co*w+si*h/2+0.5));   Ay = pty + Int_t((si*w+co*h/2+0.5)); break;
         case 3 : Ax = ptx - Int_t((co*w+si*h*3/2+0.5)); Ay = pty + Int_t((si*w+co*h*3/2+0.5)); break;
      }
   }
   Bx = Ax - Int_t((si*h+0.5)); By = Ay - Int_t((co*h+0.5));
   Cx = Bx + Int_t((co*w+0.5)); Cy = By - Int_t((si*w+0.5));
   Dx = Ax + Int_t((co*w+0.5)); Dy = Ay - Int_t((si*w+0.5));
   if (Cy==By) X1c = px;
   else X1c = (py-Cy)*(Cx-Bx)/(Cy-By)+Cx;
   if (Dy==Cy) X2c = px;
   else X2c = (py-Cy)*(Dx-Cx)/(Dy-Cy)+Cx;
   if (Dy==Ay) X3c = px;
   else X3c = (py-Dy)*(Dx-Ax)/(Dy-Ay)+Dx;
   if (Ay==By) X4c = px;
   else X4c = (py-Ay)*(Ax-Bx)/(Ay-By)+Ax;
   if (fTextAngle<90)  if (px >= X1c && px <= X2c && px >= X4c && px <= X3c) return 0;
   if (fTextAngle<180) if (px >= X1c && px >= X2c && px <= X4c && px <= X3c) return 0;
   if (fTextAngle<270) if (px <= X1c && px >= X2c && px <= X4c && px >= X3c) return 0;
   if (fTextAngle<360) if (px <= X1c && px <= X2c && px >= X4c && px >= X3c) return 0;
   return 9999;
}

//______________________________________________________________________________
TText *TText::DrawText(Double_t x, Double_t y, const char *text)
{
//*-*-*-*-*-*-*-*-*-*-*Draw this text with new coordinates*-*-*-*-*-*-*-*-*-*
//*-*                  ===================================
   TText *newtext = new TText(x, y, text);
   TAttText::Copy(*newtext);
   newtext->SetBit(kCanDelete);
   if (TestBit(kTextNDC)) newtext->SetNDC();
   newtext->AppendPad();
   return newtext;
}

//______________________________________________________________________________
TText *TText::DrawTextNDC(Double_t x, Double_t y, const char *text)
{
//*-*-*-*-*-*-*-*-*-*-*Draw this text with new coordinates in NDC*-*-*-*-*-*
//*-*                  ==========================================
   TText *newtext = DrawText(x, y, text);
   newtext->SetNDC();
   return newtext;
}

//______________________________________________________________________________
void TText::ExecuteEvent(Int_t event, Int_t px, Int_t py)
{
//*-*-*-*-*-*-*-*-*-*-*Execute action corresponding to one event*-*-*-*
//*-*                  =========================================
//  This member function must be implemented to realize the action
//  corresponding to the mouse click on the object in the window
//

   static Int_t px1, py1, pxold, pyold, Size, hauteur, largeur;
   static Bool_t resize,turn;
   Int_t dx, dy;
   Int_t kMaxDiff = 8;
   const char *text = GetTitle();
   Int_t len    = strlen(text);
   Double_t sizetowin = gPad->GetAbsHNDC()*Double_t(gPad->GetWh());
   Double_t fh  = (fTextSize*sizetowin);
   Int_t h      = Int_t(fh/2);
   Int_t w      = h*len;
   Short_t halign = fTextAlign/10;
   Short_t valign = fTextAlign - 10*halign;
   Double_t co,si,dtheta,norm;
   static Bool_t droite;
   static Double_t theta;
   Int_t Ax,Ay,Hx,Hy,Bx,By,Cx,Cy;
   Ax = Ay = 0;
   Double_t lambda, x2,y2,xy;
   Double_t dpx,dpy,xp1,yp1;

   if (!gPad->IsEditable()) return;

   switch (event) {

   case kButton1Down:
      gVirtualX->SetTextColor(-1);  // invalidate current text color (use xor mode)
      TAttText::Modify();  //*-*Change text attributes only if necessary

      // No break !!!

   case kMouseMotion:
      if (TestBit(kTextNDC)) {
         px1   = gPad->UtoPixel(fX);
         py1   = gPad->VtoPixel(fY);
      } else {
         px1   = gPad->XtoAbsPixel(fX);
         py1   = gPad->YtoAbsPixel(fY);
      }
      theta = fTextAngle;
      Size = 0;
      pxold = px;  pyold = py;
      co = TMath::Cos(fTextAngle*0.0175);
      si = TMath::Sin(fTextAngle*0.0175);
      resize = turn = kFALSE;
      if (halign == 1) {
         switch (valign) {
            case 1 : Ax = px1; Ay = py1; break;
            case 2 : Ax = px1+Int_t(si*h/2); Ay = py1+Int_t(co*h/2); break;
            case 3 : Ax = px1+Int_t(si*h*3/2); Ay = py1+Int_t(co*h*3/2); break;
         }
      }
      if (halign == 2) {
         switch (valign) {
            case 1 : Ax = px1-Int_t(co*w/2); Ay = py1+Int_t(si*w/2); break;
            case 2 : Ax = px1-Int_t(co*w/2+si*h/2); Ay = py1+Int_t(si*w/2+co*h/2); break;
            case 3 : Ax = px1-Int_t(co*w/2+si*h*3/2); Ay = py1+Int_t(si*w/2+co*h*3/2); break;
         }
      }
      if (halign == 3) {
         switch (valign) {
            case 1 : Ax = px1-Int_t(co*w); Ay = py1+Int_t(si*w); break;
            case 2 : Ax = px1-Int_t(co*w+si*h/2); Ay = py1+Int_t(si*w+co*h/2); break;
            case 3 : Ax = px1-Int_t(co*w+si*h*3/2); Ay = py1+Int_t(si*w+co*h*3/2); break;
         }
      }
      if (halign != 3) {Hx = Ax+Int_t(co*w-si*h/2); Hy = Ay-Int_t(si*w-co*h/2); droite = kFALSE;}
      else {Hx = Ax-Int_t(si*h/2); Hy = Ay-Int_t(co*h/2); droite = kTRUE;}
      if ((TMath::Abs(px-Hx)<kMaxDiff*2) && (TMath::Abs(py-Hy)<kMaxDiff*2)) {gPad->SetCursor(kRotate); turn = kTRUE; break;}
      Bx = Ax-Int_t(si*h); By = Ay-Int_t(co*h);
      if (valign == 3) {Bx = Ax; By = Ay;}
      Cx = Bx+Int_t(co*w); Cy = By-Int_t(si*w);
      lambda = Double_t(((px-Bx)*(Cx-Bx)+(py-By)*(Cy-By)))/Double_t(((Cx-Bx)*(Cx-Bx)+(Cy-By)*(Cy-By)));
      x2 = Double_t(px) - lambda*Double_t(Cx-Bx)-Double_t(Bx);
      y2 = Double_t(py) - lambda*Double_t(Cy-By)-Double_t(By);
      xy = Double_t((px-Ax)*(px-Ax)+(py-Ay)*(py-Ay));
      if ((TMath::Sqrt(x2*x2+y2*y2) < kMaxDiff/2) && (TMath::Sqrt(xy) <= w*0.3)) {
         if (fTextAngle == 0 || fTextAngle == 180) gPad->SetCursor(kArrowVer);
         else {
            if (fTextAngle == 90 || fTextAngle == 270) gPad->SetCursor(kArrowHor);
            else gPad->SetCursor(kHand);
         }
         hauteur = valign;
         largeur = halign;
         resize = kTRUE;
         break;
      }
      gPad->SetCursor(kMove);
      break;

   case kButton1Motion:
      gVirtualX->DrawText(px1, py1, theta, gVirtualX->GetTextMagnitude(), GetTitle(), TVirtualX::kClear);
      if (turn) {
         norm = TMath::Sqrt(Double_t((py-py1)*(py-py1)+(px-px1)*(px-px1)));
         if (norm>0) {
            theta = TMath::ACos((px-px1)/norm);
            dtheta= TMath::ASin((py1-py)/norm);
            if (dtheta<0) theta = -theta;
            theta = theta/TMath::ACos(-1)*180;
            if (theta<0) theta += 360;
            if (droite) {theta = theta+180; if (theta>=360) theta -= 360;}
         }
      }
      else if (resize) {

         co = TMath::Cos(fTextAngle*0.0175);
         si = TMath::Sin(fTextAngle*0.0175);
         if (largeur == 1) {
            switch (valign) {
               case 1 : Ax = px1; Ay = py1; break;
               case 2 : Ax = px1+Int_t(si*h/2); Ay = py1+Int_t(co*h/2); break;
               case 3 : Ax = px1+Int_t(si*h*3/2); Ay = py1+Int_t(co*h*3/2); break;
            }
         }
         if (largeur == 2) {
            switch (valign) {
               case 1 : Ax = px1-Int_t(co*w/2); Ay = py1+Int_t(si*w/2); break;
               case 2 : Ax = px1-Int_t(co*w/2+si*h/2); Ay = py1+Int_t(si*w/2+co*h/2); break;
               case 3 : Ax = px1-Int_t(co*w/2+si*h*3/2); Ay = py1+Int_t(si*w/2+co*h*3/2); break;
            }
         }
         if (largeur == 3) {
            switch (valign) {
               case 1 : Ax = px1-Int_t(co*w); Ay = py1+Int_t(si*w); break;
               case 2 : Ax = px1-Int_t(co*w+si*h/2); Ay = py1+Int_t(si*w+co*h/2); break;
               case 3 : Ax = px1-Int_t(co*w+si*h*3/2); Ay = py1+Int_t(si*w+co*h*3/2); break;
            }
         }
         if (hauteur == 3) {Bx = Ax-Int_t(si*h); By = Ay-Int_t(co*h);}
         else {Bx = Ax; By = Ay;}
         Cx = Bx+Int_t(co*w); Cy = By-Int_t(si*w);
         lambda = Double_t(((px-Bx)*(Cx-Bx)+(py-By)*(Cy-By)))/Double_t(((Cx-Bx)*(Cx-Bx)+(Cy-By)*(Cy-By)));
         x2 = Double_t(px) - lambda*Double_t(Cx-Bx)-Double_t(Bx);
         y2 = Double_t(py) - lambda*Double_t(Cy-By)-Double_t(By);
         Size = Int_t(TMath::Sqrt(x2*x2+y2*y2)*2);
         if (Size<4) Size = 4;

         SetTextSize(Size/sizetowin);
         TAttText::Modify();
      }
      else {
         dx = px - pxold;  px1 += dx;
         dy = py - pyold;  py1 += dy;
      }
      gVirtualX->DrawText(px1, py1, theta, gVirtualX->GetTextMagnitude(), GetTitle(), TVirtualX::kClear);
      pxold = px;  pyold = py;
      break;

   case kButton1Up:
      if (TestBit(kTextNDC)) {
         dpx  = gPad->GetX2() - gPad->GetX1();
         dpy  = gPad->GetY2() - gPad->GetY1();
         xp1  = gPad->GetX1();
         yp1  = gPad->GetY1();
         fX = (gPad->AbsPixeltoX(px1)-xp1)/dpx;
         fY = (gPad->AbsPixeltoY(py1)-yp1)/dpy;
      } else {
         fX = gPad->AbsPixeltoX(px1);
         fY = gPad->AbsPixeltoY(py1);
      }
      fTextAngle = theta;
      gPad->Modified(kTRUE);
      gVirtualX->SetTextColor(-1);
      break;

   case kButton1Locate:

      ExecuteEvent(kButton1Down, px, py);

      while (1) {
         px = py = 0;
         event = gVirtualX->RequestLocator(1, 1, px, py);

         ExecuteEvent(kButton1Motion, px, py);

         if (event != -1) {                     // button is released
            ExecuteEvent(kButton1Up, px, py);
            return;
         }
      }
   }
}

//______________________________________________________________________________
void TText::ls(Option_t *) const
{
//*-*-*-*-*-*-*-*-*-*-*-*List this text with its attributes*-*-*-*-*-*-*-*-*
//*-*                    ==================================
   TROOT::IndentLevel();
   printf("Text  X=%f Y=%f Text=%s\n",fX,fY,GetTitle());
}

//______________________________________________________________________________
void TText::Paint(Option_t *)
{
//*-*-*-*-*-*-*-*-*-*-*Paint this text with its current attributes*-*-*-*-*-*-*
//*-*                  ===========================================

   TAttText::Modify();  //Change text attributes only if necessary
   if (TestBit(kTextNDC)) gPad->PaintTextNDC(fX,fY,GetTitle());
   else                   gPad->PaintText(fX,fY,GetTitle());
}

//______________________________________________________________________________
void TText::PaintText(Double_t x, Double_t y, const char *text)
{
//*-*-*-*-*-*-*-*-*-*-*Draw this text with new coordinates*-*-*-*-*-*-*-*-*-*
//*-*                  ===================================

   TAttText::Modify();  //Change text attributes only if necessary
   gPad->PaintText(x,y,text);

}

//______________________________________________________________________________
void TText::PaintTextNDC(Double_t u, Double_t v, const char *text)
{
//*-*-*-*-*-*-*-*-*-*-*Draw this text with new coordinates in NDC*-*-*-*-*-*-*
//*-*                  ==========================================

   TAttText::Modify();  //Change text attributes only if necessary
   gPad->PaintTextNDC(u,v,text);

}

//______________________________________________________________________________
void TText::Print(Option_t *) const
{
//*-*-*-*-*-*-*-*-*-*-*Dump this text with its attributes*-*-*-*-*-*-*-*-*-*
//*-*                  ==================================

   printf("Text  X=%f Y=%f Text=%s Font=%d Size=%f",fX,fY,GetTitle(),GetTextFont(),GetTextSize());
   if (GetTextColor() != 1 ) printf(" Color=%d",GetTextColor());
   if (GetTextAlign() != 10) printf(" Align=%d",GetTextAlign());
   if (GetTextAngle() != 0 ) printf(" Angle=%f",GetTextAngle());
   printf("\n");
}

//______________________________________________________________________________
void TText::SavePrimitive(ofstream &out, Option_t *)
{
    // Save primitive as a C++ statement(s) on output stream out

   char quote = '"';
   if (gROOT->ClassSaved(TText::Class())) {
       out<<"   ";
   } else {
       out<<"   TText *";
   }
   out<<"text = new TText("<<fX<<","<<fY<<","<<quote<<GetTitle()<<quote<<");"<<endl;

   SaveTextAttributes(out,"text",11,0,1,62,1);

   out<<"   text->Draw();"<<endl;
}

//______________________________________________________________________________
void TText::SetNDC(Bool_t isNDC)
{
    // Set NDC mode on if isNDC = kTRUE, off otherwise
   ResetBit(kTextNDC);
   if (isNDC) SetBit(kTextNDC);
}

//______________________________________________________________________________
void TText::Streamer(TBuffer &R__b)
{
   // Stream an object of class TText.

   if (R__b.IsReading()) {
      UInt_t R__s, R__c;
      Version_t R__v = R__b.ReadVersion(&R__s, &R__c);
      if (R__v > 1) {
         TText::Class()->ReadBuffer(R__b, this, R__v, R__s, R__c);
         return;
      }
      //====process old versions before automatic schema evolution
      TNamed::Streamer(R__b);
      TAttText::Streamer(R__b);
      Float_t x,y;
      R__b >> x; fX = x;
      R__b >> y; fY = y;
      //====end of old versions
      
   } else {
      TText::Class()->WriteBuffer(R__b,this);
   }
}
